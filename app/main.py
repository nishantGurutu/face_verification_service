"""Face AI microservice — internal use by Spring Boot.

Full registration pipeline: detect → landmarks → align → head pose → mask →
quality → liveness → embed → validate.

Stateless. Spring Boot handles all storage via PostgreSQL + pgvector.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import anti_spoof, face_alignment, face_service, quality, store
from .config import config

app = FastAPI(
    title="Work Book Face AI",
    version="2.1.0",
    description="ArcFace embedding + strict face validation pipeline.",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_NAME = "buffalo_l"
MODEL_VERSION = "v1"
EMBEDDING_DIM = 512


@app.on_event("startup")
def _warmup():
    try:
        face_service.ensure_loaded()
        print("[face-ai] model loaded:", face_service.ensure_loaded())
    except Exception as exc:
        print("[face-ai] model load failed:", exc)


@app.get("/health")
def health():
    try:
        detail = face_service.ensure_loaded()
        return {"status": "ok", "model": detail}
    except Exception as exc:
        return {"status": "model_unavailable", "error": str(exc)}


# ── Internal: Full Registration Pipeline ──────────────────────────────────


@app.post("/internal/v1/face/register")
async def register_face_internal(image: UploadFile = File(...)) -> dict:
    """STRICT face registration pipeline.

    Pipeline: decode → detect → exactly-one → landmarks → head-pose →
    mask-check → quality → liveness → alignment → embedding → validate.

    Fails fast — ANY validation failure stops the pipeline with a specific error code.
    """
    data = await image.read()

    # 1. Decode
    try:
        bgr = face_service.decode_image(data)
    except ValueError:
        return {"success": False, "code": "INVALID_IMAGE", "message": "Could not decode image bytes"}

    h, w = bgr.shape[:2]

    # 2. Face detection
    faces = face_service.detect_faces(bgr)
    if not faces:
        return {"success": False, "code": "NO_FACE", "message": "No face detected", "faceCount": 0}
    if len(faces) > 1:
        return {"success": False, "code": "MULTIPLE_FACES", "message": f"Detected {len(faces)} faces — only one allowed", "faceCount": len(faces)}

    face = face_service.get_largest_face(faces)
    info = face_service.extract_face_info(bgr, face)

    # 3. Detection confidence
    if info["det_score"] < config.min_detection_score:
        return {"success": False, "code": "FACE_DETECTION_WEAK", "message": "Face detection confidence too low", "detail": {"det_score": info["det_score"], "threshold": config.min_detection_score}}
    if info["det_score"] < config.min_detection_score:
        return {"success": False, "code": "FACE_DETECTION_WEAK", "message": f"Detection score {info['det_score']:.2f} below threshold {config.min_detection_score}"}

    # 4. Face size
    fraction = info["fraction"]
    if fraction < config.min_face_ratio:
        return {"success": False, "code": "FACE_TOO_SMALL", "message": "Face too small — move closer"}
    if fraction > config.max_face_ratio:
        return {"success": False, "code": "FACE_TOO_LARGE", "message": "Face too large — move back"}

    # 5. Head pose
    yaw, pitch, roll = info["yaw"], info["pitch"], info["roll"]
    if abs(yaw) > config.max_yaw:
        return {"success": False, "code": "HEAD_TURNED_LEFT" if yaw > 0 else "HEAD_TURNED_RIGHT", "message": "Face turned too far — look straight"}
    if abs(pitch) > config.max_pitch:
        return {"success": False, "code": "HEAD_TILTED_UP" if pitch > 0 else "HEAD_TILTED_DOWN", "message": "Head tilted — keep level"}
    if abs(roll) > config.max_roll:
        return {"success": False, "code": "HEAD_LEANING", "message": "Head leaning — straighten up"}

    # 6. Mask / occlusion
    mask_risk = info["mask_risk"]
    if mask_risk > config.occlusion_threshold:
        return {"success": False, "code": "MASK_DETECTED" if mask_risk > 0.7 else "FACE_OCCLUDED", "message": "Face covered — remove mask or obstruction", "detail": {"mask_risk": mask_risk, "threshold": config.occlusion_threshold}}

    # 7. Quality assessment
    x1, y1, x2, y2 = info["bbox"]
    q = quality.assess_quality(bgr, (x1, y1, x2, y2), info["det_score"], fraction, yaw, pitch, roll)
    if not q["acceptable"]:
        code_map = {
            "blurry": "FACE_BLURRY", "too_dark": "FACE_TOO_DARK", "too_bright": "FACE_TOO_BRIGHT",
            "low_resolution": "FACE_RESOLUTION_TOO_LOW", "poor_pose": "FACE_POOR_ALIGNMENT",
            "face_too_small": "FACE_TOO_SMALL", "face_too_large": "FACE_TOO_LARGE",
            "weak_detection": "FACE_DETECTION_WEAK",
        }
        code = code_map.get(q["rejection_reason"], "LOW_QUALITY")
        return {"success": False, "code": code, "message": f"Image quality insufficient: {q['rejection_reason']}", "detail": q}

    # 8. Liveness
    liveness = anti_spoof.assess(bgr, info["det_score"], fraction)
    if not liveness["passed"]:
        return {"success": False, "code": "LIVENESS_FAILED", "message": "Liveness check failed", "detail": {"failed_reasons": liveness["failed_reasons"], "scores": liveness["scores"]}}

    # 9. Face alignment
    try:
        aligned = face_alignment.align_face(bgr, np.array(info["landmarks"]))
    except Exception:
        # Fallback: use bounding box crop
        aligned = face_alignment.align_face_from_insightface(face) if hasattr(face, 'image') else bgr[int(y1):int(y2), int(x1):int(x2)]

    # 10. Re-embed from aligned face for best quality
    try:
        faces_aligned = face_service.detect_faces(aligned)
        if faces_aligned:
            emb = np.asarray(faces_aligned[0].normed_embedding, dtype=np.float32).reshape(-1)
        else:
            emb = info["embedding"]
    except Exception:
        emb = info["embedding"]

    # 11. Embedding validation
    if emb.shape[0] != EMBEDDING_DIM:
        return {"success": False, "code": "INVALID_EMBEDDING", "message": f"Embedding dimension {emb.shape[0]} != {EMBEDDING_DIM}"}

    emb_norm = float(np.linalg.norm(emb))
    if emb_norm < 1e-9:
        return {"success": False, "code": "INVALID_EMBEDDING", "message": "Embedding is zero vector"}
    emb_normalized = emb / emb_norm

    emb_list = [float(x) for x in emb_normalized]

    return {
        "success": True,
        "code": "FACE_VALID",
        "qualityScore": round(q["overall"], 4),
        "qualityDetail": q,
        "livenessScore": round(liveness["overall"], 4),
        "livenessPassed": liveness["passed"],
        "livenessDetail": liveness,
        "headPose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
        "faceCount": 1,
        "maskRisk": round(mask_risk, 3),
        "modelName": MODEL_NAME,
        "modelVersion": MODEL_VERSION,
        "embeddingDimension": EMBEDDING_DIM,
        "embedding": emb_list,
        "message": "Face registered successfully",
    }


# ── Internal: Verify (probe embedding only) ───────────────────────────────


@app.post("/internal/v1/face/verify")
async def verify_face_internal(image: UploadFile = File(...)) -> dict:
    """Generate embedding for probe image. Spring Boot does cosine similarity."""
    data = await image.read()
    try:
        bgr = face_service.decode_image(data)
    except ValueError:
        return {"success": False, "reason": "INVALID_IMAGE", "message": "Could not decode image"}

    faces = face_service.detect_faces(bgr)
    if not faces:
        return {"success": False, "reason": "NO_FACE", "message": "No face detected", "faceCount": 0}

    info = face_service.extract_face_info(bgr, face_service.get_largest_face(faces))
    emb = info["embedding"]
    emb_norm = float(np.linalg.norm(emb))
    if emb_norm > 1e-9:
        emb = emb / emb_norm

    liveness = anti_spoof.assess(bgr, info["det_score"], info["fraction"])
    q = quality.assess_quality(bgr, info["bbox"], info["det_score"], info["fraction"])

    return {
        "success": True,
        "embedding": [float(x) for x in emb],
        "faceCount": 1,
        "qualityScore": round(q["overall"], 4),
        "livenessScore": round(liveness["overall"], 4),
    }


# ── Legacy endpoints ──────────────────────────────────────────────────────


@app.get("/api/faces/{employee_id}")
def face_status(employee_id: str):
    registered = store.get_embedding(employee_id) is not None
    return {"employeeId": employee_id, "registered": registered}


@app.post("/api/faces/register")
async def register_face_legacy(
    employeeId: str = Form(...), image: UploadFile = File(...), name: Optional[str] = Form(default=None),
):
    data = await image.read()
    try:
        bgr = face_service.decode_image(data)
        embedding, det_score, fraction = face_service.extract_embedding(bgr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    store.upsert_embedding(employee_id=employeeId.strip(), name=name, embedding=embedding)
    return {"status": "registered", "employeeId": employeeId.strip()}


@app.post("/api/faces/verify")
async def verify_face_legacy(employeeId: str = Form(...), image: UploadFile = File(...)):
    registered = store.get_embedding(employeeId.strip())
    if registered is None:
        raise HTTPException(status_code=404, detail="No registered face")
    _, stored_embedding = registered
    data = await image.read()
    try:
        bgr = face_service.decode_image(data)
        embedding, _, _ = face_service.extract_embedding(bgr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    score = face_service.cosine_similarity(embedding, stored_embedding)
    return {"employeeId": employeeId.strip(), "matched": score >= 0.5, "score": round(score, 4)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
