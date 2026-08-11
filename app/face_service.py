"""Face detection, landmark analysis, head pose, embedding — InsightFace ArcFace.

The model bundle (buffalo_l) is auto-downloaded by InsightFace on first use.
Detection + recognition run on CPU via onnxruntime.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.environ.setdefault("INSIGHTFACE_HOME", MODEL_DIR)

MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "buffalo_l")
EMBEDDING_DIM = 512

_app = None


def _load_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
    return _app


def ensure_loaded() -> str:
    app = _load_app()
    names = list(getattr(app, "models", {}).keys()) or [MODEL_NAME]
    return f"insightface/{MODEL_NAME} -> {names}"


def detect_faces(image_bgr: np.ndarray) -> List:
    """Return all detected faces from InsightFace. Each face has: bbox, kps, det_score, embedding, landmark_3d_68."""
    app = _load_app()
    return app.get(image_bgr)


def get_largest_face(faces: List):
    """Return the largest face by bounding box area."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def extract_face_info(image_bgr: np.ndarray, face) -> Dict:
    """Extract comprehensive face info from a detected face.

    Returns: { embedding, det_score, fraction, landmarks, bbox, head_pose, mask_risk, aligned_image }
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(float)
    bbox_area = (x2 - x1) * (y2 - y1)
    fraction = bbox_area / float(w * h) if w * h else 0.0
    embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1) if hasattr(face, 'normed_embedding') and face.normed_embedding is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32)

    landmarks = face.kps.astype(float).tolist() if hasattr(face, 'kps') and face.kps is not None else []
    landmark_3d = face.landmark_3d_68.astype(float).tolist() if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None else []

    yaw, pitch, roll = estimate_head_pose_from_landmarks(landmarks) if len(landmarks) >= 5 else (0.0, 0.0, 0.0)

    mask_risk = estimate_mask_risk(face) if landmarks else 0.0

    return {
        "embedding": embedding,
        "det_score": float(face.det_score),
        "fraction": float(fraction),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "landmarks": landmarks,
        "landmark_3d": landmark_3d,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "mask_risk": float(mask_risk),
    }


def extract_embedding(image_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Legacy: detect largest face, return embedding + detection score + fraction."""
    faces = detect_faces(image_bgr)
    if not faces:
        raise ValueError("No face detected in image")
    face = get_largest_face(faces)
    info = extract_face_info(image_bgr, face)
    return info["embedding"], info["det_score"], info["fraction"]


def estimate_head_pose_from_landmarks(landmarks) -> Tuple[float, float, float]:
    """Estimate yaw, pitch, roll from 5+ facial landmarks using a 3D head model.

    Returns (yaw_deg, pitch_deg, roll_deg).
    """
    if len(landmarks) < 5:
        return 0.0, 0.0, 0.0

    # 3D model points (mm) for: left_eye, right_eye, nose, left_mouth, right_mouth
    model_points = np.array([
        [-30.0, -30.0, -30.0],   # left eye
        [ 30.0, -30.0, -30.0],   # right eye
        [  0.0,   0.0,   0.0],   # nose
        [-20.0,  30.0, -30.0],   # left mouth
        [ 20.0,  30.0, -30.0],   # right mouth
    ], dtype=np.float64)

    image_points = np.array(landmarks[:5], dtype=np.float64).reshape(5, 2)

    focal_length = 1.0
    center = (0.0, 0.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rvec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
            pitch = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        else:
            yaw = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            roll = 0.0

        return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))
    except cv2.error:
        return 0.0, 0.0, 0.0


def estimate_mask_risk(face) -> float:
    """Estimate mask/occlusion risk from InsightFace face object.

    Uses insightface's built-in 'mask' classification if available (buffalo_l has it).
    Falls back to landmark-based heuristic.

    Returns 0.0 (no mask) to 1.0 (definitely masked).
    """
    # InsightFace buffalo_l model includes mask detection in some versions
    if hasattr(face, 'mask') and face.mask is not None:
        return float(face.mask)

    # Landmark-based heuristic: check lower face landmarks
    if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
        lm = face.landmark_3d_68
        # Lower face landmarks: 48-67 (mouth + jaw)
        if len(lm) >= 68:
            lower = lm[48:68]
            # If many lower landmarks have low confidence or are clustered,
            # it suggests occlusion
            spread = float(np.std(lower))
            if spread < 3.0:
                return 0.7
            return 0.0

    if hasattr(face, 'kps') and face.kps is not None:
        kps = face.kps
        if len(kps) >= 5:
            # Landmarks 3=nose, 4=left_mouth, 5=right_mouth (InsightFace order is typically left_eye, right_eye, nose, left_mouth, right_mouth)
            nose = np.array(kps[2])
            mouth_left = np.array(kps[3])
            mouth_right = np.array(kps[4])
            mouth_center = (mouth_left + mouth_right) / 2.0
            nose_to_mouth = float(np.linalg.norm(mouth_center - nose))
            eye_dist = float(np.linalg.norm(np.array(kps[0]) - np.array(kps[1])))
            if eye_dist > 0:
                ratio = nose_to_mouth / eye_dist
                if ratio < 0.2:
                    return 0.9
                elif ratio < 0.4:
                    return 0.5
        return 0.0

    return 0.0


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
