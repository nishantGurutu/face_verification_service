# Face Verify Service (FastAPI)

Face alignment, embedding, verification and anti-spoofing microservice for the
Work Book HR face attendance flow.

## Stack
- FastAPI + uvicorn
- InsightFace `buffalo_l` (RetinaFace detection + ArcFace recognition), CPU via onnxruntime
- SQLite embedding store (`data/faces.db`)

## Setup
```powershell
cd face_verify_service
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```
First run auto-downloads the `buffalo_l` model (~280 MB) into
`C:\Users\<you>\.insightface\models`.

## Run
```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Listen on `0.0.0.0` so the Spring Boot backend (and LAN devices) can reach it.

## Endpoints
| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/health` | - | Model + store status |
| POST | `/api/faces/register` | multipart: `employeeId`, `name?`, `image` | Store employee embedding |
| POST | `/api/faces/verify` | multipart: `employeeId`, `image` | Verify face → `{matched, score, threshold, liveness}` |
| POST | `/api/faces/verify-two` | multipart: `imageA`, `imageB` | Compare two images |

## Tuning
- `FACE_SIMILARITY_THRESHOLD` (default `0.5`) — ArcFace cosine threshold.
- `FACE_MODEL_NAME` (default `buffalo_l`).
- Anti-spoof is heuristic (sharpness, detection score, face size). A hardened
  liveness model is a future upgrade; treat `liveness.passed` as a soft gate.

## Verify it works
```powershell
curl -X POST http://127.0.0.1:8000/api/faces/register -F "employeeId=emp1" -F "image=@photo.jpg"
curl -X POST http://127.0.0.1:8000/api/faces/verify -F "employeeId=emp1" -F "image=@selfie.jpg"
```
Same person → `score ≈ 0.9-1.0`; different person → `score < 0.3`.
