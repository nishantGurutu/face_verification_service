import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import face_service, store, anti_spoof
import cv2

img = cv2.imread(r"D:\my-space\work-book-space\codnexis-app\dist\codnexis-app\browser\img\about-2.jpg")
if img is None:
    print("FAIL: could not read test image")
    sys.exit(1)

try:
    emb, det_score, fraction = face_service.extract_embedding(img)
    print(f"OK: face detected, dim={emb.shape[0]}, det_score={det_score:.4f}, fraction={fraction:.4f}")

    emb2, _, _ = face_service.extract_embedding(img)
    same_score = face_service.cosine_similarity(emb, emb2)
    print(f"Same-face self-comparison: {same_score:.4f} (should be ~1.0)")

    liveness = anti_spoof.assess(img, det_score, fraction)
    print(f"Liveness: {liveness}")
    print(f"Registered faces: {store.count_faces()}")

except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
