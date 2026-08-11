"""Face alignment using facial landmarks.

Aligns faces to a canonical position before embedding extraction.
Uses InsightFace's built-in landmark detection then applies similarity transform.
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical ArcFace alignment target (112x112, eyes + nose at standard positions)
_REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.6963],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.3655],   # right mouth
], dtype=np.float32)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, target_size: int = 112) -> np.ndarray:
    """Align face to canonical position using 5-point landmarks.

    Args:
        image_bgr: Input BGR image
        landmarks: 5x2 array of facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        target_size: Output image size (square)

    Returns:
        Aligned face image of shape (target_size, target_size, 3)
    """
    if landmarks.shape[0] < 5:
        raise ValueError(f"Need 5 landmarks for alignment, got {landmarks.shape[0]}")

    src = np.asarray(landmarks[:5], dtype=np.float32).reshape(5, 2)
    scale = target_size / 112.0
    dst = _REFERENCE_LANDMARKS * scale

    # Compute similarity transform from source landmarks to reference
    matrix = _estimate_similarity_transform(src, dst)

    aligned = cv2.warpAffine(
        image_bgr, matrix, (target_size, target_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


def align_face_from_insightface(face_obj, target_size: int = 112) -> np.ndarray:
    """Align face using InsightFace's landmark matrix.

    InsightFace stores a pre-computed transformation matrix in face.kps if available.
    Falls back to landmark-based alignment.
    """
    image = face_obj.image if hasattr(face_obj, 'image') else None
    if image is None:
        raise ValueError("Face object has no image reference")

    if hasattr(face_obj, 'landmark_3d_68') and face_obj.landmark_3d_68 is not None:
        return align_face(image, face_obj.landmark_3d_68, target_size)

    bbox = face_obj.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(image.shape[1], bbox[2]), min(image.shape[0], bbox[3])
    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise ValueError("Invalid face bounding box")
    return cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a 2D similarity transform (rotation + scale + translation) from src to dst.

    Returns 2x3 affine matrix.
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    src_std = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    dst_std = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))

    if src_std < 1e-9 or dst_std < 1e-9:
        scale = 1.0
        rotation = np.eye(2)
    else:
        scale = dst_std / src_std
        # Covariance matrix
        cov = src_centered.T @ dst_centered / src.shape[0]
        u, _, vt = np.linalg.svd(cov)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T

    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.hstack([scale * rotation, translation.reshape(2, 1)])
    return matrix
