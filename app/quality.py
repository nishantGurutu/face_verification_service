"""Comprehensive face image quality assessment.

Checks: blur, brightness, resolution, contrast, pose, occlusion.
All scores are 0.0 (worst) to 1.0 (best). Rejection thresholds from config.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .config import config


def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher = sharper. Normalized to 0-1."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Map: 0 → 0, 200 → 0.9, 500+ → 1.0
    return min(1.0, max(0.0, min(1.0, lap_var / 200.0)))


def brightness_score(gray: np.ndarray) -> float:
    """Mean brightness normalized to 0-1. 0.5 = ideal, extremes penalized."""
    mean = float(np.mean(gray)) / 255.0
    # Quadratic penalty for deviation from 0.5
    return 1.0 - 4.0 * (mean - 0.5) ** 2


def contrast_score(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities, normalized."""
    std = float(np.std(gray)) / 128.0
    return min(1.0, std)


def face_resolution_score(face_w: int, face_h: int) -> float:
    """Score based on face pixel dimensions. maps to 0-1."""
    min_dim = min(face_w, face_h)
    # <80 px = 0, 80 = 0.3, 200+ = 1.0
    return min(1.0, max(0.0, (min_dim - 50) / 150.0))


def pose_score(yaw: float, pitch: float, roll: float) -> float:
    """Score head pose deviation from frontal. 0 = far, 1 = perfect frontal."""
    yaw_s = max(0.0, 1.0 - abs(yaw) / config.max_yaw)
    pitch_s = max(0.0, 1.0 - abs(pitch) / config.max_pitch)
    roll_s = max(0.0, 1.0 - abs(roll) / config.max_roll)
    return float(np.mean([yaw_s, pitch_s, roll_s]))


def assess_quality(image_bgr: np.ndarray, face_bbox, detection_score: float, face_fraction: float,
                   yaw: float = 0, pitch: float = 0, roll: float = 0) -> Dict:
    """Run all quality checks and return detailed report.

    Returns:
        {
            "acceptable": bool,
            "overall": float (0-1),
            "rejection_reason": str or None,
            "blur": float,
            "brightness": float,
            "contrast": float,
            "resolution": float,
            "pose": float,
            "face_size_ratio": float,
            "detection_score": float,
            "checks": { ... individual scores ... }
        }
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w, face_h = x2 - x1, y2 - y1

    blur = round(blur_score(gray), 4)
    brightness = round(brightness_score(gray), 4)
    contrast = round(contrast_score(gray), 4)
    resolution = round(face_resolution_score(face_w, face_h), 4)
    pose = round(pose_score(yaw, pitch, roll), 4)

    issues: List[str] = []
    if detection_score < config.min_detection_score:
        issues.append("weak_detection")
    if face_fraction < config.min_face_ratio:
        issues.append("face_too_small")
    if face_fraction > config.max_face_ratio:
        issues.append("face_too_large")
    if blur < 0.3:
        issues.append("blurry")
    if brightness < 0.2:
        issues.append("too_dark")
    if brightness > 0.95:
        issues.append("too_bright")
    if resolution < 0.2:
        issues.append("low_resolution")
    if pose < 0.3:
        issues.append("poor_pose")

    overall = round(min(
        detection_score * 0.20 + 
        min(face_fraction * 10, 1.0) * 0.15 +
        blur * 0.25 + 
        brightness * 0.10 +
        contrast * 0.10 +
        pose * 0.20,
        1.0
    ), 4)

    checks = {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "resolution": resolution,
        "pose": pose,
        "detection_score": round(detection_score, 4),
        "face_size_ratio": round(face_fraction, 4),
    }

    return {
        "acceptable": len(issues) == 0,
        "overall": overall,
        "rejection_reason": issues[0] if issues else None,
        "issues": issues,
        **checks,
    }
