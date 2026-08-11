"""Liveness / anti-spoofing detection.

Uses multiple heuristics to distinguish real faces from:
- Printed photographs (low contrast, moire patterns, flat lighting)
- Phone screen replays (edge artifacts, pixel grid patterns)
- Replay videos (compression artifacts)

These are heuristics — not a hardened ML model. Scores should be used as soft gates.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .config import config


def sharpness(image_bgr: np.ndarray) -> float:
    """Laplacian variance — higher = sharper real face."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def texture_entropy(gray: np.ndarray) -> float:
    """Entropy of the grayscale histogram. Real faces have richer texture."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def moire_score(gray: np.ndarray) -> float:
    """Detect moire patterns from photographing a screen.

    Uses FFT-based frequency analysis.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    r = min(h, w) // 6

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (center_w, center_h), r, 1, -1)
    low_freq = np.sum(magnitude * mask) / np.sum(mask) if np.sum(mask) > 0 else 0
    high_freq = np.sum(magnitude * (1 - mask)) / np.sum(1 - mask) if np.sum(1 - mask) > 0 else 0

    if low_freq > 0:
        return float(high_freq / low_freq)
    return 0.0


def color_variance(image_bgr: np.ndarray) -> float:
    """Real faces have natural color variance. Screens/prints are flatter."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.std(hsv[:, :, 1])) / 256.0


def assess(image_bgr: np.ndarray, detection_score: float, face_fraction: float) -> Dict[str, object]:
    """Aggregate liveness heuristics.

    Returns dict with:
        passed: bool
        scores: dict of individual metric scores (0-1)
        overall: float (0-1)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sharp_val = sharpness(image_bgr)
    entropy = texture_entropy(gray)
    moire = moire_score(gray)
    color_var = color_variance(image_bgr)

    sharp_norm = min(1.0, sharp_val / 200.0)
    entropy_norm = min(1.0, entropy / 8.0)
    moire_norm = 1.0 - min(1.0, max(0.0, (moire - 0.3) / 0.7))
    color_norm = min(1.0, color_var / 0.15)

    overall = round(float(np.mean([sharp_norm, entropy_norm, moire_norm, color_norm])), 4)

    failed_reasons = []
    if sharp_val < config.liveness_sharpness_threshold:
        failed_reasons.append("blurry")
    if moire > config.liveness_replay_threshold:
        failed_reasons.append("suspected_replay")
    if entropy < 4.0:
        failed_reasons.append("low_texture")
    if detection_score < config.liveness_detection_threshold:
        failed_reasons.append("weak_detection")

    passed = len(failed_reasons) == 0

    return {
        "passed": passed,
        "overall": overall,
        "failed_reasons": failed_reasons,
        "scores": {
            "sharpness": round(sharp_norm, 4),
            "texture_entropy": round(entropy_norm, 4),
            "moire_resistance": round(moire_norm, 4),
            "color_variance": round(color_norm, 4),
        },
    }
