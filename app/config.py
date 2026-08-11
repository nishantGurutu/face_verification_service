"""Configurable thresholds for face registration validation.

All values can be overridden via environment variables.
"""
from __future__ import annotations
import os


def _env(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


class FaceRegistrationConfig:
    # Face detection
    min_detection_score: float = _env("FACE_MIN_DET_SCORE", 0.6)

    # Face size (fraction of frame area)
    min_face_ratio: float = _env("FACE_MIN_RATIO", 0.06)
    max_face_ratio: float = _env("FACE_MAX_RATIO", 0.65)

    # Head pose (degrees)
    max_yaw: float = _env("FACE_MAX_YAW", 25.0)
    max_pitch: float = _env("FACE_MAX_PITCH", 20.0)
    max_roll: float = _env("FACE_MAX_ROLL", 20.0)

    # Quality
    min_blur_sharpness: float = _env("FACE_MIN_SHARPNESS", 60.0)
    min_brightness: float = _env("FACE_MIN_BRIGHTNESS", 0.15)
    max_brightness: float = _env("FACE_MAX_BRIGHTNESS", 0.92)
    min_face_resolution: int = int(os.environ.get("FACE_MIN_RESOLUTION", "80"))

    # Mask / occlusion
    occlusion_threshold: float = _env("FACE_OCCLUSION_THRESHOLD", 0.40)

    # Liveness
    liveness_detection_threshold: float = _env("LIVENESS_DET_THRESHOLD", 0.5)
    liveness_sharpness_threshold: float = _env("LIVENESS_SHARP_THRESHOLD", 50.0)
    liveness_replay_threshold: float = _env("LIVENESS_REPLAY_THRESHOLD", 0.6)

    # Best frame
    best_frame_window: int = int(os.environ.get("BEST_FRAME_WINDOW", "5"))


config = FaceRegistrationConfig()
