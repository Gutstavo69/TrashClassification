"""Image augmentation per proposal Section 3.1.

random flip, rotation (+/-20 deg), zoom (10%), brightness shift -- training set only.
"""
from __future__ import annotations

import cv2
import numpy as np

ROTATION_RANGE_DEG = 20.0
ZOOM_RANGE = 0.10
BRIGHTNESS_RANGE = 30


def horizontal_flip(bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(bgr, 1)


def rotate(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT
    )


def zoom(bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    if factor >= 1.0:
        new_h, new_w = max(1, int(h / factor)), max(1, int(w / factor))
        y0, x0 = (h - new_h) // 2, (w - new_w) // 2
        crop = bgr[y0 : y0 + new_h, x0 : x0 + new_w]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    scaled_w, scaled_h = max(1, int(w * factor)), max(1, int(h * factor))
    scaled = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(bgr)
    y0, x0 = (h - scaled_h) // 2, (w - scaled_w) // 2
    out[y0 : y0 + scaled_h, x0 : x0 + scaled_w] = scaled
    return out


def brightness_shift(bgr: np.ndarray, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(bgr, alpha=1.0, beta=beta)


def augment_variants(bgr: np.ndarray, seed: int) -> list[np.ndarray]:
    """Four deterministic augmented variants of `bgr`: flip, rotate, zoom, brightness."""
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG))
    factor = float(rng.uniform(1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE))
    beta = int(rng.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE))
    return [
        horizontal_flip(bgr),
        rotate(bgr, angle),
        zoom(bgr, factor),
        brightness_shift(bgr, beta),
    ]
