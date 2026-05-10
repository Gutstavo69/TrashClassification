import cv2
import numpy as np
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops,
    hog,
)

from .preprocessing import prepare, to_gray, to_hsv

LBP_P = 24
LBP_R = 3
LBP_BINS = LBP_P + 2

GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

COLOR_BINS = 16


def color_histogram(bgr: np.ndarray) -> np.ndarray:
    feats = []
    for ch in range(3):
        h = cv2.calcHist([bgr], [ch], None, [COLOR_BINS], [0, 256]).flatten()
        h = h / (h.sum() + 1e-7)
        feats.append(h)

    hsv = to_hsv(bgr)
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, rng in enumerate(ranges):
        h = cv2.calcHist([hsv], [ch], None, [COLOR_BINS], list(rng)).flatten()
        h = h / (h.sum() + 1e-7)
        feats.append(h)

    return np.concatenate(feats)


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=False
    )
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-7)
    return hist


def glcm_features(gray: np.ndarray) -> np.ndarray:
    g = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        g,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=8,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        feats.append(vals.flatten())
    return np.concatenate(feats).astype(np.float32)


def hog_features(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_all(bgr: np.ndarray) -> np.ndarray:
    bgr = prepare(bgr)
    gray = to_gray(bgr)
    return np.concatenate(
        [
            color_histogram(bgr),
            lbp_histogram(gray),
            glcm_features(gray),
            hog_features(gray),
        ]
    ).astype(np.float32)


def feature_dim() -> int:
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    return extract_all(dummy).shape[0]
