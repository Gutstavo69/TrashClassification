import cv2
import numpy as np

IMG_SIZE = 224


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_square(bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def prepare(bgr: np.ndarray) -> np.ndarray:
    return resize_square(bgr, IMG_SIZE)
