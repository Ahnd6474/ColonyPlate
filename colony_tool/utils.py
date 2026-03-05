import time

import cv2
import numpy as np


def now_ms() -> int:
    return int(time.time() * 1000)


def make_id(prefix: str = "d") -> str:
    return f"{prefix}_{now_ms()}_{np.random.randint(0, 1_000_000):06d}"


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
