import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from colony_tool.models import Det
from colony_tool.utils import clamp, make_id, to_gray


def detect_plate_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 4,
        param1=120,
        param2=35,
        minRadius=int(min(gray.shape[:2]) * 0.30),
        maxRadius=int(min(gray.shape[:2]) * 0.55),
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    cx, cy, r = circles[0]
    return (cx, cy, r)


def is_near_edge_ring(cx: float, cy: float, roi_circle: Optional[Tuple[int, int, int]], ring_ratio: float = 0.06) -> bool:
    if roi_circle is None:
        return False
    x0, y0, r = roi_circle
    d = math.hypot(cx - x0, cy - y0)
    return d > (1.0 - ring_ratio) * r


def compute_feat(gray: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> List[float]:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1i = int(clamp(round(x1), 0, w - 1))
    y1i = int(clamp(round(y1), 0, h - 1))
    x2i = int(clamp(round(x2), 0, w - 1))
    y2i = int(clamp(round(y2), 0, h - 1))
    if x2i <= x1i or y2i <= y1i:
        return [0.0, 0.0, 0.0, 0.0]

    crop = gray[y1i:y2i, x1i:x2i]
    area = float((x2 - x1) * (y2 - y1) + 1e-6)
    aspect = float((x2 - x1 + 1e-6) / (y2 - y1 + 1e-6))
    mean = float(crop.mean()) / 255.0
    std = float(crop.std()) / 255.0
    return [math.log(area), aspect, mean, std]


def run_yolo_detect(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    max_det: int = 1500,
) -> List[Det]:
    res = model.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)[0]
    if res.boxes is None:
        return []

    boxes = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    gray = to_gray(img_bgr)
    dets: List[Det] = []
    for (x1, y1, x2, y2), c, cf in zip(boxes, clss, confs):
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        feat = compute_feat(gray, (float(x1), float(y1), float(x2), float(y2)))
        dets.append(
            Det(
                det_id=make_id("det"),
                cls=int(c),
                conf=float(cf),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                centroid_xy=(cx, cy),
                feat=feat,
            )
        )
    return dets


def filter_dets(
    dets: List[Det],
    img_shape: Tuple[int, int, int],
    roi_circle: Optional[Tuple[int, int, int]] = None,
    min_area_px: int = 30,
    max_area_px: int = 500000,
    min_conf: float = 0.25,
    drop_edge_ring: bool = True,
) -> List[Det]:
    h, w = img_shape[:2]
    out: List[Det] = []
    for d in dets:
        if d.conf < min_conf:
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        area = (x2 - x1) * (y2 - y1)
        if area < min_area_px or area > max_area_px:
            continue
        cx, cy = d.centroid_xy
        if drop_edge_ring and roi_circle is not None and is_near_edge_ring(cx, cy, roi_circle):
            continue

        x1 = float(clamp(x1, 0, w - 1))
        x2 = float(clamp(x2, 0, w - 1))
        y1 = float(clamp(y1, 0, h - 1))
        y2 = float(clamp(y2, 0, h - 1))
        d.bbox_xyxy = (x1, y1, x2, y2)
        d.centroid_xy = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        out.append(d)

    out.sort(key=lambda z: (z.centroid_xy[1], z.centroid_xy[0]))
    return out
