import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Ultralytics (YOLOv11 detect weights supported if your ultralytics is recent enough)
from ultralytics import YOLO

import gradio as gr


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Det:
    det_id: str
    cls: int
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2
    centroid_xy: Tuple[float, float]              # cx,cy
    feat: Optional[List[float]] = None            # simple feature vector for label recall


@dataclass
class Session:
    image_name: str
    image_size: Tuple[int, int]  # (W,H)
    created_ms: int
    dets: List[Det]
    labels: Dict[str, str]       # det_id -> label


# -----------------------------
# Utils
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def make_id(prefix="d") -> str:
    return f"{prefix}_{now_ms()}_{np.random.randint(0, 1_000_000):06d}"

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# -----------------------------
# Optional ROI masking (not strict "plate recognition")
# -----------------------------
def detect_plate_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Rough circular ROI (Hough). Used ONLY to drop background/outside and reduce edge reflection noise.
    Returns (cx, cy, r) or None.
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] // 4,
        param1=120, param2=35,
        minRadius=int(min(gray.shape[:2]) * 0.30),
        maxRadius=int(min(gray.shape[:2]) * 0.55)
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    cx, cy, r = circles[0]
    return (cx, cy, r)

def is_near_edge_ring(cx, cy, roi_circle, ring_ratio=0.06) -> bool:
    """Reject detections too close to plate rim (reflection ring)."""
    if roi_circle is None:
        return False
    x0, y0, r = roi_circle
    d = math.hypot(cx - x0, cy - y0)
    return d > (1.0 - ring_ratio) * r


# -----------------------------
# YOLOv11 Detect inference -> Det list
# -----------------------------
def compute_feat(gray: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> List[float]:
    """
    Simple feature vector for "label recall across sessions":
      [log(area), aspect, mean, std]
    """
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
        dets.append(Det(
            det_id=make_id("det"),
            cls=int(c),
            conf=float(cf),
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            centroid_xy=(cx, cy),
            feat=feat
        ))
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
    H, W = img_shape[:2]
    out = []
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
        # clip bbox into image bounds (for display safety)
        x1 = float(clamp(x1, 0, W - 1))
        x2 = float(clamp(x2, 0, W - 1))
        y1 = float(clamp(y1, 0, H - 1))
        y2 = float(clamp(y2, 0, H - 1))
        d.bbox_xyxy = (x1, y1, x2, y2)
        d.centroid_xy = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        out.append(d)
    # stable order for UX
    out.sort(key=lambda z: (z.centroid_xy[1], z.centroid_xy[0]))
    return out


# -----------------------------
# Drawing
# -----------------------------
def draw_overlay(img_bgr: np.ndarray, dets: List[Det], labels: Dict[str, str], selected_id: Optional[str] = None) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, map(round, d.bbox_xyxy))
        color = (0, 255, 255) if d.det_id != selected_id else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cx, cy = map(int, map(round, d.centroid_xy))
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

        tag = labels.get(d.det_id, "")
        if tag:
            cv2.putText(out, tag, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -----------------------------
# Click selection
# -----------------------------
def pick_det_by_click(dets: List[Det], x: float, y: float) -> Optional[Det]:
    inside = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(d)
    if inside:
        return min(inside, key=lambda dd: (dd.centroid_xy[0] - x) ** 2 + (dd.centroid_xy[1] - y) ** 2)
    if not dets:
        return None
    return min(dets, key=lambda dd: (dd.centroid_xy[0] - x) ** 2 + (dd.centroid_xy[1] - y) ** 2)


# -----------------------------
# Similarity transform (rotate+scale+translate) for remapping labels across images
# -----------------------------
def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    h = np.hstack([pts.astype(np.float64), ones])
    out = (T @ h.T).T
    return out[:, :2]

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    if n < 2:
        return np.eye(3, dtype=np.float64), float("inf")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean

    cov = (dst_d.T @ src_d) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_src = (src_d ** 2).sum() / n
    s = 1.0 if var_src < 1e-12 else float(S.sum() / var_src)
    t = dst_mean - s * (R @ src_mean)

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = s * R
    T[:2, 2] = t

    pred = apply_T(src, T)
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return T, rmse

def ransac_similarity(old_pts: np.ndarray, new_pts: np.ndarray,
                      iters: int = 400, nn_thresh: float = 18.0, min_inliers: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    if old_pts.shape[0] < 2 or new_pts.shape[0] < 2:
        return np.eye(3, dtype=np.float64), np.zeros((old_pts.shape[0],), dtype=bool), 0.0

    best_T = np.eye(3, dtype=np.float64)
    best_inliers = np.zeros((old_pts.shape[0],), dtype=bool)
    best_cnt = 0
    best_score = float("inf")

    for _ in range(iters):
        i1, i2 = rng.choice(old_pts.shape[0], size=2, replace=False)
        j1, j2 = rng.choice(new_pts.shape[0], size=2, replace=False)

        src = old_pts[[i1, i2]]
        dst = new_pts[[j1, j2]]
        T, _ = umeyama_similarity(src, dst)

        warped = apply_T(old_pts, T)
        d2 = ((warped[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2)
        nn = np.sqrt(d2.min(axis=1))
        inliers = nn < nn_thresh
        cnt = int(inliers.sum())
        score = float(nn[inliers].mean()) if cnt > 0 else float("inf")

        if cnt > best_cnt or (cnt == best_cnt and score < best_score):
            best_cnt = cnt
            best_score = score
            best_T = T
            best_inliers = inliers

    # refine with induced NN correspondences
    if best_cnt >= min_inliers:
        warped = apply_T(old_pts, best_T)
        d2 = ((warped[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2)
        nn_idx = d2.argmin(axis=1)
        src = old_pts[best_inliers]
        dst = new_pts[nn_idx[best_inliers]]
        best_T, _ = umeyama_similarity(src, dst)

    return best_T, best_inliers, (0.0 if best_cnt == 0 else float(best_score))

def match_labels_hungarian(pred_pts: np.ndarray, new_pts: np.ndarray, max_dist: float = 25.0) -> List[int]:
    if pred_pts.shape[0] == 0 or new_pts.shape[0] == 0:
        return [-1] * pred_pts.shape[0]

    d = np.sqrt(((pred_pts[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2))
    cost = d.copy()
    cost[d > max_dist] = 1e6
    r, c = linear_sum_assignment(cost)

    out = [-1] * pred_pts.shape[0]
    for ri, ci in zip(r, c):
        if cost[ri, ci] < 1e6:
            out[ri] = int(ci)
    return out


# -----------------------------
# Session I/O
# -----------------------------
def save_session(path: Path, sess: Session):
    data = {
        "image_name": sess.image_name,
        "image_size": list(sess.image_size),
        "created_ms": sess.created_ms,
        "dets": [asdict(d) for d in sess.dets],
        "labels": sess.labels
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_session(path: Path) -> Session:
    data = json.loads(path.read_text(encoding="utf-8"))
    dets = []
    for dd in data["dets"]:
        dets.append(Det(
            det_id=dd["det_id"],
            cls=int(dd["cls"]),
            conf=float(dd["conf"]),
            bbox_xyxy=tuple(dd["bbox_xyxy"]),
            centroid_xy=tuple(dd["centroid_xy"]),
            feat=dd.get("feat", None)
        ))
    return Session(
        image_name=data["image_name"],
        image_size=tuple(data["image_size"]),
        created_ms=int(data["created_ms"]),
        dets=dets,
        labels=dict(data.get("labels", {}))
    )


# -----------------------------
# Label library (for "reuse old named colony")
# -----------------------------
def lib_path(sessions_dir: Path) -> Path:
    return sessions_dir / "label_library.json"

def load_library(sessions_dir: Path) -> Dict[str, List[List[float]]]:
    p = lib_path(sessions_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_library(sessions_dir: Path, lib: Dict[str, List[List[float]]]):
    p = lib_path(sessions_dir)
    p.write_text(json.dumps(lib, ensure_ascii=False, indent=2), encoding="utf-8")

def update_library_from_session(sessions_dir: Path, sess: Session):
    lib = load_library(sessions_dir)
    id2det = {d.det_id: d for d in sess.dets}
    for det_id, label in sess.labels.items():
        det = id2det.get(det_id)
        if det is None or det.feat is None:
            continue
        lib.setdefault(label, []).append(det.feat)
        # keep last N examples per label
        if len(lib[label]) > 200:
            lib[label] = lib[label][-200:]
    save_library(sessions_dir, lib)

def suggest_labels(lib: Dict[str, List[List[float]]], feat: List[float], topk: int = 5) -> List[Tuple[str, float]]:
    """
    Return (label, score) sorted by increasing distance.
    """
    if not lib or feat is None:
        return []
    q = np.array(feat, dtype=np.float64)
    out = []
    for label, feats in lib.items():
        X = np.array(feats, dtype=np.float64)
        if X.size == 0:
            continue
        d = np.linalg.norm(X - q[None, :], axis=1)
        out.append((label, float(d.min())))
    out.sort(key=lambda x: x[1])
    return out[:topk]


# -----------------------------
# Auto pick best previous session for current image
# -----------------------------
def score_session_for_current(prev: Session, cur_dets: List[Det],
                              ransac_iters=400, nn_thresh=18.0, max_match_dist=25.0):
    old_pts = np.array([d.centroid_xy for d in prev.dets], dtype=np.float64)
    new_pts = np.array([d.centroid_xy for d in cur_dets], dtype=np.float64)
    T, inliers, score = ransac_similarity(old_pts, new_pts, iters=ransac_iters, nn_thresh=nn_thresh, min_inliers=8)

    labeled_old = [d for d in prev.dets if d.det_id in prev.labels]
    if not labeled_old:
        return None

    labeled_old_pts = np.array([d.centroid_xy for d in labeled_old], dtype=np.float64)
    pred_pts = apply_T(labeled_old_pts, T)
    matches = match_labels_hungarian(pred_pts, new_pts, max_dist=max_match_dist)

    found = sum(1 for m in matches if m != -1)
    # score: prioritize found count, then ransac avg residual
    return {
        "T": T, "found": found, "total": len(labeled_old),
        "ratio": (found / max(1, len(labeled_old))),
        "score": score
    }

def remap_labels(prev: Session, cur_dets: List[Det],
                 ransac_iters=400, nn_thresh=18.0, max_match_dist=25.0):
    old_pts = np.array([d.centroid_xy for d in prev.dets], dtype=np.float64)
    new_pts = np.array([d.centroid_xy for d in cur_dets], dtype=np.float64)
    T, _, score = ransac_similarity(old_pts, new_pts, iters=ransac_iters, nn_thresh=nn_thresh, min_inliers=8)

    labeled_old = [d for d in prev.dets if d.det_id in prev.labels]
    labeled_old_pts = np.array([d.centroid_xy for d in labeled_old], dtype=np.float64)
    pred_pts = apply_T(labeled_old_pts, T)

    matches = match_labels_hungarian(pred_pts, new_pts, max_dist=max_match_dist)

    remapped: Dict[str, str] = {}
    missing = 0
    for i, mi in enumerate(matches):
        label = prev.labels[labeled_old[i].det_id]
        if mi == -1:
            missing += 1
            continue
        cur_id = cur_dets[mi].det_id
        remapped[cur_id] = label

    return remapped, T, score, missing, len(labeled_old)


# -----------------------------
# Gradio App
# -----------------------------
def build_app(weights: str, sessions_dir: str):
    sessions_dir = Path(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    lib = load_library(sessions_dir)

    # state: current
    state = {
        "img_bgr": None,
        "img_name": "",
        "roi_circle": None,
        "dets": [],
        "labels": {},
        "selected_id": None,
        "last_session_path": None,
        "lib": lib,
    }

    def _render():
        if state["img_bgr"] is None:
            return None
        over = draw_overlay(state["img_bgr"], state["dets"], state["labels"], state["selected_id"])
        return to_rgb(over)

    def load_image(file, conf, iou, imgsz, min_area, max_area, use_roi, drop_ring):
        if file is None:
            return None, gr.update(value="(no image)"), gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")

        img_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, gr.update(value="(read failed)"), gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")

        state["img_bgr"] = img_bgr
        state["img_name"] = Path(file).name

        roi = detect_plate_roi(to_gray(img_bgr)) if use_roi else None
        state["roi_circle"] = roi

        dets = run_yolo_detect(model, img_bgr, conf=conf, iou=iou, imgsz=int(imgsz))
        dets = filter_dets(dets, img_bgr.shape, roi_circle=roi,
                           min_area_px=int(min_area), max_area_px=int(max_area),
                           min_conf=float(conf), drop_edge_ring=bool(drop_ring))
        state["dets"] = dets
        state["labels"] = {}
        state["selected_id"] = None

        # update dropdown choices for "select by id"
        choices = [f"{i:03d} | {d.det_id} | conf={d.conf:.2f}" for i, d in enumerate(dets)]
        return _render(), f"{state['img_name']} | dets={len(dets)}", gr.update(choices=choices, value=None), "", ""

    def select_by_dropdown(choice):
        if choice is None or state["img_bgr"] is None:
            state["selected_id"] = None
            return _render(), "", gr.update(choices=[])

        # parse det_id
        det_id = choice.split("|")[1].strip()
        state["selected_id"] = det_id

        det = next((d for d in state["dets"] if d.det_id == det_id), None)
        suggestions = []
        if det is not None and det.feat is not None:
            sug = suggest_labels(state["lib"], det.feat, topk=6)
            suggestions = [f"{name} (dist={dist:.3f})" for name, dist in sug]
        return _render(), det_id, gr.update(choices=suggestions, value=(suggestions[0] if suggestions else None))

    def on_click(evt: gr.SelectData):
        if state["img_bgr"] is None:
            return _render(), "", gr.update(choices=[])

        x, y = evt.index  # (x,y)
        det = pick_det_by_click(state["dets"], float(x), float(y))
        if det is None:
            state["selected_id"] = None
            return _render(), "", gr.update(choices=[])

        state["selected_id"] = det.det_id
        suggestions = []
        if det.feat is not None:
            sug = suggest_labels(state["lib"], det.feat, topk=6)
            suggestions = [f"{name} (dist={dist:.3f})" for name, dist in sug]
        return _render(), det.det_id, gr.update(choices=suggestions, value=(suggestions[0] if suggestions else None))

    def apply_label(label_text, suggestion_choice):
        if state["img_bgr"] is None or state["selected_id"] is None:
            return _render(), "선택된 det 없음"

        label = (label_text or "").strip()

        # if user didn't type, allow picking suggestion like "NAME (dist=...)"
        if not label and suggestion_choice:
            label = suggestion_choice.split(" (dist=")[0].strip()

        if not label:
            return _render(), "라벨이 비어있음"

        state["labels"][state["selected_id"]] = label
        return _render(), f"라벨 적용: {state['selected_id']} -> {label}"

    def clear_labels():
        state["labels"] = {}
        return _render(), "라벨 초기화"

    def save_current_session():
        if state["img_bgr"] is None:
            return "이미지 없음"
        sess = Session(
            image_name=state["img_name"],
            image_size=(int(state["img_bgr"].shape[1]), int(state["img_bgr"].shape[0])),
            created_ms=now_ms(),
            dets=state["dets"],
            labels=state["labels"]
        )
        out = sessions_dir / f"session_{Path(state['img_name']).stem}_{sess.created_ms}.json"
        save_session(out, sess)
        state["last_session_path"] = str(out)

        # update label library
        update_library_from_session(sessions_dir, sess)
        state["lib"] = load_library(sessions_dir)

        return f"저장됨: {out.name} (labels={len(sess.labels)})"

    def load_and_remap(prev_json_path, ransac_iters, nn_thresh, max_match_dist, merge_mode):
        """
        merge_mode:
          - "overwrite": current labels overwritten by remapped
          - "keep": only add labels that are currently unlabeled
        """
        if state["img_bgr"] is None:
            return _render(), "(현재 이미지 없음)"
        if not prev_json_path:
            return _render(), "(이전 세션 json 선택 필요)"

        prev = load_session(Path(prev_json_path))

        remapped, T, score, missing, total = remap_labels(
            prev, state["dets"],
            ransac_iters=int(ransac_iters),
            nn_thresh=float(nn_thresh),
            max_match_dist=float(max_match_dist)
        )

        if merge_mode == "keep":
            for k, v in remapped.items():
                if k not in state["labels"]:
                    state["labels"][k] = v
        else:
            state["labels"].update(remapped)

        msg = f"재매핑: {len(remapped)}/{total} 성공, missing={missing}, ransac_res={score:.2f}"
        return _render(), msg

    def auto_recall_best(ransac_iters, nn_thresh, max_match_dist, merge_mode):
        if state["img_bgr"] is None:
            return _render(), "(현재 이미지 없음)"
        jsons = sorted(sessions_dir.glob("session_*.json"))
        if not jsons:
            return _render(), "(sessions 폴더에 session_*.json 없음)"

        best = None
        best_path = None
        for p in jsons:
            prev = load_session(p)
            s = score_session_for_current(prev, state["dets"],
                                          ransac_iters=int(ransac_iters),
                                          nn_thresh=float(nn_thresh),
                                          max_match_dist=float(max_match_dist))
            if s is None:
                continue
            # primary: found count, secondary: ratio, tertiary: residual score (lower better)
            key = (s["found"], s["ratio"], -s["score"])
            if best is None or key > best:
                best = key
                best_path = p

        if best_path is None:
            return _render(), "(적절한 이전 세션을 못 찾음)"

        prev = load_session(best_path)
        remapped, T, score, missing, total = remap_labels(
            prev, state["dets"],
            ransac_iters=int(ransac_iters),
            nn_thresh=float(nn_thresh),
            max_match_dist=float(max_match_dist)
        )

        if merge_mode == "keep":
            for k, v in remapped.items():
                if k not in state["labels"]:
                    state["labels"][k] = v
        else:
            state["labels"].update(remapped)

        msg = f"[AUTO] best={best_path.name} | {len(remapped)}/{total} 성공, missing={missing}, ransac_res={score:.2f}"
        return _render(), msg

    def export_labels_csv():
        if state["img_bgr"] is None:
            return "(이미지 없음)"
        rows = ["det_id,label,cx,cy,x1,y1,x2,y2,conf,cls"]
        id2det = {d.det_id: d for d in state["dets"]}
        for det_id, label in state["labels"].items():
            d = id2det.get(det_id)
            if d is None:
                continue
            x1,y1,x2,y2 = d.bbox_xyxy
            cx,cy = d.centroid_xy
            rows.append(f"{det_id},{label},{cx:.2f},{cy:.2f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{d.conf:.3f},{d.cls}")
        return "\n".join(rows)

    # UI
    with gr.Blocks(title="Colony Tool (YOLOv11 Detect)") as demo:
        gr.Markdown("## Colony 작업 툴 (YOLOv11 Detect 기반)\n"
                    "- 검출 → 클릭 라벨링 → 세션 저장\n"
                    "- 다음 사진에서 (회전/확대/이동) 라벨 재불러오기\n"
                    "- 라벨 라이브러리로 자동 추천/불러오기")

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.File(label="이미지 파일 선택 (png/jpg)")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="conf")
                iou = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="iou")
                imgsz = gr.Slider(320, 2048, value=1024, step=32, label="imgsz")
                min_area = gr.Slider(1, 10000, value=30, step=1, label="min bbox area (px^2)")
                max_area = gr.Slider(1000, 2_000_000, value=500_000, step=1000, label="max bbox area (px^2)")
                use_roi = gr.Checkbox(value=True, label="원형 ROI 마스킹(권장)")
                drop_ring = gr.Checkbox(value=True, label="가장자리 반사띠 주변 drop")
                btn_load = gr.Button("이미지 로드 + 검출")

                status = gr.Textbox(label="상태", value="")

                det_dropdown = gr.Dropdown(label="det 선택(드롭다운)", choices=[], value=None)
                selected_id = gr.Textbox(label="선택된 det_id", value="", interactive=False)

                suggestion = gr.Dropdown(label="라벨 추천(라이브러리)", choices=[], value=None)
                label_text = gr.Textbox(label="라벨 입력(빈칸이면 추천 사용)", value="")
                btn_apply = gr.Button("라벨 적용")
                btn_clear = gr.Button("라벨 초기화")

                btn_save = gr.Button("현재 세션 저장(session_*.json)")
                save_msg = gr.Textbox(label="저장 로그", value="", interactive=False)

                gr.Markdown("---")
                prev_json = gr.File(label="이전 세션 JSON 선택 (session_*.json)")
                ransac_iters = gr.Slider(50, 3000, value=400, step=50, label="RANSAC iters")
                nn_thresh = gr.Slider(3, 80, value=18, step=1, label="RANSAC NN thresh(px)")
                max_match_dist = gr.Slider(3, 120, value=25, step=1, label="매칭 허용 거리(px)")
                merge_mode = gr.Radio(["overwrite", "keep"], value="overwrite", label="병합 방식")

                btn_remap = gr.Button("선택한 이전 세션으로 재매핑")
                btn_auto = gr.Button("sessions 폴더에서 자동으로 best 세션 찾아 재매핑")
                remap_msg = gr.Textbox(label="재매핑 로그", value="", interactive=False)

                gr.Markdown("---")
                btn_export = gr.Button("라벨 CSV 내보내기(텍스트)")
                csv_out = gr.Textbox(label="CSV", value="", lines=10)

            with gr.Column(scale=2):
                img_out = gr.Image(label="결과(클릭해서 det 선택)", type="numpy")

        # wire
        btn_load.click(
            fn=load_image,
            inputs=[img_in, conf, iou, imgsz, min_area, max_area, use_roi, drop_ring],
            outputs=[img_out, status, det_dropdown, selected_id, csv_out]
        )

        det_dropdown.change(fn=select_by_dropdown, inputs=[det_dropdown], outputs=[img_out, selected_id, suggestion])
        img_out.select(fn=on_click, inputs=None, outputs=[img_out, selected_id, suggestion])

        btn_apply.click(fn=apply_label, inputs=[label_text, suggestion], outputs=[img_out, status])
        btn_clear.click(fn=clear_labels, inputs=None, outputs=[img_out, status])

        btn_save.click(fn=save_current_session, inputs=None, outputs=[save_msg])

        btn_remap.click(
            fn=lambda f, a, b, c, m: load_and_remap(
                prev_json_path=f.name if f else "",
                ransac_iters=a, nn_thresh=b, max_match_dist=c, merge_mode=m
            ),
            inputs=[prev_json, ransac_iters, nn_thresh, max_match_dist, merge_mode],
            outputs=[img_out, remap_msg]
        )

        btn_auto.click(
            fn=lambda a, b, c, m: auto_recall_best(a, b, c, m),
            inputs=[ransac_iters, nn_thresh, max_match_dist, merge_mode],
            outputs=[img_out, remap_msg]
        )

        btn_export.click(fn=export_labels_csv, inputs=None, outputs=[csv_out])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="YOLOv11 detect weights (best.pt)")
    ap.add_argument("--sessions", type=str, default="./sessions", help="folder to store sessions + label_library.json")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = build_app(args.weights, args.sessions)
    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
