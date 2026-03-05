import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import streamlit as st
from scipy.optimize import linear_sum_assignment

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Colony:
    colony_id: str
    centroid: Tuple[float, float]  # (x, y)
    area: float
    circularity: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

@dataclass
class LabeledColony:
    colony: Colony
    label: str

# -----------------------------
# Utility
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def make_id(prefix="c") -> str:
    return f"{prefix}_{now_ms()}_{np.random.randint(0, 1_000_000):06d}"

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def draw_overlay(img_bgr: np.ndarray,
                 colonies: List[Colony],
                 labels: Dict[str, str],
                 color=(0, 255, 255)) -> np.ndarray:
    out = img_bgr.copy()
    for c in colonies:
        x, y, w, h = c.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cx, cy = map(int, c.centroid)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        if c.colony_id in labels:
            cv2.putText(out, labels[c.colony_id], (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# -----------------------------
# Colony detection (no training baseline)
# -----------------------------
def detect_plate_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Rough circular ROI using HoughCircles. This is NOT 'plate recognition' in a strict sense,
    just masking outside background to reduce noise.
    Returns (cx, cy, r) or None.
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] // 4,
        param1=100, param2=30,
        minRadius=int(min(gray.shape[:2]) * 0.3),
        maxRadius=int(min(gray.shape[:2]) * 0.55)
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    # choose largest radius
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    cx, cy, r = circles[0]
    return (cx, cy, r)

def colony_mask(gray: np.ndarray, roi_circle: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Create a binary mask for bright colonies:
    - optional circular ROI mask
    - adaptive threshold + morphology
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    # normalize contrast a bit
    g = cv2.equalizeHist(g)

    # adaptive threshold works well for uneven illumination
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=41, C=-5
    )

    # remove small noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    if roi_circle is not None:
        cx, cy, r = roi_circle
        rr = int(r * 0.98)  # slightly smaller to avoid edge reflection ring
        roi = np.zeros_like(th)
        cv2.circle(roi, (cx, cy), rr, 255, -1)
        th = cv2.bitwise_and(th, roi)

    return th

def extract_colonies(bin_mask: np.ndarray,
                     min_area: int = 80,
                     max_area: int = 200000) -> List[Colony]:
    """
    Connected components -> colonies with centroid/area/circularity/bbox
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    colonies: List[Colony] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]

        # contour-based perimeter for circularity
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        circ = 0.0 if perim <= 1e-6 else float(4.0 * math.pi * area / (perim * perim))

        colony = Colony(
            colony_id=make_id("col"),
            centroid=(float(cx), float(cy)),
            area=float(area),
            circularity=float(circ),
            bbox=(int(x), int(y), int(w), int(h))
        )
        colonies.append(colony)

    # sort stable (left->right, top->bottom)
    colonies.sort(key=lambda c: (c.centroid[1], c.centroid[0]))
    return colonies

def find_clicked_colony(colonies: List[Colony], x: float, y: float) -> Optional[Colony]:
    """
    Choose colony by nearest centroid among those whose bbox contains point.
    Fallback: nearest centroid overall.
    """
    inside = []
    for c in colonies:
        bx, by, bw, bh = c.bbox
        if bx <= x <= bx + bw and by <= y <= by + bh:
            inside.append(c)
    if inside:
        return min(inside, key=lambda c: (c.centroid[0]-x)**2 + (c.centroid[1]-y)**2)
    if not colonies:
        return None
    return min(colonies, key=lambda c: (c.centroid[0]-x)**2 + (c.centroid[1]-y)**2)

# -----------------------------
# Similarity transform (rotate+scale+translate)
# -----------------------------
def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Estimate similarity transform T such that:
      dst ~ s * R * src + t
    Returns (3x3 homography-like matrix), inlier RMSE (on given correspondences).
    src,dst: (N,2)
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    if n < 2:
        return np.eye(3, dtype=np.float64), float("inf")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # covariance
    cov = (dst_demean.T @ src_demean) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_src = (src_demean**2).sum() / n
    if var_src < 1e-12:
        s = 1.0
    else:
        s = S.sum() / var_src

    t = dst_mean - s * (R @ src_mean)

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = s * R
    T[:2, 2] = t

    pred = apply_T(src, T)
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return T, rmse

def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    pts: (N,2)
    T: (3,3)
    """
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    h = np.hstack([pts.astype(np.float64), ones])
    out = (T @ h.T).T
    return out[:, :2]

def ransac_similarity(old_pts: np.ndarray,
                      new_pts: np.ndarray,
                      iters: int = 300,
                      thresh: float = 18.0,
                      min_inliers: int = 6,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust similarity transform estimation without known correspondences.
    Strategy:
      - sample 2 points from old, 2 from new to propose T using provisional pairing
      - score by nearest-neighbor residuals under T
    Returns: (best_T, inlier_mask over old_pts by NN residual)
    """
    rng = np.random.default_rng(seed)
    if old_pts.shape[0] < 2 or new_pts.shape[0] < 2:
        return np.eye(3, dtype=np.float64), np.zeros((old_pts.shape[0],), dtype=bool)

    best_T = np.eye(3, dtype=np.float64)
    best_inliers = np.zeros((old_pts.shape[0],), dtype=bool)
    best_count = 0
    best_score = float("inf")

    # prebuild for NN (brute-force is ok for tens~few hundreds)
    for _ in range(iters):
        i1, i2 = rng.choice(old_pts.shape[0], size=2, replace=False)
        j1, j2 = rng.choice(new_pts.shape[0], size=2, replace=False)

        src = old_pts[[i1, i2]]
        dst = new_pts[[j1, j2]]
        T, _ = umeyama_similarity(src, dst)

        warped = apply_T(old_pts, T)  # (No,2)

        # NN residuals to new_pts
        d2 = ((warped[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2)
        nn = np.sqrt(d2.min(axis=1))
        inliers = nn < thresh
        cnt = int(inliers.sum())
        score = float(nn[inliers].mean()) if cnt > 0 else float("inf")

        if cnt > best_count or (cnt == best_count and score < best_score):
            best_count = cnt
            best_score = score
            best_T = T
            best_inliers = inliers

    # refine using correspondences induced by NN on inliers
    if best_count >= min_inliers:
        warped = apply_T(old_pts, best_T)
        d2 = ((warped[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2)
        nn_idx = d2.argmin(axis=1)

        src = old_pts[best_inliers]
        dst = new_pts[nn_idx[best_inliers]]
        best_T, _ = umeyama_similarity(src, dst)

    return best_T, best_inliers

def match_labels_by_hungarian(pred_pts: np.ndarray,
                              new_pts: np.ndarray,
                              max_dist: float = 25.0) -> List[int]:
    """
    pred_pts: (K,2) predicted positions of labeled colonies
    new_pts: (M,2) current colonies centroids
    Return list of length K: matched index in [0..M-1] or -1 if no good match
    """
    if pred_pts.shape[0] == 0 or new_pts.shape[0] == 0:
        return [-1] * pred_pts.shape[0]

    d = np.sqrt(((pred_pts[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2))
    cost = d.copy()
    # Large penalty for too-far matches
    cost[d > max_dist] = 1e6

    r, c = linear_sum_assignment(cost)
    out = [-1] * pred_pts.shape[0]
    for ri, ci in zip(r, c):
        if cost[ri, ci] < 1e6:
            out[ri] = int(ci)
    return out

# -----------------------------
# Label database I/O
# -----------------------------
def save_session(path: str,
                 image_name: str,
                 colonies: List[Colony],
                 labels: Dict[str, str]) -> None:
    data = {
        "image_name": image_name,
        "colonies": [asdict(c) for c in colonies],
        "labels": labels
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(path: str) -> Tuple[str, List[Colony], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    colonies = []
    for cd in data["colonies"]:
        colonies.append(Colony(
            colony_id=cd["colony_id"],
            centroid=tuple(cd["centroid"]),
            area=float(cd["area"]),
            circularity=float(cd["circularity"]),
            bbox=tuple(cd["bbox"])
        ))
    return data.get("image_name", ""), colonies, dict(data.get("labels", {}))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Colony Label Tool", layout="wide")
st.title("Colony 작업 툴 (프로토타입)")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) 현재 이미지")
    up = st.file_uploader("이미지 업로드 (png/jpg)", type=["png", "jpg", "jpeg"], key="img_cur")
    min_area = st.slider("min_area", 10, 5000, 80, 10)
    max_area = st.slider("max_area", 5000, 500000, 200000, 5000)
    use_roi = st.checkbox("원형 ROI 마스킹(권장)", value=True)
    click_x = st.number_input("클릭 x (수동 입력) ", value=0.0, step=1.0)
    click_y = st.number_input("클릭 y (수동 입력) ", value=0.0, step=1.0)
    label_text = st.text_input("라벨 이름(입력 후 '라벨 적용')", value="")

with col2:
    st.subheader("2) 과거 라벨 세션(JSON) 불러오기 + 재매핑")
    up_json = st.file_uploader("세션 JSON 업로드", type=["json"], key="json_prev")
    max_dist = st.slider("재매핑 매칭 허용 거리(px)", 5, 80, 25, 1)
    ransac_thresh = st.slider("RANSAC inlier thresh(px)", 5, 60, 18, 1)
    ransac_iters = st.slider("RANSAC iters", 50, 2000, 300, 50)

# state
if "cur_colonies" not in st.session_state:
    st.session_state.cur_colonies = []
if "cur_labels" not in st.session_state:
    st.session_state.cur_labels = {}  # colony_id -> label
if "cur_img_name" not in st.session_state:
    st.session_state.cur_img_name = "current"

# Load current image and detect colonies
cur_img = None
cur_gray = None
roi_circle = None

if up is not None:
    file_bytes = np.frombuffer(up.read(), np.uint8)
    cur_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if cur_img is None:
        st.error("이미지 로드 실패")
    else:
        cur_gray = to_gray(cur_img)
        if use_roi:
            roi_circle = detect_plate_roi(cur_gray)
        mask = colony_mask(cur_gray, roi_circle)
        colonies = extract_colonies(mask, min_area=min_area, max_area=max_area)

        # keep stable ids per run? (prototype: new ids each run)
        st.session_state.cur_colonies = colonies
        st.session_state.cur_img_name = up.name

        # show mask preview
        with col1:
            st.caption("이진 마스크(디버그)")
            st.image(mask, clamp=True, channels="GRAY")

        overlay = draw_overlay(cur_img, colonies, st.session_state.cur_labels)
        with col1:
            st.caption("검출 결과(박스/중심)")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), clamp=True)

        # Click labeling (manual input version)
        with col1:
            if st.button("라벨 적용(현재 클릭좌표에 가장 가까운 콜로니)"):
                if label_text.strip() == "":
                    st.warning("라벨 이름이 비어있음")
                else:
                    c = find_clicked_colony(colonies, click_x, click_y)
                    if c is None:
                        st.warning("콜로니를 찾지 못함")
                    else:
                        st.session_state.cur_labels[c.colony_id] = label_text.strip()
                        st.success(f"라벨 적용: {c.colony_id} -> {label_text.strip()}")

            if st.button("라벨 전체 초기화"):
                st.session_state.cur_labels = {}

        # Save current session
        with col1:
            save_path = st.text_input("저장 파일명", value="session.json")
            if st.button("현재 세션 저장(JSON)"):
                save_session(save_path, st.session_state.cur_img_name,
                             st.session_state.cur_colonies, st.session_state.cur_labels)
                st.success(f"저장 완료: {save_path}")

# Load previous session and remap
if up_json is not None and cur_img is not None:
    prev_tmp = "prev_session_tmp.json"
    with open(prev_tmp, "wb") as f:
        f.write(up_json.read())

    prev_img_name, prev_colonies, prev_labels = load_session(prev_tmp)

    # Extract point sets
    old_pts = np.array([c.centroid for c in prev_colonies], dtype=np.float64)
    new_pts = np.array([c.centroid for c in st.session_state.cur_colonies], dtype=np.float64)

    with col2:
        st.write(f"이전 세션 이미지: `{prev_img_name}`")
        st.write(f"이전 콜로니 수: {len(prev_colonies)}, 라벨 수: {len(prev_labels)}")
        st.write(f"현재 콜로니 수: {len(st.session_state.cur_colonies)}")

    # estimate transform using centroid sets
    T, inliers = ransac_similarity(
        old_pts, new_pts,
        iters=ransac_iters,
        thresh=float(ransac_thresh),
        min_inliers=6,
        seed=0
    )

    # Predict positions for labeled colonies only
    labeled_old = [c for c in prev_colonies if c.colony_id in prev_labels]
    labeled_old_pts = np.array([c.centroid for c in labeled_old], dtype=np.float64)
    pred_pts = apply_T(labeled_old_pts, T)

    # match to current colonies
    matches = match_labels_by_hungarian(pred_pts, new_pts, max_dist=float(max_dist))

    # prepare overlay labels
    remapped_labels: Dict[str, str] = {}
    missing = 0
    for i, mi in enumerate(matches):
        label = prev_labels[labeled_old[i].colony_id]
        if mi == -1:
            missing += 1
            continue
        cur_id = st.session_state.cur_colonies[mi].colony_id
        remapped_labels[cur_id] = label

    # show results
    with col2:
        st.subheader("재매핑 결과")
        st.write("추정 변환 행렬 T (similarity):")
        st.code(np.array2string(T, precision=4, suppress_small=True))
        st.write(f"라벨 재매핑 성공: {len(remapped_labels)} / {len(labeled_old)} (missing {missing})")

        overlay2 = draw_overlay(cur_img, st.session_state.cur_colonies, remapped_labels, color=(0, 255, 0))
        st.image(cv2.cvtColor(overlay2, cv2.COLOR_BGR2RGB), clamp=True)

        if st.button("재매핑 라벨을 현재 세션에 병합(덮어쓰기)"):
            st.session_state.cur_labels.update(remapped_labels)
            st.success("병합 완료")

# Notes for user
st.markdown("""
### 사용 팁
- 이 프로토타입은 **학습 모델 없이** 밝은 콜로니를 threshold로 잡습니다. (데이터가 일정하면 의외로 잘 먹습니다)
- 클릭 UI는 지금은 좌표 수동 입력 버전입니다.  
  실제 클릭 이벤트까지 하려면 `streamlit-drawable-canvas`(외부 컴포넌트) 또는 Gradio로 바꾸면 됩니다.
- 재매핑은 **centroid 점집합 기반** similarity transform(RANSAC+Umeyama)로 동작합니다.
""")
