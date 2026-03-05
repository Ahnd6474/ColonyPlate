from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from colony_tool.models import Det, Session


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
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    var_src = (src_d**2).sum() / n
    scale = 1.0 if var_src < 1e-12 else float(s.sum() / var_src)
    t = dst_mean - scale * (r @ src_mean)

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = scale * r
    T[:2, 2] = t

    pred = apply_T(src, T)
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return T, rmse


def ransac_similarity(
    old_pts: np.ndarray,
    new_pts: np.ndarray,
    iters: int = 400,
    nn_thresh: float = 18.0,
    min_inliers: int = 8,
    seed: int = 0,
):
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

    if best_cnt >= min_inliers:
        best_T = refine_similarity_iterative(
            old_pts,
            new_pts,
            init_T=best_T,
            max_nn_thresh=nn_thresh,
            min_inliers=min_inliers,
            steps=3,
        )

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


def refine_similarity_iterative(
    old_pts: np.ndarray,
    new_pts: np.ndarray,
    init_T: np.ndarray,
    max_nn_thresh: float,
    min_inliers: int = 8,
    steps: int = 3,
) -> np.ndarray:
    """Refine similarity transform with progressively tighter inlier threshold."""
    T = init_T.copy()
    if old_pts.shape[0] < 2 or new_pts.shape[0] < 2:
        return T

    for step in range(max(1, steps)):
        thr = max_nn_thresh * (0.7 ** step)
        warped = apply_T(old_pts, T)
        d2 = ((warped[:, None, :] - new_pts[None, :, :]) ** 2).sum(axis=2)
        nn_idx = d2.argmin(axis=1)
        nn_dist = np.sqrt(d2[np.arange(d2.shape[0]), nn_idx])

        # Mutual NN constraint reduces accidental pairings in dense colonies.
        rev_idx = d2.argmin(axis=0)
        mutual = np.array([rev_idx[j] == i for i, j in enumerate(nn_idx)], dtype=bool)
        inliers = (nn_dist < thr) & mutual
        if int(inliers.sum()) < min_inliers:
            continue

        src = old_pts[inliers]
        dst = new_pts[nn_idx[inliers]]
        T, _ = umeyama_similarity(src, dst)
    return T


def score_session_for_current(prev: Session, cur_dets: List[Det], ransac_iters=400, nn_thresh=18.0, max_match_dist=25.0):
    old_pts = np.array([d.centroid_xy for d in prev.dets], dtype=np.float64)
    new_pts = np.array([d.centroid_xy for d in cur_dets], dtype=np.float64)
    T, _, score = ransac_similarity(old_pts, new_pts, iters=ransac_iters, nn_thresh=nn_thresh, min_inliers=8)

    labeled_old = [d for d in prev.dets if d.det_id in prev.labels]
    if not labeled_old:
        return None

    labeled_old_pts = np.array([d.centroid_xy for d in labeled_old], dtype=np.float64)
    pred_pts = apply_T(labeled_old_pts, T)
    matches = match_labels_hungarian(pred_pts, new_pts, max_dist=max_match_dist)

    found = sum(1 for m in matches if m != -1)
    return {
        "T": T,
        "found": found,
        "total": len(labeled_old),
        "ratio": (found / max(1, len(labeled_old))),
        "score": score,
    }


def remap_labels(prev: Session, cur_dets: List[Det], ransac_iters=400, nn_thresh=18.0, max_match_dist=25.0):
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
