import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from colony_tool.models import Det, Session


def save_session(path: Path, sess: Session):
    data = {
        "image_name": sess.image_name,
        "image_size": list(sess.image_size),
        "created_ms": sess.created_ms,
        "dets": [asdict(d) for d in sess.dets],
        "labels": sess.labels,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(path: Path) -> Session:
    data = json.loads(path.read_text(encoding="utf-8"))
    dets = []
    for dd in data["dets"]:
        dets.append(
            Det(
                det_id=dd["det_id"],
                cls=int(dd["cls"]),
                conf=float(dd["conf"]),
                bbox_xyxy=tuple(dd["bbox_xyxy"]),
                centroid_xy=tuple(dd["centroid_xy"]),
                feat=dd.get("feat", None),
            )
        )

    return Session(
        image_name=data["image_name"],
        image_size=tuple(data["image_size"]),
        created_ms=int(data["created_ms"]),
        dets=dets,
        labels=dict(data.get("labels", {})),
    )


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
        if len(lib[label]) > 200:
            lib[label] = lib[label][-200:]
    save_library(sessions_dir, lib)


def suggest_labels(lib: Dict[str, List[List[float]]], feat: List[float], topk: int = 5) -> List[Tuple[str, float]]:
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
