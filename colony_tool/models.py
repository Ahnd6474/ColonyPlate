from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Det:
    det_id: str
    cls: int
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]
    centroid_xy: Tuple[float, float]
    feat: Optional[List[float]] = None


@dataclass
class Session:
    image_name: str
    image_size: Tuple[int, int]
    created_ms: int
    dets: List[Det]
    labels: Dict[str, str]
