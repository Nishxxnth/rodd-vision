"""
rodd/spatial.py
Direction (left/ahead/right), distance (near/medium/far), and priority logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import RODDConfig, SpatialConfig
from .detectors import Detection


# ---------------------------------------------------------------------------
# Spatial annotation
# ---------------------------------------------------------------------------

@dataclass
class SpatialAnnotation:
    detection: Detection
    direction: str          # "left" | "ahead" | "right"
    distance: str           # "near" | "medium" | "far"
    priority: int           # lower number = higher priority

    @property
    def label(self) -> str:
        return self.detection.label

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def bbox(self):
        return self.detection.bbox

    def announce_text(self) -> str:
        """Return the text spoken / displayed for this obstacle."""
        if self.distance == "medium":
            return f"{self.label} {self.direction}"
        return f"{self.label} {self.direction}, {self.distance}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_direction(cx: float, frame_w: int, cfg: SpatialConfig) -> str:
    """
    Determine direction from the box centre-x relative to frame width.

    Parameters
    ----------
    cx : box centre x in pixels
    frame_w : frame width in pixels
    cfg : SpatialConfig with left/right thresholds (0-1 fractions)
    """
    rel = cx / frame_w
    if rel < cfg.left_threshold:
        return "left"
    elif rel > cfg.right_threshold:
        return "right"
    else:
        return "ahead"
    # Note: rel == left_threshold → falls into else ("ahead"); rel == right_threshold → "ahead"


def get_distance(
    bbox: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    cfg: SpatialConfig,
) -> str:
    """
    Estimate distance from box area ratio and bottom-edge position.

    Near heuristics:
      - box area / frame area >= near_area_ratio, OR
      - box bottom-edge y / frame_h  >= near_bottom_y

    Far heuristic:
      - box area / frame area <= far_area_ratio

    Otherwise: medium.
    """
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    area_ratio = box_area / max(frame_area, 1)
    bottom_y_ratio = y2 / max(frame_h, 1)

    if area_ratio >= cfg.near_area_ratio or bottom_y_ratio >= cfg.near_bottom_y:
        return "near"
    elif area_ratio <= cfg.far_area_ratio:
        return "far"
    else:
        return "medium"


def get_priority(label: str, danger_order: List[str]) -> int:
    """Lower index in danger_order → higher priority (lower number)."""
    try:
        return danger_order.index(label)
    except ValueError:
        return len(danger_order)  # unknown classes get lowest priority


# ---------------------------------------------------------------------------
# Annotate a list of detections
# ---------------------------------------------------------------------------

def annotate(
    detections: List[Detection],
    frame_w: int,
    frame_h: int,
    cfg: RODDConfig,
) -> List[SpatialAnnotation]:
    """
    Attach direction, distance, and priority to every detection.
    Returns the list sorted by priority (highest first).
    """
    annotated: List[SpatialAnnotation] = []
    for det in detections:
        cx = (det.x1 + det.x2) / 2.0
        direction = get_direction(cx, frame_w, cfg.spatial)
        distance = get_distance(det.bbox, frame_w, frame_h, cfg.spatial)
        priority = get_priority(det.label, cfg.danger_order)
        annotated.append(
            SpatialAnnotation(
                detection=det,
                direction=direction,
                distance=distance,
                priority=priority,
            )
        )
    # sort: priority ASC (lower = more dangerous), then distance (near < medium < far)
    dist_rank = {"near": 0, "medium": 1, "far": 2}
    annotated.sort(key=lambda a: (a.priority, dist_rank.get(a.distance, 1)))
    return annotated


def top_priority(annotated: List[SpatialAnnotation]) -> Optional[SpatialAnnotation]:
    """Return the highest-priority annotated obstacle, or None."""
    return annotated[0] if annotated else None
