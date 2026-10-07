"""
rodd/utils.py
Drawing utilities, FPS counter, logging setup.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np

from .config import RODDConfig
from .spatial import SpatialAnnotation


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Sliding-window FPS counter."""

    def __init__(self, window: int = 30):
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._ts) - 1) / elapsed


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE: dict[str, tuple[int, int, int]] = {
    "building":      (128, 128,   0),
    "bus":           (  0, 200, 255),
    "car":           (  0, 128, 255),
    "door":          (180,  90,   0),
    "dustbin":       ( 80, 200,  80),
    "electric pole": (200,   0, 200),
    "person":        (  0, 255, 128),
    "pillar":        (160,  80,  40),
    "pit":           (  0,   0, 255),
    "road sign":     (255, 165,   0),
    "stair":         (255,   0,   0),
    "street chair":  ( 64, 224, 208),
    "tram":          (255,  20, 147),
    "tree":          ( 34, 139,  34),
}
_DEFAULT_COLOR = (200, 200, 200)


def _color(label: str) -> tuple[int, int, int]:
    return _PALETTE.get(label, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    annotations: List[SpatialAnnotation],
    fps: float,
    inference_ms: float,
    cfg: RODDConfig,
) -> np.ndarray:
    """Draw bounding boxes, labels, and HUD overlay on *frame* (in-place)."""
    h, w = frame.shape[:2]

    for ann in annotations:
        x1, y1, x2, y2 = [int(v) for v in ann.bbox]
        color = _color(ann.label)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        label_str = ann.label
        if cfg.output.draw_conf:
            label_str += f" {ann.confidence:.0%}"
        label_str += f" | {ann.direction}"

        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label_str, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # Distance badge at bottom-left of box
        dist_color = {"near": (0, 0, 255), "medium": (0, 200, 255), "far": (0, 255, 0)}
        cv2.putText(
            frame, ann.distance, (x1 + 2, y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, dist_color.get(ann.distance, (255, 255, 255)), 1,
        )

    # HUD overlay (top-left)
    if cfg.output.draw_fps:
        hud = f"FPS: {fps:.1f}  Inf: {inference_ms:.0f}ms  Det: {len(annotations)}"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    return frame


def save_frame(frame: np.ndarray, out_path: str) -> None:
    """Write *frame* to *out_path*."""
    cv2.imwrite(out_path, frame)
