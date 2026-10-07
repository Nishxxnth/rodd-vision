"""
tests/test_spatial.py
Tests for direction, distance, and priority logic in rodd/spatial.py
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rodd.config import load_config, SpatialConfig
from rodd.detectors import Detection
from rodd.spatial import (
    get_direction,
    get_distance,
    get_priority,
    annotate,
    top_priority,
)

# Default spatial config (matches config.yaml defaults)
SCFG = SpatialConfig(
    left_threshold=0.33,
    right_threshold=0.66,
    near_area_ratio=0.15,
    far_area_ratio=0.03,
    near_bottom_y=0.75,
)

FRAME_W, FRAME_H = 640, 480


# ---------------------------------------------------------------------------
# Direction tests
# ---------------------------------------------------------------------------

class TestGetDirection:
    def test_far_left(self):
        # cx = 10% of 640 = 64px  → left
        assert get_direction(64, FRAME_W, SCFG) == "left"

    def test_boundary_left(self):
        # rel exactly at left_threshold → ahead (not left)
        # Use ceil so cx/frame_w >= threshold (int truncation can overshoot)
        import math
        cx = math.ceil(FRAME_W * SCFG.left_threshold)  # 212 → rel=0.3312 > 0.33 → ahead
        assert get_direction(cx, FRAME_W, SCFG) == "ahead"

    def test_just_inside_left(self):
        cx = int(FRAME_W * SCFG.left_threshold) - 1  # 210
        assert get_direction(cx, FRAME_W, SCFG) == "left"

    def test_centre(self):
        assert get_direction(FRAME_W // 2, FRAME_W, SCFG) == "ahead"

    def test_boundary_right(self):
        # cx just above right_threshold → right
        cx = int(FRAME_W * SCFG.right_threshold) + 1  # 423
        assert get_direction(cx, FRAME_W, SCFG) == "right"

    def test_far_right(self):
        assert get_direction(FRAME_W - 10, FRAME_W, SCFG) == "right"


# ---------------------------------------------------------------------------
# Distance tests
# ---------------------------------------------------------------------------

class TestGetDistance:
    def _bbox_from_area_ratio(self, ratio: float):
        """Produce a centred bbox with the given area ratio."""
        side = (ratio * FRAME_W * FRAME_H) ** 0.5
        cx, cy = FRAME_W / 2, FRAME_H / 2
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    def test_near_by_area(self):
        # area ratio = 0.20 >= near_area_ratio 0.15
        bbox = self._bbox_from_area_ratio(0.20)
        assert get_distance(bbox, FRAME_W, FRAME_H, SCFG) == "near"

    def test_far_by_area(self):
        # area ratio = 0.01 <= far_area_ratio 0.03
        bbox = self._bbox_from_area_ratio(0.01)
        assert get_distance(bbox, FRAME_W, FRAME_H, SCFG) == "far"

    def test_medium(self):
        # area ratio = 0.07 → medium
        bbox = self._bbox_from_area_ratio(0.07)
        assert get_distance(bbox, FRAME_W, FRAME_H, SCFG) == "medium"

    def test_near_by_bottom_y(self):
        # box bottom at 90% of frame height → near override
        y2 = FRAME_H * 0.90
        bbox = (100, FRAME_H * 0.50, 200, y2)
        assert get_distance(bbox, FRAME_W, FRAME_H, SCFG) == "near"


# ---------------------------------------------------------------------------
# Priority tests
# ---------------------------------------------------------------------------

DANGER_ORDER = [
    "pit", "stair", "car", "bus", "tram",
    "person", "electric pole", "road sign",
    "dustbin", "door", "building", "pillar",
    "street chair", "tree",
]


class TestGetPriority:
    def test_highest_priority(self):
        assert get_priority("pit", DANGER_ORDER) == 0

    def test_second_priority(self):
        assert get_priority("stair", DANGER_ORDER) == 1

    def test_person_priority(self):
        p_person = get_priority("person", DANGER_ORDER)
        p_car    = get_priority("car", DANGER_ORDER)
        assert p_person > p_car  # car is more dangerous

    def test_unknown_class(self):
        p = get_priority("unknown_class", DANGER_ORDER)
        assert p == len(DANGER_ORDER)


# ---------------------------------------------------------------------------
# End-to-end annotate + top_priority
# ---------------------------------------------------------------------------

class TestAnnotatePipeline:
    def _det(self, label, bbox):
        return Detection(label=label, confidence=0.9, bbox=bbox)

    def test_top_priority_is_most_dangerous(self):
        cfg = load_config()
        detections = [
            self._det("tree",  (400, 100, 600, 300)),   # low priority
            self._det("stair", (100, 200, 300, 450)),   # high priority, near (bottom y > 75%)
            self._det("car",   (200, 100, 350, 250)),   # medium priority
        ]
        annotated = annotate(detections, FRAME_W, FRAME_H, cfg)
        top = top_priority(annotated)
        assert top is not None
        assert top.label == "stair"

    def test_empty_detections(self):
        cfg = load_config()
        annotated = annotate([], FRAME_W, FRAME_H, cfg)
        assert annotated == []
        assert top_priority(annotated) is None

    def test_direction_assignment(self):
        cfg = load_config()
        det_left  = self._det("person", (0, 100, 60, 300))    # cx=30 → left
        det_right = self._det("car",    (580, 100, 640, 300)) # cx=610 → right
        det_ahead = self._det("bus",    (250, 100, 400, 300)) # cx=325 → ahead
        annotated = annotate([det_left, det_right, det_ahead], FRAME_W, FRAME_H, cfg)
        dirs = {a.label: a.direction for a in annotated}
        assert dirs["person"] == "left"
        assert dirs["car"]    == "right"
        assert dirs["bus"]    == "ahead"
