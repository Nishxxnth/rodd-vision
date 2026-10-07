"""
tests/test_detectors_mapping.py
Tests for COCO→RODD class mapping and filtering of unmapped classes.
No actual model is loaded – we mock the YOLO predictions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import types
import pytest
import numpy as np

from rodd.detectors import (
    COCO_TO_RODD,
    COCO_UNMAPPED,
    CocoFallbackDetector,
    YoloWorldDetector,
    Detection,
)
from rodd.config import load_config


# ---------------------------------------------------------------------------
# Mapping contract tests (no model needed)
# ---------------------------------------------------------------------------

class TestCocoRoddMapping:
    def test_person_maps_correctly(self):
        assert COCO_TO_RODD["person"] == "person"

    def test_train_maps_to_tram(self):
        assert COCO_TO_RODD["train"] == "tram"

    def test_bench_maps_to_street_chair(self):
        assert COCO_TO_RODD["bench"] == "street chair"

    def test_stop_sign_maps_to_road_sign(self):
        assert COCO_TO_RODD["stop sign"] == "road sign"

    def test_potted_plant_maps_to_tree(self):
        assert COCO_TO_RODD["potted plant"] == "tree"

    def test_car_maps_correctly(self):
        assert COCO_TO_RODD["car"] == "car"

    def test_bus_maps_correctly(self):
        assert COCO_TO_RODD["bus"] == "bus"

    def test_unmapped_coco_classes_not_in_mapping(self):
        # Common COCO classes that SHOULD NOT map
        for cls in ["cat", "dog", "pizza", "laptop", "bicycle"]:
            assert cls not in COCO_TO_RODD, f"'{cls}' should not be in COCO_TO_RODD"

    def test_all_mapped_rodd_classes_are_valid_rodd_targets(self):
        cfg = load_config()
        for coco_cls, rodd_cls in COCO_TO_RODD.items():
            assert rodd_cls in cfg.classes, (
                f"COCO class '{coco_cls}' maps to '{rodd_cls}' which is not in config classes"
            )

    def test_coco_unmapped_contains_expected_classes(self):
        # These classes MUST be in the unmapped list
        for cls in ["pit", "stair", "electric pole", "pillar"]:
            assert cls in COCO_UNMAPPED, f"'{cls}' should be in COCO_UNMAPPED"

    def test_coco_unmapped_not_in_mapping_values(self):
        mapped_values = set(COCO_TO_RODD.values())
        for cls in COCO_UNMAPPED:
            assert cls not in mapped_values, (
                f"'{cls}' is in COCO_UNMAPPED but also appears as a mapping target"
            )


# ---------------------------------------------------------------------------
# Mocked CocoFallbackDetector.detect()
# ---------------------------------------------------------------------------

class _FakeBox:
    """Minimal mock of a YOLO detection box."""
    def __init__(self, cls_id, conf, xyxy):
        import torch
        self.cls  = [torch.tensor(float(cls_id))]
        self.conf = [torch.tensor(float(conf))]
        self.xyxy = [torch.tensor(xyxy, dtype=torch.float32)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _patch_coco_model(detector: CocoFallbackDetector, fake_boxes: list):
    """Replace the detector's internal model.predict with a controlled stub."""
    fake_results = [_FakeResult(fake_boxes)]
    detector._model.predict = lambda *a, **kw: fake_results


class TestCocoFallbackDetect:
    def _make_detector(self):
        """Build a CocoFallbackDetector but with a mocked YOLO model."""
        # We'll create the object without calling __init__ to avoid downloading
        cfg = load_config()
        cfg.model.confidence = 0.3
        det = CocoFallbackDetector.__new__(CocoFallbackDetector)
        det.cfg = cfg
        det.conf_thresh = cfg.model.confidence
        det.imgsz = cfg.model.imgsz

        # Build a minimal fake model with .names
        fake_model = types.SimpleNamespace()
        fake_model.names = {
            0:  "person",
            2:  "car",
            5:  "bus",
            6:  "train",
            13: "bench",
            11: "stop sign",
            58: "potted plant",
            15: "cat",           # unmapped
            16: "dog",           # unmapped
        }
        fake_model.predict = lambda *a, **kw: []
        det._model = fake_model
        return det

    def test_mapped_class_included(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=0, conf=0.9, xyxy=[10, 20, 100, 200])]  # person
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "person"

    def test_unmapped_class_excluded(self):
        """cat (cls_id=15) is not in COCO_TO_RODD – must be dropped."""
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=15, conf=0.9, xyxy=[10, 20, 100, 200])]  # cat
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 0

    def test_train_maps_to_tram(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=6, conf=0.85, xyxy=[50, 50, 400, 400])]  # train
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "tram"

    def test_bench_maps_to_street_chair(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=13, conf=0.75, xyxy=[100, 100, 200, 200])]  # bench
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "street chair"

    def test_mixed_mapped_and_unmapped(self):
        """Only the mapped classes should appear in results."""
        import torch
        det = self._make_detector()
        boxes = [
            _FakeBox(cls_id=0,  conf=0.9, xyxy=[10, 20, 100, 200]),   # person → person
            _FakeBox(cls_id=16, conf=0.8, xyxy=[200, 50, 400, 300]),  # dog → dropped
            _FakeBox(cls_id=2,  conf=0.7, xyxy=[50, 60, 300, 400]),   # car → car
        ]
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 2
        labels = {r.label for r in results}
        assert labels == {"person", "car"}

    def test_confidence_and_bbox_preserved(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=5, conf=0.77, xyxy=[10.0, 20.0, 300.0, 400.0])]  # bus
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].confidence == pytest.approx(0.77, abs=0.01)
        assert results[0].bbox[0] == pytest.approx(10.0, abs=0.01)
        assert results[0].bbox[3] == pytest.approx(400.0, abs=0.01)
