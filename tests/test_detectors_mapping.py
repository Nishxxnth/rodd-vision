"""
tests/test_detectors_mapping.py
Tests for COCO->RODD class mapping, filtering, NMS, profiles, per-class
thresholds, hybrid merge, and new indoor classes.
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
    HybridDetector,
    Detection,
    class_wise_nms,
    _iou,
)
from rodd.config import load_config, VALID_PROFILES


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

    # ── Indoor class mappings ──────────────────────────────────────────

    def test_chair_maps_correctly(self):
        assert COCO_TO_RODD["chair"] == "chair"

    def test_dining_table_maps_to_table(self):
        assert COCO_TO_RODD["dining table"] == "table"

    def test_laptop_maps_correctly(self):
        assert COCO_TO_RODD["laptop"] == "laptop"

    def test_bottle_maps_correctly(self):
        assert COCO_TO_RODD["bottle"] == "bottle"

    def test_unmapped_coco_classes_not_in_mapping(self):
        for cls in ["cat", "dog", "pizza", "bicycle"]:
            assert cls not in COCO_TO_RODD, f"'{cls}' should not be in COCO_TO_RODD"

    def test_all_mapped_rodd_classes_are_valid_rodd_targets(self):
        # Use the 'all' profile which includes all classes
        cfg = load_config(profile="all")
        for coco_cls, rodd_cls in COCO_TO_RODD.items():
            assert rodd_cls in cfg.classes, (
                f"COCO class '{coco_cls}' maps to '{rodd_cls}' which is not in 'all' profile classes"
            )

    def test_coco_unmapped_contains_expected_classes(self):
        for cls in ["pit", "stair", "electric pole", "pillar"]:
            assert cls in COCO_UNMAPPED, f"'{cls}' should be in COCO_UNMAPPED"

    def test_coco_unmapped_not_in_mapping_values(self):
        mapped_values = set(COCO_TO_RODD.values())
        for cls in COCO_UNMAPPED:
            assert cls not in mapped_values, (
                f"'{cls}' is in COCO_UNMAPPED but also appears as a mapping target"
            )


# ---------------------------------------------------------------------------
# Profile and config tests
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_valid_profiles_constant(self):
        assert set(VALID_PROFILES) == {"outdoor", "indoor", "all"}

    def test_outdoor_profile_classes(self):
        cfg = load_config(profile="outdoor")
        for cls in ["building", "bus", "car", "pit", "stair", "electric pole"]:
            assert cls in cfg.classes

    def test_indoor_profile_classes(self):
        cfg = load_config(profile="indoor")
        for cls in ["chair", "table", "door", "person"]:
            assert cls in cfg.classes

    def test_indoor_profile_excludes_outdoor_only_classes(self):
        cfg = load_config(profile="indoor")
        # outdoor-specific classes should not be in indoor profile
        for cls in ["electric pole", "tram", "bus", "car"]:
            assert cls not in cfg.classes, f"'{cls}' should not be in indoor profile"

    def test_all_profile_contains_indoor_and_outdoor(self):
        cfg = load_config(profile="all")
        for cls in ["chair", "table", "building", "bus", "stair"]:
            assert cls in cfg.classes

    def test_profile_sets_active_profile_field(self):
        cfg = load_config(profile="indoor")
        assert cfg.active_profile == "indoor"

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            load_config(profile="nonexistent")

    def test_outdoor_danger_order_starts_with_pit(self):
        cfg = load_config(profile="outdoor")
        assert cfg.danger_order[0] == "pit"

    def test_indoor_danger_order_starts_with_stair(self):
        cfg = load_config(profile="indoor")
        assert cfg.danger_order[0] == "stair"


# ---------------------------------------------------------------------------
# Per-class threshold tests
# ---------------------------------------------------------------------------

class TestPerClassThresholds:
    def test_pit_has_low_threshold(self):
        cfg = load_config()
        assert cfg.threshold_for("pit") <= 0.25

    def test_car_has_higher_threshold(self):
        cfg = load_config()
        assert cfg.threshold_for("car") >= 0.40

    def test_missing_class_falls_back_to_global(self):
        cfg = load_config()
        unknown_thresh = cfg.threshold_for("__nonexistent_class__")
        assert unknown_thresh == cfg.model.confidence

    def test_stair_threshold_lower_than_car(self):
        cfg = load_config()
        assert cfg.threshold_for("stair") < cfg.threshold_for("car")


# ---------------------------------------------------------------------------
# Prompt engineering tests
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_pit_prompt_mentions_pothole(self):
        cfg = load_config()
        assert "pothole" in cfg.prompt_for("pit").lower()

    def test_electric_pole_prompt_mentions_utility(self):
        cfg = load_config()
        assert "utility" in cfg.prompt_for("electric pole").lower() or \
               "pole" in cfg.prompt_for("electric pole").lower()

    def test_missing_class_prompt_falls_back_to_class_name(self):
        cfg = load_config()
        assert cfg.prompt_for("__unknown__") == "__unknown__"

    def test_prompts_for_classes_returns_correct_length(self):
        cfg = load_config(profile="outdoor")
        prompts = cfg.prompts_for_classes()
        assert len(prompts) == len(cfg.classes)

    def test_prompts_for_classes_aligns_with_class_list(self):
        cfg = load_config(profile="outdoor")
        prompts = cfg.prompts_for_classes()
        for cls, prompt in zip(cfg.classes, prompts):
            expected = cfg.prompt_for(cls)
            assert prompt == expected


# ---------------------------------------------------------------------------
# NMS utility tests
# ---------------------------------------------------------------------------

class TestClassWiseNMS:
    def _det(self, label, conf, bbox):
        return Detection(label=label, confidence=conf, bbox=bbox)

    def test_no_overlap_keeps_all(self):
        dets = [
            self._det("car", 0.9, (0, 0, 100, 100)),
            self._det("car", 0.8, (200, 200, 300, 300)),
        ]
        out = class_wise_nms(dets, iou_threshold=0.5)
        assert len(out) == 2

    def test_high_overlap_suppresses_lower(self):
        dets = [
            self._det("car", 0.9, (0, 0, 100, 100)),
            self._det("car", 0.6, (5, 5, 105, 105)),  # high overlap
        ]
        out = class_wise_nms(dets, iou_threshold=0.5)
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.9)

    def test_different_classes_not_suppressed(self):
        dets = [
            self._det("car",  0.9, (0, 0, 100, 100)),
            self._det("bus",  0.9, (5, 5, 105, 105)),  # same bbox, different class
        ]
        out = class_wise_nms(dets, iou_threshold=0.5)
        assert len(out) == 2

    def test_iou_self_is_one(self):
        b = (10.0, 20.0, 100.0, 200.0)
        assert _iou(b, b) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_iou_partial(self):
        a = (0, 0, 100, 100)
        b = (50, 0, 150, 100)
        # Intersection: 50x100=5000; Union: 10000+10000-5000=15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, abs=0.01)

    def test_empty_input(self):
        assert class_wise_nms([], 0.5) == []


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
    def _make_detector(self, profile="outdoor"):
        cfg = load_config(profile=profile)
        cfg.model.confidence = 0.0  # disable filtering for test
        cfg.class_thresholds = {}   # disable per-class thresholds for test
        det = CocoFallbackDetector.__new__(CocoFallbackDetector)
        det.cfg = cfg
        det.conf_thresh = cfg.model.confidence
        det.imgsz = cfg.model.imgsz

        fake_model = types.SimpleNamespace()
        fake_model.names = {
            0:  "person",
            2:  "car",
            5:  "bus",
            6:  "train",
            13: "bench",
            11: "stop sign",
            58: "potted plant",
            56: "chair",
            60: "dining table",
            63: "laptop",
            39: "bottle",
            15: "cat",    # unmapped
            16: "dog",    # unmapped
        }
        fake_model.predict = lambda *a, **kw: []
        det._model = fake_model
        return det

    def test_mapped_class_included(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=0, conf=0.9, xyxy=[10, 20, 100, 200])]
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "person"

    def test_unmapped_class_excluded(self):
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
        boxes = [_FakeBox(cls_id=6, conf=0.85, xyxy=[50, 50, 400, 400])]
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "tram"

    def test_bench_maps_to_street_chair(self):
        import torch
        det = self._make_detector()
        boxes = [_FakeBox(cls_id=13, conf=0.75, xyxy=[100, 100, 200, 200])]
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "street chair"

    def test_mixed_mapped_and_unmapped(self):
        import torch
        det = self._make_detector()
        boxes = [
            _FakeBox(cls_id=0,  conf=0.9, xyxy=[10, 20, 100, 200]),   # person
            _FakeBox(cls_id=16, conf=0.8, xyxy=[200, 50, 400, 300]),   # dog -> dropped
            _FakeBox(cls_id=2,  conf=0.7, xyxy=[50, 60, 300, 400]),    # car
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

    # ── New indoor class tests ──────────────────────────────────────────

    def test_chair_maps_correctly_indoor(self):
        import torch
        det = self._make_detector(profile="indoor")
        boxes = [_FakeBox(cls_id=56, conf=0.8, xyxy=[100, 100, 300, 400])]  # chair
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "chair"

    def test_dining_table_maps_to_table_indoor(self):
        import torch
        det = self._make_detector(profile="indoor")
        boxes = [_FakeBox(cls_id=60, conf=0.75, xyxy=[50, 50, 500, 300])]  # dining table
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].label == "table"

    def test_coco_outdoor_class_excluded_in_indoor_profile(self):
        """A COCO 'car' detection should be dropped when using indoor profile."""
        import torch
        det = self._make_detector(profile="indoor")
        boxes = [_FakeBox(cls_id=2, conf=0.9, xyxy=[0, 0, 200, 200])]  # car
        _patch_coco_model(det, boxes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        # car is not in indoor profile -> should be dropped
        assert all(r.label != "car" for r in results)


# ---------------------------------------------------------------------------
# HybridDetector merge / NMS tests (mocked, no real model)
# ---------------------------------------------------------------------------

class TestHybridMerge:
    """Test the class_wise_nms merge logic used inside HybridDetector."""

    def _det(self, label, conf, bbox):
        return Detection(label=label, confidence=conf, bbox=bbox)

    def test_hybrid_merge_deduplicates(self):
        """Same class detected by both COCO and OV branches: NMS keeps best."""
        coco_dets = [self._det("person", 0.85, (10, 10, 100, 200))]
        ov_dets   = [self._det("person", 0.60, (15, 12, 105, 205))]
        merged = class_wise_nms(coco_dets + ov_dets, iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == pytest.approx(0.85)

    def test_hybrid_merge_keeps_non_overlapping(self):
        """OV-only detection (stair) + COCO detection (car): both kept."""
        dets = [
            self._det("car",   0.85, (0, 0, 100, 100)),
            self._det("stair", 0.45, (400, 300, 600, 480)),
        ]
        merged = class_wise_nms(dets, iou_threshold=0.5)
        assert len(merged) == 2
        labels = {d.label for d in merged}
        assert "car" in labels and "stair" in labels

    def test_hybrid_merge_different_classes_not_suppressed_even_if_overlapping(self):
        """Overlapping boxes of different classes must NOT suppress each other."""
        dets = [
            self._det("person", 0.9, (0, 0, 100, 200)),
            self._det("pillar", 0.8, (5, 5, 95, 195)),
        ]
        merged = class_wise_nms(dets, iou_threshold=0.5)
        assert len(merged) == 2
