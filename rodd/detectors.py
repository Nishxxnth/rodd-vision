"""
rodd/detectors.py
Detectors: YoloWorldDetector, YoloWorldLargeDetector, CocoDetector,
           HybridDetector, CustomWeightsDetector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import RODDConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]


# ---------------------------------------------------------------------------
# COCO → RODD class mapping
# ---------------------------------------------------------------------------

COCO_TO_RODD: dict[str, str] = {
    "person":        "person",
    "car":           "car",
    "bus":           "bus",
    "train":         "tram",
    "bench":         "street chair",
    "stop sign":     "road sign",
    "potted plant":  "tree",
    "chair":         "chair",
    "dining table":  "table",
    "laptop":        "laptop",
    "bottle":        "bottle",
}

# Classes from the target set that the COCO fallback cannot detect
COCO_UNMAPPED = [
    "building",
    "door",
    "dustbin",
    "electric pole",
    "pillar",
    "pit",
    "stair",
]


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------

def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_wise_nms(
    detections: List[Detection],
    iou_threshold: float = 0.50,
) -> List[Detection]:
    """
    Per-class greedy NMS.  Detections for different classes are never merged.
    Within each class, overlapping boxes with IoU >= iou_threshold are
    suppressed (lowest-confidence box is dropped).
    """
    # Group by label
    by_label: Dict[str, List[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    result: List[Detection] = []
    for label, dets in by_label.items():
        dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        suppressed = [False] * len(dets_sorted)
        for i, di in enumerate(dets_sorted):
            if suppressed[i]:
                continue
            kept.append(di)
            for j in range(i + 1, len(dets_sorted)):
                if suppressed[j]:
                    continue
                if _iou(di.bbox, dets_sorted[j].bbox) >= iou_threshold:
                    suppressed[j] = True
        result.extend(kept)
    return result


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector:
    """Abstract base detector. Subclasses must implement detect()."""

    name: str = "base"

    def __init__(self, cfg: RODDConfig):
        self.cfg = cfg
        self.conf_thresh = cfg.model.confidence
        self.imgsz = cfg.model.imgsz

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def _apply_class_threshold(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections using per-class thresholds from config."""
        return [
            d for d in detections
            if d.confidence >= self.cfg.threshold_for(d.label)
        ]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} conf={self.conf_thresh}>"


# ---------------------------------------------------------------------------
# YOLO-World detector (small – yolov8s-worldv2, default open-vocab)
# ---------------------------------------------------------------------------

class YoloWorldDetector(BaseDetector):
    """
    Uses ultralytics YOLOWorld (small) with set_classes() for zero-shot
    detection.  Classes are set from cfg.classes; prompts are used as the
    text descriptions to maximise zero-shot accuracy.
    """

    name = "yoloworld"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLOWorld
        logger.info("Loading YOLOWorld model: %s", cfg.model.weights)
        self._model = YOLOWorld(cfg.model.weights)
        # Use descriptive prompts as class labels for better zero-shot accuracy
        prompts = cfg.prompts_for_classes()
        self._model.set_classes(prompts)
        self._classes = cfg.classes  # original names for labeling
        logger.info("YOLOWorld ready. Classes: %s", cfg.classes)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=self.conf_thresh / 2,  # use low threshold, filter per-class below
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = self._classes[cls_id] if cls_id < len(self._classes) else "unknown"
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return self._apply_class_threshold(detections)


# ---------------------------------------------------------------------------
# YOLO-World Large detector
# ---------------------------------------------------------------------------

class YoloWorldLargeDetector(BaseDetector):
    """
    Same as YoloWorldDetector but uses the larger yolov8l-worldv2 model.
    Better accuracy, ~2x slower on CPU.
    """

    name = "yoloworld_large"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLOWorld
        logger.info("Loading YOLOWorld-Large model: %s", cfg.model.large_weights)
        self._model = YOLOWorld(cfg.model.large_weights)
        prompts = cfg.prompts_for_classes()
        self._model.set_classes(prompts)
        self._classes = cfg.classes
        logger.info("YOLOWorld-Large ready.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=self.conf_thresh / 2,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = self._classes[cls_id] if cls_id < len(self._classes) else "unknown"
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return self._apply_class_threshold(detections)


# ---------------------------------------------------------------------------
# COCO detector (YOLOv8m – better baseline than nano)
# ---------------------------------------------------------------------------

class CocoFallbackDetector(BaseDetector):
    """
    Uses YOLOv8m trained on COCO-80.  Maps a subset of COCO classes to RODD
    classes via COCO_TO_RODD, including new indoor classes (chair, table,
    laptop, bottle). Logs which RODD classes this cannot detect.
    """

    name = "coco"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        weights = cfg.model.coco_weights or "yolov8m.pt"
        logger.info("Loading COCO detector: %s", weights)
        self._model = YOLO(weights)
        logger.warning(
            "Using COCO detector. The following RODD classes "
            "CANNOT be detected: %s",
            ", ".join(COCO_UNMAPPED),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=self.conf_thresh / 2,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                coco_label = self._model.names.get(cls_id, "")
                rodd_label = COCO_TO_RODD.get(coco_label)
                if rodd_label is None:
                    continue
                # Only emit if the mapped label is in the active class list
                if rodd_label not in self.cfg.classes:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=rodd_label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return self._apply_class_threshold(detections)


# ---------------------------------------------------------------------------
# Hybrid detector (COCO for mapped classes + YOLO-World for rest)
# ---------------------------------------------------------------------------

# Classes that the COCO model covers well via COCO_TO_RODD
_HYBRID_COCO_CLASSES = set(COCO_TO_RODD.values())

# Default set of classes for the open-vocab branch when using hybrid mode
_HYBRID_OV_CLASSES = [
    "building",
    "door",
    "dustbin",
    "electric pole",
    "pillar",
    "pit",
    "stair",
]


class HybridDetector(BaseDetector):
    """
    Hybrid detector: runs a COCO YOLOv8m for classes it covers well (person,
    car, bus, tram, street chair, road sign, tree, chair, table, laptop,
    bottle) and YOLO-World large for the remaining hard zero-shot classes
    (building, door, dustbin, electric pole, pillar, pit, stair).

    Outputs are merged and deduplicated with per-class NMS (IoU 0.50).
    """

    name = "hybrid"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLO, YOLOWorld

        # ── COCO branch
        coco_weights = cfg.model.coco_weights or "yolov8m.pt"
        logger.info("Hybrid: loading COCO branch: %s", coco_weights)
        self._coco = YOLO(coco_weights)

        # ── Open-vocab branch: only classes not covered by COCO
        ov_classes = [c for c in cfg.classes if c not in _HYBRID_COCO_CLASSES]
        if not ov_classes:
            ov_classes = _HYBRID_OV_CLASSES
        self._ov_classes = ov_classes
        ov_prompts = [cfg.prompt_for(c) for c in ov_classes]

        ov_weights = cfg.model.large_weights or "yolov8l-worldv2.pt"
        logger.info(
            "Hybrid: loading OV branch: %s  classes: %s",
            ov_weights, ov_classes,
        )
        self._ov = YOLOWorld(ov_weights)
        self._ov.set_classes(ov_prompts)
        self._ov_class_list = ov_classes

        self._iou_thresh = cfg.model.hybrid_iou
        logger.info("HybridDetector ready.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []

        # ── COCO branch
        try:
            coco_res = self._coco.predict(
                frame, conf=self.conf_thresh / 2, imgsz=self.imgsz, verbose=False
            )
            for r in coco_res:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    coco_label = self._coco.names.get(cls_id, "")
                    rodd_label = COCO_TO_RODD.get(coco_label)
                    if rodd_label is None or rodd_label not in self.cfg.classes:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(label=rodd_label, confidence=conf, bbox=(x1, y1, x2, y2)))
        except Exception as exc:
            logger.warning("Hybrid COCO branch error: %s", exc)

        # ── Open-vocab branch
        try:
            ov_res = self._ov.predict(
                frame, conf=self.conf_thresh / 2, imgsz=self.imgsz, verbose=False
            )
            for r in ov_res:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id < len(self._ov_class_list):
                        label = self._ov_class_list[cls_id]
                    else:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        except Exception as exc:
            logger.warning("Hybrid OV branch error: %s", exc)

        # ── Per-class threshold filtering
        detections = self._apply_class_threshold(detections)

        # ── Merge with class-wise NMS
        detections = class_wise_nms(detections, self._iou_thresh)

        return detections


# ---------------------------------------------------------------------------
# Custom weights detector
# ---------------------------------------------------------------------------

class CustomWeightsDetector(BaseDetector):
    """
    Loads a user-supplied best.pt from config.model.custom_weights_path.
    Expects the model to be trained on the active RODD class set.
    """

    name = "custom"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        weights = cfg.model.custom_weights_path
        logger.info("Loading custom weights: %s", weights)
        self._model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = self._model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DETECTOR_MAP = {
    "yoloworld":       YoloWorldDetector,
    "yoloworld_large": YoloWorldLargeDetector,
    "coco":            CocoFallbackDetector,
    "hybrid":          HybridDetector,
    "custom":          CustomWeightsDetector,
}


def build_detector(cfg: RODDConfig, detector_type: Optional[str] = None) -> BaseDetector:
    """
    Build and return the appropriate detector.

    Priority:
      1. CustomWeightsDetector (if custom_weights_path exists on disk)
      2. Requested detector_type (or cfg.model.type)
      3. Automatic fallback to CocoFallbackDetector on any load error
    """
    import os

    # Custom weights override
    if cfg.model.custom_weights_path and os.path.isfile(cfg.model.custom_weights_path):
        logger.info("Custom weights found – using CustomWeightsDetector")
        try:
            return CustomWeightsDetector(cfg)
        except Exception as exc:
            logger.error("CustomWeightsDetector failed: %s – falling back", exc)

    det_type = detector_type or cfg.model.type

    if det_type not in _DETECTOR_MAP:
        raise ValueError(
            f"Unknown detector type: {det_type!r}. "
            f"Valid choices: {list(_DETECTOR_MAP)}"
        )

    if det_type == "custom":
        if not cfg.model.custom_weights_path:
            raise ValueError("custom_weights_path must be set in config for --detector custom")
        if not os.path.isfile(cfg.model.custom_weights_path):
            raise FileNotFoundError(f"Custom weights not found: {cfg.model.custom_weights_path}")
        return CustomWeightsDetector(cfg)

    try:
        return _DETECTOR_MAP[det_type](cfg)
    except Exception as exc:
        logger.warning(
            "%s failed to load (%s). Falling back to CocoFallbackDetector.",
            det_type, exc,
        )
        return CocoFallbackDetector(cfg)
