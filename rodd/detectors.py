"""
rodd/detectors.py
BaseDetector, YoloWorldDetector, CocoFallbackDetector, CustomWeightsDetector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
}

# Classes from the 14 that the COCO fallback cannot detect
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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} conf={self.conf_thresh}>"


# ---------------------------------------------------------------------------
# YOLO-World detector (default)
# ---------------------------------------------------------------------------

class YoloWorldDetector(BaseDetector):
    """
    Uses ultralytics YOLOWorld model with set_classes() for zero-shot detection
    of the 14 RODD target classes.
    """

    name = "yoloworld"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLOWorld
        logger.info("Loading YOLOWorld model: %s", cfg.model.weights)
        self._model = YOLOWorld(cfg.model.weights)
        self._model.set_classes(cfg.classes)
        logger.info("YOLOWorld ready. Classes: %s", cfg.classes)

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
                label = self.cfg.classes[cls_id] if cls_id < len(self.cfg.classes) else "unknown"
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections


# ---------------------------------------------------------------------------
# COCO Fallback detector
# ---------------------------------------------------------------------------

class CocoFallbackDetector(BaseDetector):
    """
    Uses YOLOv8n trained on COCO-80. Maps a subset of COCO classes to RODD
    classes. Logs which RODD classes this cannot detect.
    """

    name = "coco"

    def __init__(self, cfg: RODDConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        logger.info("Loading COCO fallback model: yolov8n.pt")
        self._model = YOLO("yolov8n.pt")
        logger.warning(
            "Using COCO fallback detector. The following RODD classes "
            "CANNOT be detected: %s",
            ", ".join(COCO_UNMAPPED),
        )

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
                coco_label = self._model.names.get(cls_id, "")
                rodd_label = COCO_TO_RODD.get(coco_label)
                if rodd_label is None:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label=rodd_label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections


# ---------------------------------------------------------------------------
# Custom weights detector
# ---------------------------------------------------------------------------

class CustomWeightsDetector(BaseDetector):
    """
    Loads a user-supplied best.pt from config.model.custom_weights_path.
    Expects the model to be trained on the 14 RODD classes.
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

    if det_type == "yoloworld":
        try:
            return YoloWorldDetector(cfg)
        except Exception as exc:
            logger.warning(
                "YoloWorldDetector failed to load (%s). "
                "Falling back to CocoFallbackDetector.",
                exc,
            )
            return CocoFallbackDetector(cfg)

    elif det_type == "coco":
        return CocoFallbackDetector(cfg)

    elif det_type == "custom":
        if not cfg.model.custom_weights_path:
            raise ValueError("custom_weights_path must be set in config for --detector custom")
        if not os.path.isfile(cfg.model.custom_weights_path):
            raise FileNotFoundError(f"Custom weights not found: {cfg.model.custom_weights_path}")
        return CustomWeightsDetector(cfg)

    else:
        raise ValueError(f"Unknown detector type: {det_type!r}")
