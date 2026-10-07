"""
rodd/pipeline.py
Ties together: detector → spatial annotator → speech engine.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .config import RODDConfig
from .detectors import BaseDetector, Detection, build_detector
from .spatial import SpatialAnnotation, annotate, top_priority
from .speech import SpeechEngine
from .utils import draw_detections, FPSCounter

logger = logging.getLogger(__name__)


class RODDPipeline:
    """
    Complete RODD inference pipeline.

    Usage::

        pipeline = RODDPipeline(cfg)
        for frame in source:
            annotated_frame, annotations = pipeline.process(frame)
    """

    def __init__(
        self,
        cfg: RODDConfig,
        detector: Optional[BaseDetector] = None,
        speech: Optional[SpeechEngine] = None,
        mute: bool = False,
    ):
        self.cfg = cfg
        self.detector: BaseDetector = detector or build_detector(cfg)
        self.speech: SpeechEngine = speech or SpeechEngine.from_config(cfg)
        if mute:
            self.speech.mute()
        self._fps = FPSCounter()

    # ------------------------------------------------------------------
    def process(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, List[SpatialAnnotation], float]:
        """
        Run a single frame through the full pipeline.

        Returns
        -------
        annotated_frame : frame with bounding boxes drawn
        annotations     : list of SpatialAnnotation (sorted by priority)
        inference_ms    : detector inference time in milliseconds
        """
        h, w = frame.shape[:2]

        # 1. Detect
        t0 = time.perf_counter()
        detections: List[Detection] = self.detector.detect(frame)
        inference_ms = (time.perf_counter() - t0) * 1000

        # 2. Spatial annotation
        annotations = annotate(detections, w, h, self.cfg)

        # 3. Speak top priority
        top = top_priority(annotations)
        if top is not None:
            self.speech.announce(top.announce_text())

        # 4. Draw
        fps = self._fps.tick()
        annotated = draw_detections(frame.copy(), annotations, fps, inference_ms, self.cfg)

        return annotated, annotations, inference_ms

    def shutdown(self) -> None:
        self.speech.shutdown()
