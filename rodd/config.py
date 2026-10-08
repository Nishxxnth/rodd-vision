"""
rodd/config.py
Loads config.yaml into typed dataclasses.
Supports profiles (outdoor | indoor | all), per-class thresholds, and prompts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    type: str = "hybrid"
    weights: str = "yolov8s-worldv2.pt"
    large_weights: str = "yolov8l-worldv2.pt"
    coco_weights: str = "yolov8m.pt"
    custom_weights_path: str = ""
    confidence: float = 0.30
    imgsz: int = 640
    hybrid_iou: float = 0.50


@dataclass
class SpatialConfig:
    left_threshold: float = 0.33
    right_threshold: float = 0.66
    near_area_ratio: float = 0.15
    far_area_ratio: float = 0.03
    near_bottom_y: float = 0.75


@dataclass
class SpeechConfig:
    enabled: bool = True
    cooldown_seconds: float = 3.0
    dedup_window_seconds: float = 6.0
    rate: int = 150
    volume: float = 1.0


@dataclass
class SourceConfig:
    webcam_index: int = 0
    skip_frames: int = 2


@dataclass
class OutputConfig:
    save_dir: str = "outputs"
    display: bool = True
    draw_conf: bool = True
    draw_fps: bool = True


@dataclass
class RODDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: List[str] = field(default_factory=list)
    danger_order: List[str] = field(default_factory=list)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Per-class confidence thresholds (class_name -> threshold).
    # If a class is absent, model.confidence is used as fallback.
    class_thresholds: Dict[str, float] = field(default_factory=dict)
    # Descriptive prompt strings for open-vocab models (class_name -> prompt).
    prompts: Dict[str, str] = field(default_factory=dict)
    # The profile that was applied (informational)
    active_profile: str = "outdoor"
    # All profiles loaded from config (profile_name -> {classes, danger_order})
    profiles: Dict[str, dict] = field(default_factory=dict)

    def threshold_for(self, class_name: str) -> float:
        """Return the per-class confidence threshold, falling back to global."""
        return self.class_thresholds.get(class_name, self.model.confidence)

    def prompt_for(self, class_name: str) -> str:
        """Return the descriptive prompt for open-vocab querying, fallback to name."""
        return self.prompts.get(class_name, class_name)

    def prompts_for_classes(self, classes: Optional[List[str]] = None) -> List[str]:
        """Return list of prompts in same order as classes."""
        cls = classes if classes is not None else self.classes
        return [self.prompt_for(c) for c in cls]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CFG = Path(__file__).parent.parent / "config.yaml"

VALID_PROFILES = ("outdoor", "indoor", "all")


def load_config(
    path: str | Path | None = None,
    profile: Optional[str] = None,
) -> RODDConfig:
    """Load YAML config file and return RODDConfig dataclass.

    Parameters
    ----------
    path    : path to config.yaml (default: rodd_vision/config.yaml)
    profile : one of 'outdoor', 'indoor', 'all'.  Overrides active_profile
              from the yaml and sets cfg.classes / cfg.danger_order from that
              profile's section.  Pass None to use the yaml's active_profile.
    """
    cfg_path = Path(path) if path else _DEFAULT_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    cfg = RODDConfig()

    # -- model
    m = raw.get("model", {})
    cfg.model = ModelConfig(
        type=m.get("type", "hybrid"),
        weights=m.get("weights", "yolov8s-worldv2.pt"),
        large_weights=m.get("large_weights", "yolov8l-worldv2.pt"),
        coco_weights=m.get("coco_weights", "yolov8m.pt"),
        custom_weights_path=m.get("custom_weights_path", ""),
        confidence=float(m.get("confidence", 0.30)),
        imgsz=int(m.get("imgsz", 640)),
        hybrid_iou=float(m.get("hybrid_iou", 0.50)),
    )

    # -- profiles
    cfg.profiles = raw.get("profiles", {})

    # -- determine active profile
    requested_profile = profile or raw.get("active_profile", "outdoor")
    if requested_profile not in VALID_PROFILES:
        raise ValueError(
            f"Unknown profile {requested_profile!r}. "
            f"Valid choices: {VALID_PROFILES}"
        )
    cfg.active_profile = requested_profile

    # -- classes & danger_order (from profile if available, else flat list)
    profile_data = cfg.profiles.get(requested_profile, {})
    cfg.classes = profile_data.get("classes", raw.get("classes", []))
    cfg.danger_order = profile_data.get(
        "danger_order", raw.get("danger_order", [])
    )

    # -- per-class thresholds
    cfg.class_thresholds = {
        str(k): float(v)
        for k, v in raw.get("class_thresholds", {}).items()
    }

    # -- prompts
    cfg.prompts = {
        str(k): str(v)
        for k, v in raw.get("prompts", {}).items()
    }

    # -- spatial
    s = raw.get("spatial", {})
    cfg.spatial = SpatialConfig(
        left_threshold=float(s.get("left_threshold", 0.33)),
        right_threshold=float(s.get("right_threshold", 0.66)),
        near_area_ratio=float(s.get("near_area_ratio", 0.15)),
        far_area_ratio=float(s.get("far_area_ratio", 0.03)),
        near_bottom_y=float(s.get("near_bottom_y", 0.75)),
    )

    # -- speech
    sp = raw.get("speech", {})
    cfg.speech = SpeechConfig(
        enabled=bool(sp.get("enabled", True)),
        cooldown_seconds=float(sp.get("cooldown_seconds", 3.0)),
        dedup_window_seconds=float(sp.get("dedup_window_seconds", 6.0)),
        rate=int(sp.get("rate", 150)),
        volume=float(sp.get("volume", 1.0)),
    )

    # -- source
    src = raw.get("source", {})
    cfg.source = SourceConfig(
        webcam_index=int(src.get("webcam_index", 0)),
        skip_frames=int(src.get("skip_frames", 2)),
    )

    # -- output
    out = raw.get("output", {})
    cfg.output = OutputConfig(
        save_dir=str(out.get("save_dir", "outputs")),
        display=bool(out.get("display", True)),
        draw_conf=bool(out.get("draw_conf", True)),
        draw_fps=bool(out.get("draw_fps", True)),
    )

    return cfg
