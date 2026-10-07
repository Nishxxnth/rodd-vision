"""
rodd/config.py
Loads config.yaml into typed dataclasses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    type: str = "yoloworld"
    weights: str = "yolov8s-worldv2.pt"
    custom_weights_path: str = ""
    confidence: float = 0.4
    imgsz: int = 640


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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CFG = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> RODDConfig:
    """Load YAML config file and return RODDConfig dataclass."""
    cfg_path = Path(path) if path else _DEFAULT_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    cfg = RODDConfig()

    # -- model
    m = raw.get("model", {})
    cfg.model = ModelConfig(
        type=m.get("type", "yoloworld"),
        weights=m.get("weights", "yolov8s-worldv2.pt"),
        custom_weights_path=m.get("custom_weights_path", ""),
        confidence=float(m.get("confidence", 0.4)),
        imgsz=int(m.get("imgsz", 640)),
    )

    # -- classes
    cfg.classes = raw.get("classes", [])

    # -- danger_order
    cfg.danger_order = raw.get("danger_order", [])

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
