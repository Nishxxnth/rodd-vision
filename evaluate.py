"""
evaluate.py – Batch evaluation over a folder.
Saves annotated images to outputs/ and prints a per-class summary.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rodd.config import load_config
from rodd.detectors import build_detector
from rodd.pipeline import RODDPipeline
from rodd.sources import folder_source, image_source, IMAGE_EXTS
from rodd.utils import setup_logging, save_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RODD-Vision Batch Evaluator")
    p.add_argument("--folder", required=True, help="Input folder with images")
    p.add_argument("--detector", choices=["yoloworld", "coco", "custom"], default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("rodd.evaluate")

    cfg = load_config(args.config)
    if args.conf:
        cfg.model.confidence = args.conf
    cfg.speech.enabled = False  # no audio during batch evaluation

    out_dir = Path(cfg.output.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(cfg, detector_type=args.detector)
    pipeline = RODDPipeline(cfg, detector=detector, mute=True)

    folder = Path(args.folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        logger.error("No images found in %s", folder)
        sys.exit(1)

    logger.info("Evaluating %d images from %s …", len(images), folder)

    class_counts: dict[str, int] = defaultdict(int)
    inference_times: list[float] = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read %s – skipping", img_path.name)
            continue

        annotated, annotations, inf_ms = pipeline.process(frame)
        inference_times.append(inf_ms)

        for ann in annotations:
            class_counts[ann.label] += 1

        out_path = out_dir / f"eval_{img_path.stem}.jpg"
        save_frame(annotated, str(out_path))
        logger.info("  %s → %d detections (%.0f ms) → %s",
                    img_path.name, len(annotations), inf_ms, out_path.name)

    pipeline.shutdown()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RODD-Vision Evaluation Summary")
    print("=" * 60)
    print(f"Images processed : {len(inference_times)}")
    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times)
        print(f"Avg inference    : {avg_ms:.1f} ms / image")
    print("\nPer-class detection counts:")
    print(f"  {'Class':<20} {'Count':>6}")
    print(f"  {'-'*20} {'-'*6}")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c]):
        print(f"  {cls:<20} {class_counts[cls]:>6}")
    if not class_counts:
        print("  (no detections)")
    print(f"\nAnnotated images saved to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
