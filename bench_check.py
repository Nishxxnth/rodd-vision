"""
bench_check.py – RODD-Vision Benchmark Presence Check
======================================================
Runs a detector on every image in bench_images/ and records, per class,
whether the expected class was detected (hit/miss), the best confidence
seen, and any false-positive classes.

This is a *presence check*, not mAP: we ask "did the detector fire at
least once for the expected class on images labelled for that class?"

Usage
-----
  python bench_check.py                          # use config default detector
  python bench_check.py --detector yoloworld
  python bench_check.py --detector yoloworld_large
  python bench_check.py --detector coco
  python bench_check.py --detector hybrid        # recommended
  python bench_check.py --detector hybrid --profile outdoor
  python bench_check.py --all                    # run all detectors, print comparison

Folder layout
-------------
  bench_images/
    <class_name>/          # one sub-folder per class
      image1.jpg
      image2.jpg
      ...

Results are printed as a table and also saved to results/bench_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rodd.config import load_config, VALID_PROFILES
from rodd.detectors import build_detector, _DETECTOR_MAP, Detection


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

BENCH_DIR = _ROOT / "bench_images"
RESULTS_DIR = _ROOT / "results"


def load_bench_images() -> Dict[str, List[Path]]:
    """Return {class_name: [image_path, ...]} for all sub-folders."""
    if not BENCH_DIR.exists():
        raise FileNotFoundError(
            f"Benchmark directory not found: {BENCH_DIR}\n"
            "Run:  python bench_check.py --download  to fetch sample images."
        )
    bench: Dict[str, List[Path]] = {}
    for cls_dir in sorted(BENCH_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if images:
            bench[cls_dir.name] = images
    return bench


def run_detector_on_bench(
    bench: Dict[str, List[Path]],
    detector,
    cfg,
) -> Dict[str, dict]:
    """
    For each class folder, run detector on each image.
    Returns {class_name: {hits, misses, best_conf, false_positives_set, timings}}.
    """
    results: Dict[str, dict] = {}
    total_images = sum(len(v) for v in bench.values())
    processed = 0

    for cls_name, images in bench.items():
        hits = 0
        misses = 0
        best_conf = 0.0
        fps: Dict[str, float] = defaultdict(float)  # false-positive label -> max conf
        timings: List[float] = []

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [WARN] Cannot read {img_path}, skipping.")
                misses += 1
                processed += 1
                continue

            t0 = time.perf_counter()
            detections: List[Detection] = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)

            # Check if expected class was detected
            matched = [d for d in detections if d.label == cls_name]
            if matched:
                hits += 1
                best_conf = max(best_conf, max(d.confidence for d in matched))
            else:
                misses += 1

            # Collect false positives (other classes detected on this image)
            for d in detections:
                if d.label != cls_name:
                    fps[d.label] = max(fps.get(d.label, 0.0), d.confidence)

            processed += 1
            print(
                f"\r  [{processed}/{total_images}] {cls_name}/{img_path.name} "
                f"→ {'HIT' if matched else 'miss'} {elapsed_ms:.0f}ms   ",
                end="",
                flush=True,
            )

        results[cls_name] = {
            "hits": hits,
            "misses": misses,
            "total": hits + misses,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0.0,
            "best_conf": round(best_conf, 3),
            "false_positives": dict(sorted(fps.items(), key=lambda x: -x[1])),
            "avg_ms": round(sum(timings) / len(timings), 1) if timings else 0.0,
        }

    print()  # newline after progress
    return results


def print_table(results: Dict[str, dict], detector_name: str, profile: str) -> None:
    """Print a formatted hit-rate table."""
    header = f"\n{'─'*70}\n  Detector: {detector_name}   Profile: {profile}\n{'─'*70}"
    print(header)
    fmt = "  {:<18} {:>5} {:>5} {:>8}  {:>8}  {}"
    print(fmt.format("Class", "Hits", "Tot", "Hit-Rate", "BestConf", "False-pos (top-3)"))
    print("  " + "─" * 68)

    all_hits = 0
    all_total = 0
    all_ms = []

    for cls_name, r in sorted(results.items()):
        hit_rate_str = f"{r['hit_rate']:.0%}"
        fp_top3 = ", ".join(
            f"{k}({v:.2f})" for k, v in list(r["false_positives"].items())[:3]
        )
        print(fmt.format(
            cls_name[:18],
            r["hits"], r["total"],
            hit_rate_str,
            f"{r['best_conf']:.3f}",
            fp_top3 or "—",
        ))
        all_hits += r["hits"]
        all_total += r["total"]
        all_ms.append(r["avg_ms"])

    print("  " + "─" * 68)
    overall = all_hits / all_total if all_total > 0 else 0.0
    avg_ms = sum(all_ms) / len(all_ms) if all_ms else 0.0
    print(f"  {'OVERALL':<18} {all_hits:>5} {all_total:>5} {overall:>8.0%}  "
          f"{'':>8}  avg {avg_ms:.0f}ms/image")
    print("─" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RODD-Vision benchmark presence check")
    p.add_argument(
        "--detector",
        choices=list(_DETECTOR_MAP.keys()),
        default=None,
        help="Detector to benchmark (default: from config.model.type)",
    )
    p.add_argument(
        "--profile",
        choices=list(VALID_PROFILES),
        default="outdoor",
        help="Detection profile (default: outdoor)",
    )
    p.add_argument(
        "--classes-profile",
        choices=list(VALID_PROFILES),
        default=None,
        help="Test only the classes in this profile (e.g. indoor). Defaults to --profile.",
    )
    p.add_argument(
        "--all", dest="run_all", action="store_true",
        help="Run all detectors and print a comparison table",
    )
    p.add_argument(
        "--conf", type=float, default=None,
        help="Override global confidence threshold",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save results to results/bench_results.json",
    )
    return p.parse_args()


def benchmark_one(detector_name: str, profile: str, classes_profile: str | None, conf_override: float | None) -> Dict[str, dict]:
    cfg = load_config(profile=profile)
    if conf_override is not None:
        cfg.model.confidence = conf_override
        
    target_classes_cfg = load_config(profile=classes_profile or profile)
    target_classes = set(target_classes_cfg.classes)
    
    print(f"\n[bench] Loading detector: {detector_name}  profile={profile}")
    try:
        detector = build_detector(cfg, detector_type=detector_name)
    except Exception as exc:
        print(f"  ERROR loading {detector_name}: {exc}")
        return {}
        
    bench = load_bench_images()
    # Filter to only the target classes
    bench = {k: v for k, v in bench.items() if k in target_classes}
    
    print(f"[bench] Found {sum(len(v) for v in bench.values())} images "
          f"across {len(bench)} classes.")
    results = run_detector_on_bench(bench, detector, cfg)
    print_table(results, detector_name, profile)
    return results


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        all_results = {}
        for det_name in _DETECTOR_MAP:
            if det_name == "custom":
                continue  # skip – needs user-supplied weights
            r = benchmark_one(det_name, args.profile, args.classes_profile, args.conf)
            if r:
                all_results[det_name] = r

        # Comparison summary
        print("\n" + "=" * 70)
        print("  DETECTOR COMPARISON – Overall Hit-Rate")
        print("=" * 70)
        fmt = "  {:<20} {:>10}  {:>10}"
        print(fmt.format("Detector", "Hit-Rate", "Avg ms"))
        print("  " + "─" * 46)
        for det_name, res in all_results.items():
            hits = sum(r["hits"] for r in res.values())
            total = sum(r["total"] for r in res.values())
            avg_ms = (
                sum(r["avg_ms"] for r in res.values()) / len(res)
                if res else 0.0
            )
            overall = hits / total if total > 0 else 0.0
            print(fmt.format(det_name, f"{overall:.0%}", f"{avg_ms:.0f}ms"))
        print("=" * 70)

        if args.save:
            out_path = RESULTS_DIR / "bench_results_all.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved → {out_path}")
    else:
        det_name = args.detector or "hybrid"
        r = benchmark_one(det_name, args.profile, args.classes_profile, args.conf)
        if r and args.save:
            out_path = RESULTS_DIR / f"bench_results_{det_name}.json"
            with open(out_path, "w") as f:
                json.dump(r, f, indent=2)
            print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
