"""
setup_bench.py – Build bench_images/ for RODD-Vision benchmark

Strategy:
  1. Copy samples/ images to appropriate class folders.
  2. Download known-good images from Ultralytics/Unsplash (no-key required).
  3. For classes with no real images yet, generate a synthetic labeled image
     (just a coloured rectangle with text) as a known-miss placeholder — this
     honestly records a miss in the baseline and shows the limitation.

Run once:   python setup_bench.py
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
BENCH = ROOT / "bench_images"
SAMPLES = ROOT / "samples"


# ---------------------------------------------------------------------------
# Step 1 – Copy existing sample images that match known classes
# ---------------------------------------------------------------------------

SAMPLE_COPIES = [
    # (src relative to samples/, class_name, dest filename)
    ("bus.jpg",        "bus",     "bus_01.jpg"),
    ("zidane.jpg",     "person",  "person_01.jpg"),
    ("pothole.jpg",    "pit",     "pit_01.jpg"),
    ("stairs.jpg",     "stair",   "stair_01.jpg"),
    ("car_street.jpg", "car",     "car_01.jpg"),
    ("street.jpg",     "car",     "car_02.jpg"),       # street has cars
    ("street.jpg",     "tree",    "tree_01.jpg"),      # street has trees
    ("street.jpg",     "building","building_01.jpg"),  # street has buildings
    ("street.jpg",     "person",  "person_02.jpg"),    # street has people
]


def copy_samples():
    for src_name, cls_name, dst_name in SAMPLE_COPIES:
        src = SAMPLES / src_name
        dst_dir = BENCH / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / dst_name
        if not src.exists():
            print(f"  [SKIP] {src_name} not found in samples/")
            continue
        if dst.exists():
            print(f"  [SKIP] {cls_name}/{dst_name} already exists")
            continue
        shutil.copy2(src, dst)
        print(f"  [COPY] {src_name} -> {cls_name}/{dst_name}")


# ---------------------------------------------------------------------------
# Step 2 – Download reliable images
# ---------------------------------------------------------------------------

DOWNLOADS = [
    # bus
    ("https://ultralytics.com/images/bus.jpg",            "bus",          "bus_02.jpg"),
    # person
    ("https://ultralytics.com/images/zidane.jpg",         "person",       "person_03.jpg"),
    # tram – Wikipedia direct image (small reliable)
    ("https://upload.wikimedia.org/wikipedia/commons/9/96/Tram_in_Prague.jpg",
                                                           "tram",         "tram_01.jpg"),
    # road sign – Wikipedia SVG-rendered PNG (use commons REST API)
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Stop_sign_MUTCD.svg/480px-Stop_sign_MUTCD.svg.png",
                                                           "road sign",    "roadsign_01.jpg"),
]


def download_one(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 rodd-bench/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        if len(data) < 500:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"    [FAIL] {exc}")
        return False


def download_images():
    for url, cls_name, fname in DOWNLOADS:
        dst_dir = BENCH / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / fname
        if dst.exists() and dst.stat().st_size > 500:
            print(f"  [SKIP] {cls_name}/{fname}")
            continue
        print(f"  [DOWN] {cls_name}/{fname} … ", end="", flush=True)
        ok = download_one(url, dst)
        print("OK" if ok else "FAILED")


# ---------------------------------------------------------------------------
# Step 3 – Generate synthetic images for classes we can't easily download
# ---------------------------------------------------------------------------

# (class_name, background BGR, text, n_images_to_generate)
SYNTHETIC = [
    ("stair",         (80, 60, 40),   "STAIRS",        2),
    ("door",          (120, 80, 60),  "DOOR",          3),
    ("electric pole", (50, 50, 50),   "UTILITY POLE",  3),
    ("pillar",        (90, 90, 100),  "PILLAR",        3),
    ("dustbin",       (30, 80, 30),   "DUSTBIN",       3),
    ("tram",          (20, 60, 140),  "TRAM",          2),
    ("road sign",     (0, 0, 180),    "ROAD SIGN",     2),
    ("street chair",  (80, 60, 20),   "BENCH",         3),
    ("building",      (100, 100, 140),"BUILDING",      2),
    ("tree",          (30, 100, 30),  "TREE",          2),
    ("chair",         (60, 40, 20),   "CHAIR",         3),
    ("table",         (80, 70, 50),   "TABLE",         3),
    ("car",           (20, 20, 120),  "CAR",           1),
    ("bus",           (20, 80, 20),   "BUS",           1),
    ("person",        (150, 120, 100),"PERSON",        1),
    ("pit",           (40, 30, 20),   "POTHOLE",       2),
]


def make_synthetic(cls_name: str, bg: tuple, text: str, n: int):
    """Create n synthetic labelled images for a class."""
    dst_dir = BENCH / cls_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        fname = f"synthetic_{i+1:02d}.jpg"
        dst = dst_dir / fname
        if dst.exists():
            continue
        h, w = 480, 640
        img = np.full((h, w, 3), bg, dtype=np.uint8)
        # Draw a centred rectangle representing the object
        pad = 80
        cv2.rectangle(img, (pad, pad), (w - pad, h - pad), (200, 200, 200), 3)
        # Put class label
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, f"[synthetic #{i+1}]", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imwrite(str(dst), img)
    print(f"  [SYN ] {cls_name}: generated up to {n} synthetic image(s)")


def generate_synthetic():
    for cls_name, bg, text, n in SYNTHETIC:
        dst_dir = BENCH / cls_name
        # Only generate if we have fewer than 3 real images
        existing = list(dst_dir.glob("*.jpg")) + list(dst_dir.glob("*.png"))
        real = [f for f in existing if "synthetic" not in f.name]
        needed = max(0, 3 - len(real))
        if needed == 0:
            continue
        make_synthetic(cls_name, bg, text, needed)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n── Benchmark image summary ──────────────────────────────")
    total = 0
    for cls_dir in sorted(BENCH.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        real = [f for f in imgs if "synthetic" not in f.name]
        syn  = [f for f in imgs if "synthetic" in f.name]
        total += len(imgs)
        print(f"  {cls_dir.name:<18} {len(imgs):>3} images  "
              f"({len(real)} real, {len(syn)} synthetic)")
    print(f"  {'TOTAL':<18} {total:>3} images")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    print("Setting up bench_images/ …\n")
    print("Step 1 – Copying sample images …")
    copy_samples()
    print("\nStep 2 – Downloading additional images …")
    download_images()
    print("\nStep 3 – Generating synthetic images for missing classes …")
    generate_synthetic()
    print_summary()
    print("\nDone! Run:  python bench_check.py --detector yoloworld")
