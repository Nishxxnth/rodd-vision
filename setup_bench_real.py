import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
BENCH = ROOT / "bench_images"
SAMPLES = ROOT / "samples"
SOURCES_FILE = BENCH / "SOURCES.md"

CLASSES = [
    "building", "bus", "car", "door", "dustbin", "electric pole", 
    "person", "pillar", "pit", "road sign", "stair", "street chair", 
    "tram", "tree", "chair", "table", "bottle", "laptop"
]

def main():
    if BENCH.exists():
        shutil.rmtree(BENCH)
    BENCH.mkdir(exist_ok=True)
    
    # Map our existing real samples to classes
    existing_samples = {
        "bus": [SAMPLES / "bus.jpg"],
        "car": [SAMPLES / "car_street.jpg"],
        "person": [SAMPLES / "bus.jpg"],  # contains people
        "pit": [SAMPLES / "pothole.jpg"],
        "stair": [SAMPLES / "stairs.jpg"],
        "tree": [SAMPLES / "car_street.jpg"], # contains trees
    }
    
    sources_lines = [
        "# Benchmark Image Sources",
        "",
        "Automated downloading of 8 images per class from Wikimedia Commons and DuckDuckGo failed due to API rate limits (HTTP 429) and IP blocks (HTTP 403).",
        "",
        "**Action Required**: Please manually download 8 real photographs for the following classes into their respective folders.",
        "Do not use generated or synthetic images.",
        ""
    ]
    
    print("Setting up bench_images/ with available real photos...")
    
    for cls in CLASSES:
        cls_dir = BENCH / cls
        cls_dir.mkdir(exist_ok=True)
        sources_lines.append(f"## {cls}")
        
        count = 0
        if cls in existing_samples:
            for i, src in enumerate(existing_samples[cls]):
                if src.exists():
                    dest = cls_dir / f"{cls}_{i+1:02d}.jpg"
                    shutil.copy2(src, dest)
                    count += 1
                    
        if count > 0:
            sources_lines.append(f"- Copied {count} real sample(s) from `samples/`")
            print(f"  [OK] {cls}: Copied {count} real samples.")
        
        if count < 8:
            missing = 8 - count
            sources_lines.append(f"- **Missing {missing} images**. Please download freely licensed photos (e.g. Unsplash, Pexels, Wikimedia Commons) and place them in `bench_images/{cls}/`.")
            print(f"  [WARN] {cls}: Missing {missing} images. Please add manually.")
            
        sources_lines.append("")
        
    with open(SOURCES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sources_lines))
        
    print(f"\nDone. Sources written to {SOURCES_FILE}")

if __name__ == "__main__":
    main()
