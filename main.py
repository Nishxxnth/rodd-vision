"""
main.py – RODD-Vision CLI entry point
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import signal
from pathlib import Path

import cv2

# Ensure rodd_vision/ is on the path when run from its own directory
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rodd.config import load_config, VALID_PROFILES
from rodd.detectors import build_detector, _DETECTOR_MAP
from rodd.pipeline import RODDPipeline
from rodd.sources import build_source
from rodd.utils import setup_logging, save_frame


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RODD-Vision: Road Obstacle Detection & Description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python main.py --source webcam
  python main.py --source webcam   --profile indoor
  python main.py --source video    --path samples/demo.mp4
  python main.py --source image    --path samples/bus.jpg --save --mute
  python main.py --source folder   --path samples/
  python main.py --detector hybrid --profile outdoor
        """,
    )
    p.add_argument("--source", choices=["webcam", "video", "image", "folder"],
                   default="webcam", help="Input source type")
    p.add_argument("--path", type=str, default=None,
                   help="Path to video file / image / folder (not needed for webcam)")
    p.add_argument("--conf", type=float, default=None,
                   help="Detection confidence threshold (overrides config)")
    p.add_argument("--mute", action="store_true",
                   help="Disable TTS speech (still prints announcements)")
    p.add_argument("--no-display", action="store_true",
                   help="Do not show OpenCV window")
    p.add_argument("--save", action="store_true",
                   help="Save annotated frames to outputs/")
    p.add_argument(
        "--detector",
        choices=list(_DETECTOR_MAP.keys()),
        default=None,
        help="Detector backend to use (overrides config.model.type)",
    )
    p.add_argument(
        "--profile",
        choices=list(VALID_PROFILES),
        default=None,
        help="Detection profile: outdoor | indoor | all (overrides config.active_profile)",
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml (default: rodd_vision/config.yaml)")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("rodd.main")

    # --- config
    cfg = load_config(args.config, profile=args.profile)
    if args.conf is not None:
        cfg.model.confidence = args.conf

    logger.info(
        "Profile: %s | %d classes | detector: %s",
        cfg.active_profile,
        len(cfg.classes),
        args.detector or cfg.model.type,
    )

    # --- output dir
    out_dir = Path(cfg.output.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pipeline
    detector = build_detector(cfg, detector_type=args.detector)
    pipeline = RODDPipeline(cfg, detector=detector, mute=args.mute)

    # --- source
    source = build_source(
        source_type=args.source,
        path=args.path,
        webcam_index=cfg.source.webcam_index,
        skip_frames=cfg.source.skip_frames,
    )

    # --- graceful shutdown on Ctrl+C
    _running = [True]

    def _sigint(_sig, _frame):
        logger.info("Ctrl+C – shutting down …")
        _running[0] = False

    signal.signal(signal.SIGINT, _sigint)

    win_name = "RODD-Vision (q to quit)"
    frame_idx = 0
    logger.info("Starting pipeline | source=%s | detector=%s", args.source, detector.name)

    try:
        for frame in source:
            if not _running[0]:
                break

            annotated, annotations, inf_ms = pipeline.process(frame)

            # Display
            if not args.no_display:
                cv2.imshow(win_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed – quitting")
                    break

            # Save
            if args.save:
                stem = args.path if args.path else "webcam"
                stem = Path(stem).stem if args.source != "webcam" else "webcam"
                out_name = f"{stem}_frame{frame_idx:04d}.jpg"
                out_path = str(out_dir / out_name)
                save_frame(annotated, out_path)
                logger.info("Saved → %s", out_path)

            frame_idx += 1

    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    finally:
        pipeline.shutdown()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Done. %d frames processed.", frame_idx)


if __name__ == "__main__":
    main()
