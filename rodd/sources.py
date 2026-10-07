"""
rodd/sources.py
Frame sources: webcam, video file, single image, image folder.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Iterable, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Base generator protocol
# ---------------------------------------------------------------------------

FrameSource = Generator[np.ndarray, None, None]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def webcam_source(
    index: int = 0,
    skip_frames: int = 1,
) -> FrameSource:
    """Yield frames from a webcam."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        msg = (
            f"Cannot open webcam at index {index}. "
            "Check that a camera is connected and not in use by another application."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Webcam opened (index=%d, skip=%d)", index, skip_frames)
    frame_n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Webcam read failed – stream ended?")
                break
            frame_n += 1
            if frame_n % max(skip_frames, 1) == 0:
                yield frame
    finally:
        cap.release()
        logger.info("Webcam released")


def video_source(
    path: str | Path,
    skip_frames: int = 1,
) -> FrameSource:
    """Yield frames from a video file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    logger.info("Video opened: %s (skip=%d)", path.name, skip_frames)
    frame_n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1
            if frame_n % max(skip_frames, 1) == 0:
                yield frame
    finally:
        cap.release()


def image_source(path: str | Path) -> FrameSource:
    """Yield a single image frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cv2 could not read image: {path}")

    logger.info("Image loaded: %s", path.name)
    yield frame


def folder_source(
    folder: str | Path,
    skip_frames: int = 1,
) -> FrameSource:
    """Yield frames from all images in a folder (sorted)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    images: List[Path] = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    logger.info("Folder source: %d images from %s", len(images), folder)
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Skipping unreadable file: %s", img_path.name)
            continue
        yield frame


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_source(
    source_type: str,
    path: str | Path | None = None,
    webcam_index: int = 0,
    skip_frames: int = 1,
) -> FrameSource:
    """
    Build the appropriate frame source.

    Parameters
    ----------
    source_type : "webcam" | "video" | "image" | "folder"
    path        : file/folder path (not needed for webcam)
    webcam_index: camera index
    skip_frames : process every Nth frame
    """
    st = source_type.lower().strip()
    if st == "webcam":
        return webcam_source(index=webcam_index, skip_frames=skip_frames)
    elif st == "video":
        if path is None:
            raise ValueError("--path is required for source=video")
        return video_source(path=path, skip_frames=skip_frames)
    elif st == "image":
        if path is None:
            raise ValueError("--path is required for source=image")
        return image_source(path=path)
    elif st == "folder":
        if path is None:
            raise ValueError("--path is required for source=folder")
        return folder_source(folder=path, skip_frames=skip_frames)
    else:
        raise ValueError(f"Unknown source type: {source_type!r}. Choose webcam/video/image/folder")
