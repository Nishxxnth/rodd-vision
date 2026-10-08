"""
app.py – Streamlit UI for RODD-Vision
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rodd.config import load_config, RODDConfig, VALID_PROFILES
from rodd.detectors import build_detector, BaseDetector, _DETECTOR_MAP
from rodd.pipeline import RODDPipeline
from rodd.speech import SpeechEngine
from rodd.spatial import SpatialAnnotation
from rodd.utils import setup_logging

setup_logging(logging.WARNING)  # suppress verbose logs in UI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RODD-Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #1a1d27;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 4px 0;
        border-left: 4px solid #ff6b35;
    }
    .announce-item {
        background: #1e2330;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px 0;
        font-family: monospace;
        font-size: 0.92em;
        color: #a8e6cf;
    }
    .stDataFrame { font-size: 0.85em; }
    .profile-badge {
        display: inline-block;
        background: #2d3561;
        color: #a8e6cf;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.85em;
        font-weight: 600;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/traffic-light.png", width=60)
    st.title("🚦 RODD-Vision")
    st.caption("Road Obstacle Detection & Description")
    st.divider()

    source_type = st.selectbox("Input Source", ["Upload Image", "Upload Video", "Webcam"],
                               key="source_sel")

    # Profile selector
    profile = st.selectbox(
        "Detection Profile",
        list(VALID_PROFILES),
        index=0,
        key="profile_sel",
        help="outdoor: road classes | indoor: room classes | all: combined",
    )

    detector_type = st.selectbox(
        "Detector",
        list(_DETECTOR_MAP.keys()),
        index=list(_DETECTOR_MAP.keys()).index("hybrid") if "hybrid" in _DETECTOR_MAP else 0,
        key="det_sel",
        help="hybrid: best accuracy | yoloworld_large: fast single model | coco: fastest",
    )
    conf = st.slider("Confidence Threshold (global)", 0.1, 0.9, 0.30, 0.05, key="conf_sl")
    mute = st.toggle("🔇 Mute Speech", value=False, key="mute_tog")
    cooldown = st.slider("Speech Cooldown (s)", 1.0, 10.0, 3.0, 0.5, key="cool_sl")

    st.divider()
    # Show active classes for selected profile
    _preview_cfg = load_config(profile=profile)
    st.markdown(f"**Profile: `{profile}`** — {len(_preview_cfg.classes)} classes")
    st.caption("  •  ".join(sorted(_preview_cfg.classes)))


# ---------------------------------------------------------------------------
# Cached detector (keyed by detector type + profile + conf)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading detector model …")
def get_detector(det_type: str, profile_name: str, conf_thresh: float) -> BaseDetector:
    cfg = load_config(profile=profile_name)
    cfg.model.type = det_type
    cfg.model.confidence = conf_thresh
    return build_detector(cfg, detector_type=det_type)


# ---------------------------------------------------------------------------
# Build pipeline (not cached – depends on mutable speech settings)
# ---------------------------------------------------------------------------

def make_pipeline(
    detector: BaseDetector,
    cfg: RODDConfig,
    mute_flag: bool,
    cooldown_s: float,
) -> RODDPipeline:
    speech = SpeechEngine(
        enabled=not mute_flag,
        cooldown=cooldown_s,
        dedup_window=cooldown_s * 2,
        rate=cfg.speech.rate,
        volume=cfg.speech.volume,
    )
    return RODDPipeline(cfg, detector=detector, speech=speech, mute=mute_flag)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🚦 RODD-Vision – Road Obstacle Detection")
st.caption("Assistive pipeline | profile: **" + profile + "** | detector: **" + detector_type + "**")

cfg = load_config(profile=profile)
cfg.model.confidence = conf
cfg.speech.cooldown_seconds = cooldown

detector = get_detector(detector_type, profile, conf)

announce_log: list[str] = st.session_state.get("announce_log", [])
st.session_state["announce_log"] = announce_log

col_frame, col_info = st.columns([2, 1])

# ---------------------------------------------------------------------------
# IMAGE mode
# ---------------------------------------------------------------------------

if source_type == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 key="img_uploader")
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            pipeline = make_pipeline(detector, cfg, mute, cooldown)
            annotated, annotations, inf_ms = pipeline.process(frame)
            pipeline.shutdown()

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col_frame:
                st.image(annotated_rgb, caption="Annotated Frame", use_container_width=True)

            with col_info:
                st.metric("Detections", len(annotations))
                st.metric("Inference", f"{inf_ms:.0f} ms")
                st.markdown(
                    f'<div class="profile-badge">Profile: {profile}</div>',
                    unsafe_allow_html=True,
                )
                st.divider()

                if annotations:
                    st.markdown("**Detections**")
                    table_data = [
                        {
                            "Label": a.label,
                            "Conf": f"{a.confidence:.0%}",
                            "Direction": a.direction,
                            "Distance": a.distance,
                        }
                        for a in annotations
                    ]
                    st.dataframe(table_data, use_container_width=True, hide_index=True)

                    top = annotations[0]
                    announce_text = top.announce_text()
                    if announce_text not in announce_log:
                        announce_log.insert(0, f"[{time.strftime('%H:%M:%S')}] {announce_text}")
                        announce_log = announce_log[:20]
                        st.session_state["announce_log"] = announce_log
                    st.markdown("**Voice Announcement**")
                    st.success(f"🔊 {announce_text}")
                else:
                    st.info("No obstacles detected")

        st.divider()
        if announce_log:
            st.markdown("**📋 Announcement Log**")
            for entry in announce_log[:10]:
                st.markdown(f'<div class="announce-item">{entry}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# VIDEO mode
# ---------------------------------------------------------------------------

elif source_type == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"],
                                       key="vid_uploader")
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.info(f"Processing video: {uploaded_video.name}")
        pipeline = make_pipeline(detector, cfg, mute, cooldown)

        cap = cv2.VideoCapture(tmp_path)
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_n = 0

        stop_btn = st.button("⏹ Stop", key="vid_stop")

        while cap.isOpened() and not stop_btn:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1
            if frame_n % max(cfg.source.skip_frames, 1) != 0:
                continue

            annotated, annotations, inf_ms = pipeline.process(frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, use_container_width=True)

            if total_frames > 0:
                progress.progress(min(frame_n / total_frames, 1.0))

            if annotations:
                top = annotations[0]
                announce_text = top.announce_text()
                info_placeholder.success(f"🔊 {announce_text}")
                if announce_text not in [e.split("] ", 1)[-1] for e in announce_log[:1]]:
                    announce_log.insert(0, f"[{time.strftime('%H:%M:%S')}] {announce_text}")
                    st.session_state["announce_log"] = announce_log[:20]

        cap.release()
        pipeline.shutdown()
        progress.progress(1.0)
        st.success("Video processing complete")

# ---------------------------------------------------------------------------
# WEBCAM mode
# ---------------------------------------------------------------------------

elif source_type == "Webcam":
    st.info("Webcam mode: The Start button will open your camera. Press Stop to release it.")
    col_btn1, col_btn2 = st.columns(2)
    start = col_btn1.button("▶ Start Webcam", key="cam_start")
    stop  = col_btn2.button("⏹ Stop Webcam",  key="cam_stop")

    frame_placeholder = st.empty()
    ann_placeholder = st.empty()

    if start:
        st.session_state["cam_running"] = True
    if stop:
        st.session_state["cam_running"] = False

    if st.session_state.get("cam_running", False):
        pipeline = make_pipeline(detector, cfg, mute, cooldown)
        cap = cv2.VideoCapture(cfg.source.webcam_index)

        if not cap.isOpened():
            st.error("Cannot open webcam. Check that it is connected and not in use.")
            st.session_state["cam_running"] = False
        else:
            frame_n = 0
            while st.session_state.get("cam_running", False):
                ok, frame = cap.read()
                if not ok:
                    st.warning("Webcam stream ended")
                    break
                frame_n += 1
                if frame_n % max(cfg.source.skip_frames, 1) != 0:
                    continue

                annotated, annotations, inf_ms = pipeline.process(frame)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, use_container_width=True)

                if annotations:
                    top = annotations[0]
                    ann_placeholder.success(f"🔊 {top.announce_text()}")
                    announce_log.insert(0, f"[{time.strftime('%H:%M:%S')}] {top.announce_text()}")
                    st.session_state["announce_log"] = announce_log[:20]
                else:
                    ann_placeholder.empty()

            cap.release()
            pipeline.shutdown()
            st.info("Webcam stopped.")
