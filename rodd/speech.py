"""
rodd/speech.py
Non-blocking TTS using pyttsx3 in a dedicated worker thread.
Features: cooldown, dedup, mute mode, graceful degradation.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TTSWorker(threading.Thread):
    """
    Background thread that owns the pyttsx3 engine and drains a queue.
    The engine must be initialised inside this thread (pyttsx3 constraint).
    """

    def __init__(self, rate: int, volume: float):
        super().__init__(name="rodd-tts", daemon=True)
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._engine = None
        self._ready = threading.Event()
        self._degraded = False  # True if engine init failed

    def run(self) -> None:
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            logger.info("pyttsx3 TTS engine initialised (rate=%d)", self._rate)
        except Exception as exc:
            logger.warning("TTS engine init failed: %s – degrading to print mode", exc)
            self._degraded = True
        finally:
            self._ready.set()

        while True:
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            if msg is None:          # sentinel → shutdown
                break

            if self._degraded:
                print(f"[RODD VOICE] {msg}")
            else:
                try:
                    self._engine.say(msg)
                    self._engine.runAndWait()
                except Exception as exc:
                    logger.warning("TTS speak error: %s", exc)
                    print(f"[RODD VOICE] {msg}")

    def speak(self, text: str) -> None:
        self._q.put(text)

    def stop(self) -> None:
        self._q.put(None)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class SpeechEngine:
    """
    Thread-safe speech manager.

    Parameters
    ----------
    enabled : whether to actually speak
    cooldown : seconds between any announcement
    dedup_window : seconds before the same label+direction can repeat
    rate / volume : pyttsx3 settings
    clock_fn : injectable time function (for testing)
    """

    def __init__(
        self,
        enabled: bool = True,
        cooldown: float = 3.0,
        dedup_window: float = 6.0,
        rate: int = 150,
        volume: float = 1.0,
        clock_fn=None,
    ):
        self._enabled = enabled
        self._cooldown = cooldown
        self._dedup_window = dedup_window
        self._clock = clock_fn or time.monotonic

        self._last_speak_time: float = float("-inf")
        self._dedup_cache: dict[str, float] = {}   # key → last spoken time
        self._lock = threading.Lock()

        self._worker: Optional[_TTSWorker] = None
        if enabled:
            self._worker = _TTSWorker(rate=rate, volume=volume)
            self._worker.start()
            self._worker._ready.wait(timeout=5)

    # ------------------------------------------------------------------
    def announce(self, text: str) -> bool:
        """
        Announce *text* if cooldown and dedup allow it.
        Returns True if the announcement was dispatched.
        """
        now = self._clock()
        with self._lock:
            # Global cooldown
            if now - self._last_speak_time < self._cooldown:
                return False
            # Dedup
            last_time = self._dedup_cache.get(text, float("-inf"))
            if now - last_time < self._dedup_window:
                return False

            self._last_speak_time = now
            self._dedup_cache[text] = now

        if self._enabled and self._worker:
            self._worker.speak(text)
        else:
            print(f"[RODD VOICE] {text}")

        logger.info("Announced: %s", text)
        return True

    def mute(self) -> None:
        self._enabled = False

    def unmute(self) -> None:
        self._enabled = True

    def shutdown(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "SpeechEngine":
        return cls(
            enabled=cfg.speech.enabled,
            cooldown=cfg.speech.cooldown_seconds,
            dedup_window=cfg.speech.dedup_window_seconds,
            rate=cfg.speech.rate,
            volume=cfg.speech.volume,
        )
