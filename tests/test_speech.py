"""
tests/test_speech.py
Tests for cooldown and dedup logic using a fake clock (no real audio).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rodd.speech import SpeechEngine


# ---------------------------------------------------------------------------
# Fake clock
# ---------------------------------------------------------------------------

class FakeClock:
    """Mutable monotonic clock for deterministic testing."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(cooldown: float = 3.0, dedup: float = 6.0, clock=None) -> SpeechEngine:
    """Build a muted SpeechEngine with a fake clock for unit testing."""
    clk = clock or FakeClock()
    engine = SpeechEngine(
        enabled=False,   # no audio during tests
        cooldown=cooldown,
        dedup_window=dedup,
        clock_fn=clk,
    )
    return engine, clk


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_first_announcement_always_passes(self):
        engine, _ = make_engine()
        assert engine.announce("car ahead") is True

    def test_immediate_repeat_blocked(self):
        engine, clk = make_engine(cooldown=3.0)
        engine.announce("car ahead")
        # 1 second later – still in cooldown
        clk.advance(1.0)
        assert engine.announce("bus ahead") is False

    def test_after_cooldown_passes(self):
        engine, clk = make_engine(cooldown=3.0)
        engine.announce("car ahead")
        clk.advance(3.1)
        assert engine.announce("bus ahead") is True

    def test_exactly_at_cooldown_boundary_blocked(self):
        engine, clk = make_engine(cooldown=3.0)
        engine.announce("car ahead")
        clk.advance(3.0)  # exactly 3s – passes because check is `< cooldown`, not `<= cooldown`
        assert engine.announce("bus ahead") is True   # at boundary: allowed

    def test_just_past_cooldown_passes(self):
        engine, clk = make_engine(cooldown=3.0)
        engine.announce("car ahead")
        clk.advance(3.01)
        assert engine.announce("bus ahead") is True


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

class TestDedup:
    def test_same_message_blocked_within_dedup_window(self):
        engine, clk = make_engine(cooldown=3.0, dedup=6.0)
        engine.announce("stair ahead, near")
        clk.advance(3.1)   # past cooldown
        # same label+direction again – dedup window 6s not elapsed
        assert engine.announce("stair ahead, near") is False

    def test_same_message_passes_after_dedup_window(self):
        engine, clk = make_engine(cooldown=3.0, dedup=6.0)
        engine.announce("stair ahead, near")
        clk.advance(6.1)   # past both cooldown and dedup
        assert engine.announce("stair ahead, near") is True

    def test_different_message_not_affected_by_dedup(self):
        engine, clk = make_engine(cooldown=3.0, dedup=6.0)
        engine.announce("stair ahead, near")
        clk.advance(3.1)
        # Different message → dedup key is different → should pass
        assert engine.announce("car on your right") is True

    def test_dedup_tracks_per_message(self):
        engine, clk = make_engine(cooldown=3.0, dedup=10.0)  # 10s dedup window
        engine.announce("stair ahead, near")    # t=0
        clk.advance(3.1)
        engine.announce("car ahead")             # t=3.1 → different key → passes
        clk.advance(3.1)                         # t=6.2
        # stair dedup window (10s) not yet elapsed from first announce (t=0)
        assert engine.announce("stair ahead, near") is False


# ---------------------------------------------------------------------------
# Mute
# ---------------------------------------------------------------------------

class TestMute:
    def test_mute_still_returns_true_but_no_audio(self):
        """announce() should track timing even in muted mode."""
        engine, clk = make_engine()
        engine.mute()
        result = engine.announce("pit ahead, near")
        # muted SpeechEngine (enabled=False) with no worker: still returns True
        assert result is True

    def test_double_announce_blocked_when_muted(self):
        engine, clk = make_engine(cooldown=3.0)
        engine.mute()
        engine.announce("pit ahead, near")
        clk.advance(1.0)
        assert engine.announce("car ahead") is False

    def test_worker_not_called_when_muted(self):
        from unittest.mock import MagicMock
        engine, clk = make_engine()
        mock_worker = MagicMock()
        engine._worker = mock_worker
        engine._enabled = True
        engine.mute()
        assert engine.announce("car ahead") is True
        mock_worker.speak.assert_not_called()
