"""
tests/test_state_machine.py

Unit tests for the GestureStateMachine:
  - temporal confirmation
  - debouncing (one event per gesture activation)
  - cooldown
  - state transitions
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gestures.features import HandFeatureSet
from gestures.types import GestureType, StateMachineState, ControlMode
from gestures.state_machine import GestureStateMachine


# ─── Helper ───────────────────────────────────────────────────────────────────

def make_open_palm_fs():
    """HandFeatureSet that reliably classifies as OPEN_PALM."""
    return HandFeatureSet(
        valid=True,
        thumb_extended=True,
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
        thumb_angle=60.0,
        index_angle=155.0,
        middle_angle=155.0,
        ring_angle=155.0,
        pinky_angle=155.0,
        thumb_index_distance=0.6,
        palm_x=0.5,
        palm_y=0.5,
    )


def make_fist_fs():
    return HandFeatureSet(
        valid=True,
        thumb_extended=False,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
        thumb_angle=20.0,
        index_angle=20.0,
        middle_angle=20.0,
        ring_angle=20.0,
        pinky_angle=20.0,
        thumb_index_distance=0.6,
        palm_x=0.5,
        palm_y=0.5,
    )


# ─── Temporal confirmation ────────────────────────────────────────────────────

class TestTemporalConfirmation:

    def test_single_frame_not_confirmed(self):
        """One matching frame should NOT produce an ACTIVE gesture."""
        sm = GestureStateMachine(confirm_frames=5, cooldown_ms=0)
        fs = make_open_palm_fs()
        result, event = sm.update(fs)
        assert not result.active
        assert event is None

    def test_confirmed_after_n_frames(self):
        """Exactly confirm_frames matching frames should activate the gesture."""
        N = 5
        sm = GestureStateMachine(confirm_frames=N, cooldown_ms=0)
        fs = make_open_palm_fs()
        result, event = None, None
        for _ in range(N):
            result, event = sm.update(fs)
        assert result.active
        assert result.gesture == GestureType.OPEN_PALM
        assert event is not None
        assert event.gesture == GestureType.OPEN_PALM

    def test_interrupted_candidate_resets(self):
        """Alternating frames should NOT confirm a gesture."""
        sm = GestureStateMachine(confirm_frames=5, cooldown_ms=0)
        for i in range(10):
            fs = make_open_palm_fs() if i % 2 == 0 else make_fist_fs()
            result, event = sm.update(fs)
        # Should never have become ACTIVE for OPEN_PALM
        # (could be FIST if last frames agreed, but not guaranteed)
        # The important thing is we didn't get a rogue ACTIVE
        # (test: no event fired for open palm with alternating pattern)
        # Reset and verify clean state
        sm2 = GestureStateMachine(confirm_frames=5, cooldown_ms=0)
        events = []
        for i in range(10):
            fs = make_open_palm_fs() if i % 2 == 0 else make_fist_fs()
            _, ev = sm2.update(fs)
            if ev is not None:
                events.append(ev.gesture)
        # No OPEN_PALM event should have fired given the alternation
        assert GestureType.OPEN_PALM not in events


# ─── Debouncing ───────────────────────────────────────────────────────────────

class TestDebouncing:

    def test_gesture_event_fires_once(self):
        """Holding a gesture for many frames should produce exactly ONE event."""
        sm = GestureStateMachine(confirm_frames=5, cooldown_ms=0)
        fs = make_open_palm_fs()
        events = []
        for _ in range(30):
            _, event = sm.update(fs)
            if event is not None:
                events.append(event)
        assert len(events) == 1
        assert events[0].gesture == GestureType.OPEN_PALM


# ─── State transitions ────────────────────────────────────────────────────────

class TestStateTransitions:

    def test_idle_to_active_to_cooldown(self):
        """Full state transition: NO_GESTURE → CANDIDATE → ACTIVE → COOLDOWN."""
        sm = GestureStateMachine(confirm_frames=3, cooldown_ms=100)
        fs = make_open_palm_fs()

        # Feed 3 frames → should reach ACTIVE
        result = None
        for _ in range(3):
            result, _ = sm.update(fs)
        assert result.active
        assert result.state == StateMachineState.ACTIVE

        # Now switch to a fist → should enter COOLDOWN
        fist = make_fist_fs()
        result, _ = sm.update(fist)
        assert result.state == StateMachineState.COOLDOWN or not result.active

    def test_no_gesture_when_hand_absent(self):
        """No hand → no gesture."""
        sm = GestureStateMachine(confirm_frames=5, cooldown_ms=0)
        fs = HandFeatureSet(valid=False)
        for _ in range(10):
            result, event = sm.update(fs)
        assert result.gesture == GestureType.NONE
        assert not result.active
        assert event is None

    def test_open_palm_then_fist(self):
        """OPEN_PALM → (release) → FIST should each fire one event."""
        sm = GestureStateMachine(confirm_frames=3, cooldown_ms=0)

        open_events = []
        fist_events = []

        # Confirm OPEN_PALM
        for _ in range(5):
            _, ev = sm.update(make_open_palm_fs())
            if ev and ev.gesture == GestureType.OPEN_PALM:
                open_events.append(ev)

        # Switch to FIST
        for _ in range(5):
            _, ev = sm.update(make_fist_fs())
            if ev and ev.gesture == GestureType.FIST:
                fist_events.append(ev)

        assert len(open_events) == 1
        assert len(fist_events) == 1


# ─── Missing hand safety ──────────────────────────────────────────────────────

class TestSafety:

    def test_no_crash_on_none_landmarks(self):
        """State machine must never crash on invalid features."""
        sm = GestureStateMachine()
        for _ in range(50):
            result, event = sm.update(HandFeatureSet(valid=False))
            assert result is not None

    def test_confidence_always_in_range(self):
        sm = GestureStateMachine(confirm_frames=3, cooldown_ms=0)
        for fs in [make_open_palm_fs(), make_fist_fs(), HandFeatureSet(valid=False)]:
            for _ in range(5):
                result, _ = sm.update(fs)
                assert 0.0 <= result.confidence <= 1.0
