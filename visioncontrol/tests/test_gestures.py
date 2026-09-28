"""
tests/test_gestures.py

Unit tests for Phase 4 static gesture recognition,
pinch hysteresis, swipe detection and gesture mapper.
"""

import pytest
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gestures.features import HandFeatureSet
from gestures.types import GestureType, GestureAction, ControlMode
from gestures.state_machine import StaticRecognizer, SwipeDetector
from gestures.gesture_mapper import GestureMapper


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_fs(
    valid=True,
    thumb=False, index=False, middle=False, ring=False, pinky=False,
    thumb_angle=60.0, index_angle=60.0, middle_angle=60.0,
    ring_angle=60.0, pinky_angle=60.0,
    thumb_index_distance=0.5,
    palm_x=0.5, palm_y=0.5,
):
    """Build a HandFeatureSet with sane defaults for testing."""
    fs = HandFeatureSet(
        valid=valid,
        thumb_extended=thumb,
        index_extended=index,
        middle_extended=middle,
        ring_extended=ring,
        pinky_extended=pinky,
        thumb_angle=thumb_angle,
        index_angle=index_angle,
        middle_angle=middle_angle,
        ring_angle=ring_angle,
        pinky_angle=pinky_angle,
        thumb_index_distance=thumb_index_distance,
        palm_x=palm_x,
        palm_y=palm_y,
    )
    return fs


# ─── StaticRecognizer ─────────────────────────────────────────────────────────

class TestStaticRecognizer:
    def setup_method(self):
        self.rec = StaticRecognizer()

    def test_open_palm(self):
        fs = make_fs(thumb=True, index=True, middle=True, ring=True, pinky=True,
                     index_angle=150.0, middle_angle=150.0,
                     ring_angle=150.0, pinky_angle=150.0)
        g, c = self.rec.classify(fs)
        assert g == GestureType.OPEN_PALM
        assert c > 0.0

    def test_fist(self):
        fs = make_fs(thumb=False, index=False, middle=False, ring=False, pinky=False,
                     index_angle=20.0, middle_angle=20.0,
                     ring_angle=20.0, pinky_angle=20.0)
        g, c = self.rec.classify(fs)
        assert g == GestureType.FIST
        assert c > 0.0

    def test_two_fingers(self):
        fs = make_fs(index=True, middle=True, ring=False, pinky=False,
                     index_angle=140.0, middle_angle=140.0,
                     ring_angle=30.0, pinky_angle=30.0)
        g, c = self.rec.classify(fs)
        assert g == GestureType.TWO_FINGERS
        assert c > 0.0

    def test_pinch_below_on_threshold(self):
        fs = make_fs(thumb_index_distance=0.15)
        g, c = self.rec.classify(fs)
        assert g == GestureType.PINCH
        assert c > 0.0  # confidence is geometric (1 - dist/threshold); valid as long as > 0

    def test_pinch_hysteresis_stay_active(self):
        """Distance just above ON but still below OFF should keep pinch ACTIVE."""
        rec = StaticRecognizer(pinch_on_threshold=0.22, pinch_off_threshold=0.28)
        # Enter pinch
        g, _ = rec.classify(make_fs(thumb_index_distance=0.15))
        assert g == GestureType.PINCH
        # Distance moves into hysteresis zone (between 0.22 and 0.28) → stays pinch
        g, _ = rec.classify(make_fs(thumb_index_distance=0.25))
        assert g == GestureType.PINCH

    def test_pinch_hysteresis_deactivate(self):
        """Distance above OFF threshold should deactivate pinch."""
        rec = StaticRecognizer(pinch_on_threshold=0.22, pinch_off_threshold=0.28)
        rec.classify(make_fs(thumb_index_distance=0.15))  # activate
        g, _ = rec.classify(make_fs(thumb_index_distance=0.35))  # above OFF
        assert g != GestureType.PINCH

    def test_none_when_invalid(self):
        fs = HandFeatureSet(valid=False)
        g, c = self.rec.classify(fs)
        assert g == GestureType.NONE
        assert c == 0.0

    def test_confidence_in_range(self):
        """All confidence values must be in [0, 1]."""
        cases = [
            make_fs(thumb=True, index=True, middle=True, ring=True, pinky=True,
                    index_angle=150.0, middle_angle=150.0, ring_angle=150.0, pinky_angle=150.0),
            make_fs(thumb_index_distance=0.10),
        ]
        for fs in cases:
            _, c = self.rec.classify(fs)
            assert 0.0 <= c <= 1.0


# ─── SwipeDetector ────────────────────────────────────────────────────────────

class TestSwipeDetector:

    def _make_detector(self):
        return SwipeDetector(
            history_size=10,
            min_distance=0.15,
            min_speed=0.3,
            max_vertical_ratio=0.6,
            max_duration=1.0,
            horizontal_dominance=1.4,
            cooldown_s=0.0,   # no cooldown for test speed
        )

    def _push_swipe(self, detector, xs, base_y=0.5, dt=0.05):
        """Push a sequence of x positions and return the last gesture result."""
        t = time.time()
        result = None
        for x in xs:
            fs = make_fs(palm_x=x, palm_y=base_y)
            r = detector.update(fs)
            if r is not None:
                result = r
            t += dt
        return result

    def test_swipe_right(self):
        d = self._make_detector()
        result = self._push_swipe(d, [0.20, 0.25, 0.32, 0.40, 0.48])
        assert result is not None
        assert result[0] == GestureType.SWIPE_RIGHT

    def test_swipe_left(self):
        d = self._make_detector()
        result = self._push_swipe(d, [0.50, 0.43, 0.35, 0.28, 0.20])
        assert result is not None
        assert result[0] == GestureType.SWIPE_LEFT

    def test_vertical_movement_no_swipe(self):
        """Pure vertical movement should NOT trigger a swipe."""
        d = self._make_detector()
        xs = [0.50] * 8   # x stays constant
        ys = [0.2, 0.25, 0.3, 0.38, 0.46, 0.54, 0.62, 0.70]  # y moves a lot
        result = None
        for x, y in zip(xs, ys):
            fs = make_fs(palm_x=x, palm_y=y)
            r = d.update(fs)
            if r:
                result = r
        assert result is None

    def test_missing_hand_clears_history(self):
        d = self._make_detector()
        d.update(make_fs(palm_x=0.2))
        d.update(HandFeatureSet(valid=False))   # hand disappears
        result = d.update(make_fs(palm_x=0.5))  # small move after reset
        # Should NOT fire a swipe since history was cleared
        assert result is None


# ─── GestureMapper ────────────────────────────────────────────────────────────

class TestGestureMapper:
    def setup_method(self):
        self.mapper = GestureMapper()

    def test_open_palm_enters_control(self):
        from gestures.types import GestureEvent
        ev = GestureEvent(gesture=GestureType.OPEN_PALM)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.ENTER_CONTROL
        assert self.mapper.mode == ControlMode.CONTROL

    def test_fist_emergency_stop(self):
        from gestures.types import GestureEvent
        # First enter control
        self.mapper.map_event(GestureEvent(gesture=GestureType.OPEN_PALM))
        ev = GestureEvent(gesture=GestureType.FIST)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.EMERGENCY_STOP
        assert self.mapper.mode == ControlMode.IDLE

    def test_swipe_right_next(self):
        from gestures.types import GestureEvent
        ev = GestureEvent(gesture=GestureType.SWIPE_RIGHT)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.NEXT

    def test_swipe_left_previous(self):
        from gestures.types import GestureEvent
        ev = GestureEvent(gesture=GestureType.SWIPE_LEFT)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.PREVIOUS

    def test_two_fingers_select(self):
        from gestures.types import GestureEvent
        ev = GestureEvent(gesture=GestureType.TWO_FINGERS)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.SELECT

    def test_pinch_confirm(self):
        from gestures.types import GestureEvent
        ev = GestureEvent(gesture=GestureType.PINCH)
        action = self.mapper.map_event(ev)
        assert action == GestureAction.CONFIRM
