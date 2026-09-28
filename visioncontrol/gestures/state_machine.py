"""
gestures/state_machine.py

Phase 4: Gesture Recognition State Machine.

Consumes HandFeatureSet (Phase 3) and produces GestureResult + GestureEvent.

Architecture:
    HandFeatureSet → StaticRecognizer → CandidateTracker → GestureResult
                   → SwipeDetector                       → GestureEvent

This module does NOT:
    - control devices
    - call OpenAI
    - send Arduino commands
    - contain UI logic
"""

import time
import math
from collections import deque
from typing import Optional, Tuple, List

from gestures.features import HandFeatureSet
from gestures.types import (
    GestureType, StateMachineState, GestureResult, GestureEvent
)


# ─── Static Gesture Recognizer ────────────────────────────────────────────────

class StaticRecognizer:
    """
    Classifies a single HandFeatureSet into a candidate GestureType + confidence.
    Does NOT do temporal debouncing — that belongs to the state machine.
    """

    def __init__(
        self,
        pinch_on_threshold: float  = 0.22,
        pinch_off_threshold: float = 0.28,
    ):
        self.pinch_on_threshold  = pinch_on_threshold
        self.pinch_off_threshold = pinch_off_threshold
        self._pinch_active = False  # hysteresis state

    def classify(self, fs: HandFeatureSet) -> Tuple[GestureType, float]:
        """
        Returns (GestureType, confidence ∈ [0,1]).
        Swipes are NOT classified here — they require movement history.
        """
        if not fs.valid:
            return GestureType.NONE, 0.0

        # ── PINCH (hysteresis) ────────────────────────────────────────────────
        tid = fs.thumb_index_distance
        if self._pinch_active:
            if tid > self.pinch_off_threshold:
                self._pinch_active = False
        else:
            if tid < self.pinch_on_threshold:
                self._pinch_active = True

        if self._pinch_active:
            # Confidence inversely proportional to distance (closer = more confident)
            confidence = max(0.0, 1.0 - tid / self.pinch_on_threshold)
            return GestureType.PINCH, round(min(confidence, 1.0), 3)

        ext = (
            fs.thumb_extended,
            fs.index_extended,
            fs.middle_extended,
            fs.ring_extended,
            fs.pinky_extended,
        )
        n_extended = sum(ext)

        # ── OPEN_PALM ─────────────────────────────────────────────────────────
        if all(ext):
            # Extra validation: angles should all be relatively open
            avg_angle = (fs.index_angle + fs.middle_angle + fs.ring_angle + fs.pinky_angle) / 4.0
            angle_conf = min(avg_angle / 140.0, 1.0)  # 140° is a well-open finger
            finger_conf = n_extended / 5.0
            confidence = (angle_conf * 0.5 + finger_conf * 0.5)
            return GestureType.OPEN_PALM, round(confidence, 3)

        # ── FIST ──────────────────────────────────────────────────────────────
        if n_extended == 0:
            # Confidence: angles should all be low (curled)
            avg_angle = (fs.index_angle + fs.middle_angle + fs.ring_angle + fs.pinky_angle) / 4.0
            angle_conf = max(0.0, 1.0 - avg_angle / 90.0)
            confidence = (angle_conf * 0.6 + (1.0 - n_extended / 5.0) * 0.4)
            return GestureType.FIST, round(min(confidence, 1.0), 3)

        # ── TWO_FINGERS ───────────────────────────────────────────────────────
        # index + middle extended, ring + pinky folded; thumb state is flexible
        if fs.index_extended and fs.middle_extended and not fs.ring_extended and not fs.pinky_extended:
            # Confidence based on how well the two fingers stand out
            pair_angles = (fs.index_angle + fs.middle_angle) / 2.0
            folded_angles = (fs.ring_angle + fs.pinky_angle) / 2.0
            angle_contrast = min((pair_angles - folded_angles) / 90.0, 1.0)
            confidence = max(0.5, angle_contrast)
            return GestureType.TWO_FINGERS, round(min(confidence, 1.0), 3)

        return GestureType.NONE, 0.0


# ─── Swipe Detector ───────────────────────────────────────────────────────────

class SwipeDetector:
    """
    Detects swipe gestures using a rolling history of palm positions.
    Swipes require: distance, velocity, horizontal dominance, duration.
    """

    def __init__(
        self,
        history_size: int        = 15,
        min_distance: float      = 0.18,
        min_speed: float         = 0.6,
        max_vertical_ratio: float = 0.6,
        max_duration: float      = 0.8,
        horizontal_dominance: float = 1.4,
        cooldown_s: float        = 0.5,
    ):
        self.min_distance         = min_distance
        self.min_speed            = min_speed
        self.max_vertical_ratio   = max_vertical_ratio
        self.max_duration         = max_duration
        self.horizontal_dominance = horizontal_dominance
        self.cooldown_s           = cooldown_s

        # Each entry: (x, y, timestamp)
        self._history: deque = deque(maxlen=history_size)
        self._last_swipe_time: float = 0.0

    def update(self, fs: HandFeatureSet) -> Optional[Tuple[GestureType, float]]:
        """
        Push the latest frame's palm position and check for a swipe.
        Returns (GestureType, confidence) or None.
        """
        if not fs.valid:
            self._history.clear()
            return None

        now = time.time()
        self._history.append((fs.palm_x, fs.palm_y, now))

        if len(self._history) < 5:
            return None

        # Cooldown guard
        if now - self._last_swipe_time < self.cooldown_s:
            return None

        return self._evaluate(now)

    def _evaluate(self, now: float) -> Optional[Tuple[GestureType, float]]:
        oldest_x, oldest_y, t_start = self._history[0]
        newest_x, newest_y, t_end   = self._history[-1]

        total_dx   = newest_x - oldest_x
        total_dy   = newest_y - oldest_y
        duration   = t_end - t_start
        distance   = abs(total_dx)
        vert_dist  = abs(total_dy)

        if duration < 1e-6:
            return None

        speed = distance / duration

        # ── Reject diagonal swipes ────────────────────────────────────────────
        if vert_dist > 0 and (distance / vert_dist) < self.horizontal_dominance:
            return None

        # ── Reject slow, short, or too-long swipes ────────────────────────────
        if distance < self.min_distance:
            return None
        if speed < self.min_speed:
            return None
        if duration > self.max_duration:
            return None
        if vert_dist > 0 and (vert_dist / max(distance, 1e-6)) > self.max_vertical_ratio:
            return None

        # ── Determine direction ───────────────────────────────────────────────
        if total_dx > 0:
            gesture = GestureType.SWIPE_RIGHT
        else:
            gesture = GestureType.SWIPE_LEFT

        # Confidence: how clean the swipe is (speed + horizontal dominance)
        speed_conf  = min(speed / (self.min_speed * 2), 1.0)
        horiz_conf  = 1.0 - min(vert_dist / max(distance, 1e-6), 1.0)
        confidence  = round((speed_conf * 0.5 + horiz_conf * 0.5), 3)

        self._last_swipe_time = time.time()
        self._history.clear()

        return gesture, confidence


# ─── Gesture State Machine ────────────────────────────────────────────────────

class GestureStateMachine:
    """
    Main Phase 4 gesture engine.

    Consumes HandFeatureSet every frame and produces:
        - result:  GestureResult  (current state, updated every frame)
        - event:   GestureEvent   (fired once on transition to ACTIVE)

    States: NO_GESTURE → CANDIDATE → ACTIVE → COOLDOWN → NO_GESTURE
    """

    def __init__(
        self,
        confirm_frames: int        = 5,
        cooldown_ms: float         = 500.0,
        pinch_on_threshold: float  = 0.22,
        pinch_off_threshold: float = 0.28,
        swipe_history_size: int    = 15,
        swipe_min_distance: float  = 0.18,
        swipe_min_speed: float     = 0.6,
        swipe_max_vertical_ratio: float = 0.6,
        swipe_max_duration: float  = 0.8,
        swipe_horizontal_dominance: float = 1.4,
    ):
        self.confirm_frames = confirm_frames
        self.cooldown_s     = cooldown_ms / 1000.0

        self._recognizer = StaticRecognizer(
            pinch_on_threshold=pinch_on_threshold,
            pinch_off_threshold=pinch_off_threshold,
        )
        self._swipe_detector = SwipeDetector(
            history_size=swipe_history_size,
            min_distance=swipe_min_distance,
            min_speed=swipe_min_speed,
            max_vertical_ratio=swipe_max_vertical_ratio,
            max_duration=swipe_max_duration,
            horizontal_dominance=swipe_horizontal_dominance,
            cooldown_s=self.cooldown_s,
        )

        # State
        self._state            = StateMachineState.NO_GESTURE
        self._candidate        = GestureType.NONE
        self._candidate_frames = 0
        self._candidate_conf   = 0.0
        self._active_gesture   = GestureType.NONE
        self._cooldown_end     = 0.0

        # Published outputs
        self.result: GestureResult = GestureResult()
        self.event:  Optional[GestureEvent] = None  # None = no new event this frame

    def update(self, fs: HandFeatureSet) -> Tuple[GestureResult, Optional[GestureEvent]]:
        """
        Process one frame of hand features.
        Returns (GestureResult, GestureEvent|None).
        GestureEvent is non-None only on the frame a gesture becomes ACTIVE.
        """
        self.event = None
        now = time.time()

        # ── Check cooldown ────────────────────────────────────────────────────
        if self._state == StateMachineState.COOLDOWN:
            if now >= self._cooldown_end:
                self._state = StateMachineState.NO_GESTURE
            else:
                self.result = GestureResult(
                    gesture=GestureType.NONE,
                    confidence=0.0,
                    active=False,
                    state=self._state,
                )
                return self.result, self.event

        # ── Swipe detection (event-style, bypasses normal state machine) ──────
        swipe = self._swipe_detector.update(fs)
        if swipe is not None:
            s_type, s_conf = swipe
            self.event = GestureEvent(
                gesture=s_type,
                confidence=s_conf,
            )
            self.result = GestureResult(
                gesture=s_type,
                confidence=s_conf,
                active=True,
                state=StateMachineState.ACTIVE,
            )
            # Short cooldown after swipe
            self._state       = StateMachineState.COOLDOWN
            self._cooldown_end = now + self.cooldown_s
            self._candidate   = GestureType.NONE
            self._candidate_frames = 0
            return self.result, self.event

        # ── Static gesture classification ─────────────────────────────────────
        candidate, confidence = self._recognizer.classify(fs)

        if not fs.valid:
            self._reset_candidate()
            self.result = GestureResult(state=StateMachineState.NO_GESTURE)
            return self.result, self.event

        # ── Temporal confirmation ──────────────────────────────────────────────
        if candidate == self._candidate and candidate != GestureType.NONE:
            self._candidate_frames += 1
            self._candidate_conf = confidence
        else:
            # Candidate changed — restart counter
            self._candidate        = candidate
            self._candidate_frames = 1
            self._candidate_conf   = confidence
            if self._state == StateMachineState.ACTIVE:
                # Gesture was released or changed
                self._enter_cooldown(now)
                self.result = GestureResult(
                    gesture=GestureType.NONE,
                    confidence=0.0,
                    active=False,
                    state=StateMachineState.COOLDOWN,
                )
                return self.result, self.event

        # ── Transition: CANDIDATE → ACTIVE ────────────────────────────────────
        if (self._state in (StateMachineState.NO_GESTURE, StateMachineState.CANDIDATE)
                and self._candidate_frames >= self.confirm_frames
                and candidate != GestureType.NONE):
            if self._state != StateMachineState.ACTIVE or self._active_gesture != candidate:
                self._state          = StateMachineState.ACTIVE
                self._active_gesture = candidate
                self.event = GestureEvent(
                    gesture=candidate,
                    confidence=confidence,
                )
        elif self._candidate_frames < self.confirm_frames:
            self._state = StateMachineState.CANDIDATE

        # ── Keep ACTIVE while same gesture holds ──────────────────────────────
        if self._state == StateMachineState.ACTIVE and self._active_gesture == candidate:
            self.result = GestureResult(
                gesture=candidate,
                confidence=confidence,
                active=True,
                state=StateMachineState.ACTIVE,
            )
        elif self._state == StateMachineState.CANDIDATE:
            self.result = GestureResult(
                gesture=candidate,
                confidence=confidence,
                active=False,
                state=StateMachineState.CANDIDATE,
            )
        else:
            self.result = GestureResult(state=StateMachineState.NO_GESTURE)

        return self.result, self.event

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_candidate(self):
        self._candidate        = GestureType.NONE
        self._candidate_frames = 0
        self._candidate_conf   = 0.0
        self._state            = StateMachineState.NO_GESTURE
        self._active_gesture   = GestureType.NONE

    def _enter_cooldown(self, now: float):
        self._state            = StateMachineState.COOLDOWN
        self._cooldown_end     = now + self.cooldown_s
        self._active_gesture   = GestureType.NONE
        self._candidate        = GestureType.NONE
        self._candidate_frames = 0
