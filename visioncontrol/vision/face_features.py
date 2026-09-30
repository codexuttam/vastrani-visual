"""
vision/face_features.py

Phase 7: Face data models and feature extraction module.

Responsibilities:
    - Strongly typed FaceLandmarks, FaceAnchor, FaceFeatures, and FaceState models
    - Extract face center, size, scale, orientation (yaw, pitch, roll), and movement
    - Exponential smoothing and face-loss grace period handling
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vision.smoothing import ExponentialSmoother, PointSmoother, AngleSmoother

# Landmark indices for MediaPipe Face Mesh (468 landmarks)
NOSE_TIP        = 1
CHIN            = 152
FOREHEAD        = 10
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
LEFT_FACE_EDGE  = 234
RIGHT_FACE_EDGE = 454


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class FaceLandmarks:
    """Normalized face landmarks container."""
    points: list
    detected: bool = True


@dataclass
class FaceAnchor:
    """Primary AR anchor point contract for Phase 8."""
    x: float
    y: float
    scale: float
    rotation: float  # roll angle in degrees
    valid: bool


@dataclass
class FaceFeatures:
    """Extracted raw spatial face metrics."""
    detected: bool
    center_x: float
    center_y: float
    width: float
    height: float
    scale: float


@dataclass
class FaceState:
    """
    Complete output contract for Face Tracking (Phase 7).
    Consumed by Face Feature Extractor and Phase 8 AR Renderer.
    """
    detected: bool

    center_x: float
    center_y: float

    width: float
    height: float
    scale: float

    yaw: float
    pitch: float
    roll: float

    dx: float
    dy: float
    speed: float

    timestamp: float

    @property
    def anchor(self) -> FaceAnchor:
        return FaceAnchor(
            x=self.center_x,
            y=self.center_y,
            scale=self.scale,
            rotation=self.roll,
            valid=self.detected,
        )


# ─── Feature Extractor ────────────────────────────────────────────────────────

class FaceFeatureExtractor:
    """
    Processes raw face landmarks and outputs a smoothed, normalized FaceState.
    Handles smoothing, movement tracking, and face loss timeouts.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.35,
        lost_timeout_ms: float = 500.0,
    ):
        self.alpha: float = smoothing_alpha
        self.lost_timeout_ms: float = lost_timeout_ms

        # Smoothers
        self._center_smoother = PointSmoother(alpha=smoothing_alpha)
        self._width_smoother  = ExponentialSmoother(alpha=smoothing_alpha)
        self._height_smoother = ExponentialSmoother(alpha=smoothing_alpha)
        self._scale_smoother  = ExponentialSmoother(alpha=smoothing_alpha)

        self._yaw_smoother   = AngleSmoother(alpha=smoothing_alpha)
        self._pitch_smoother = AngleSmoother(alpha=smoothing_alpha)
        self._roll_smoother  = AngleSmoother(alpha=smoothing_alpha)

        # Movement tracking state
        self._prev_center: Optional[Tuple[float, float]] = None
        self._prev_time: Optional[float] = None

        # Grace period / face loss tracking
        self._last_valid_time: Optional[float] = None
        self._last_state: Optional[FaceState] = None

    def extract(
        self,
        landmarks: Optional[List],
        timestamp: Optional[float] = None,
    ) -> FaceState:
        """
        Extract features from landmark list.
        If landmarks is None or empty, handles short-term grace period or timeout.
        """
        now = timestamp if timestamp is not None else time.time()

        # Handle face loss / no detection
        if not landmarks or len(landmarks) == 0:
            return self._handle_face_loss(now)

        self._last_valid_time = now

        # 1. Bounding box & Center
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        raw_cx = (xmin + xmax) / 2.0
        raw_cy = (ymin + ymax) / 2.0

        raw_w = xmax - xmin
        raw_h = ymax - ymin
        raw_scale = math.sqrt(raw_w * raw_w + raw_h * raw_h)

        # 2. Orientation (Roll, Yaw, Pitch)
        n_points = len(landmarks)

        # Roll: eye-line angle
        if n_points > RIGHT_EYE_OUTER and n_points > LEFT_EYE_OUTER:
            l_eye = landmarks[LEFT_EYE_OUTER]
            r_eye = landmarks[RIGHT_EYE_OUTER]
            dx_e = r_eye.x - l_eye.x
            dy_e = r_eye.y - l_eye.y
            raw_roll = math.degrees(math.atan2(dy_e, dx_e))
        else:
            raw_roll = 0.0

        # Yaw: nose horizontal displacement relative to face boundary
        if n_points > RIGHT_FACE_EDGE and n_points > LEFT_FACE_EDGE and n_points > NOSE_TIP:
            l_face = landmarks[LEFT_FACE_EDGE]
            r_face = landmarks[RIGHT_FACE_EDGE]
            nose = landmarks[NOSE_TIP]
            face_w = max(abs(r_face.x - l_face.x), 1e-4)
            mid_x = (l_face.x + r_face.x) / 2.0
            rel_x = (nose.x - mid_x) / (face_w / 2.0)
            rel_x = max(-1.0, min(1.0, rel_x))
            raw_yaw = rel_x * 45.0
        else:
            raw_yaw = 0.0

        # Pitch: nose vertical displacement relative to eye-line and chin
        if n_points > CHIN and n_points > NOSE_TIP and n_points > RIGHT_EYE_OUTER and n_points > LEFT_EYE_OUTER:
            eye_mid_y = (landmarks[LEFT_EYE_OUTER].y + landmarks[RIGHT_EYE_OUTER].y) / 2.0
            chin_y = landmarks[CHIN].y
            face_h = max(abs(chin_y - eye_mid_y), 1e-4)
            rel_y = (landmarks[NOSE_TIP].y - eye_mid_y) / face_h
            raw_pitch = (rel_y - 0.40) * 90.0
            raw_pitch = max(-60.0, min(60.0, raw_pitch))
        else:
            raw_pitch = 0.0

        # 3. Apply Exponential Smoothing
        cx, cy = self._center_smoother.update(raw_cx, raw_cy)
        w      = self._width_smoother.update(raw_w)
        h      = self._height_smoother.update(raw_h)
        scale  = self._scale_smoother.update(raw_scale)

        roll  = self._roll_smoother.update(raw_roll)
        yaw   = self._yaw_smoother.update(raw_yaw)
        pitch = self._pitch_smoother.update(raw_pitch)

        # 4. Calculate Movement (dx, dy, speed)
        dx, dy, speed = 0.0, 0.0, 0.0
        if self._prev_center is not None and self._prev_time is not None:
            dt = max(now - self._prev_time, 1e-3)
            dx = cx - self._prev_center[0]
            dy = cy - self._prev_center[1]
            dist = math.sqrt(dx * dx + dy * dy)
            speed = dist / dt

        self._prev_center = (cx, cy)
        self._prev_time   = now

        state = FaceState(
            detected=True,
            center_x=cx,
            center_y=cy,
            width=w,
            height=h,
            scale=scale,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            dx=dx,
            dy=dy,
            speed=speed,
            timestamp=now,
        )

        self._last_state = state
        return state

    def _handle_face_loss(self, now: float) -> FaceState:
        """
        Handle short-term face loss grace period.
        If within FACE_LOST_TIMEOUT_MS, hold last stable anchor with detected=False.
        Otherwise, reset smoothers.
        """
        if (
            self._last_valid_time is not None
            and self._last_state is not None
            and (now - self._last_valid_time) * 1000.0 < self.lost_timeout_ms
        ):
            # Grace period active
            return FaceState(
                detected=False,
                center_x=self._last_state.center_x,
                center_y=self._last_state.center_y,
                width=self._last_state.width,
                height=self._last_state.height,
                scale=self._last_state.scale,
                yaw=self._last_state.yaw,
                pitch=self._last_state.pitch,
                roll=self._last_state.roll,
                dx=0.0,
                dy=0.0,
                speed=0.0,
                timestamp=now,
            )

        # Timeout reached / completely lost
        self.reset()
        return FaceState(
            detected=False,
            center_x=0.0,
            center_y=0.0,
            width=0.0,
            height=0.0,
            scale=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            dx=0.0,
            dy=0.0,
            speed=0.0,
            timestamp=now,
        )

    def reset(self):
        """Reset internal smoothers and tracking history."""
        self._center_smoother.reset()
        self._width_smoother.reset()
        self._height_smoother.reset()
        self._scale_smoother.reset()

        self._yaw_smoother.reset()
        self._pitch_smoother.reset()
        self._roll_smoother.reset()

        self._prev_center = None
        self._prev_time = None
        self._last_valid_time = None
        self._last_state = None
