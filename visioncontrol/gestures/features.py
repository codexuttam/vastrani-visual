"""
gestures/features.py

Phase 3: Hand Feature Extraction.

Converts raw MediaPipe landmarks into stable, normalized numerical features
suitable for gesture recognition (Phase 4).

Architecture:
    Camera → HandTracker → Raw Landmarks → FeatureExtractor → HandFeatureSet → [Phase 4]

This module does NOT:
    - classify gestures
    - call OpenAI
    - send Arduino commands
    - contain UI logic
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List

from gestures.smoothing import EMAFilter, VectorEMAFilter

# ─── MediaPipe landmark indices ────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# ─── Movement direction labels ─────────────────────────────────────────────────
DIRECTION_LEFT = "LEFT"
DIRECTION_RIGHT = "RIGHT"
DIRECTION_UP = "UP"
DIRECTION_DOWN = "DOWN"
DIRECTION_STATIONARY = "STATIONARY"


# ─── Feature data structure ────────────────────────────────────────────────────

@dataclass
class HandFeatureSet:
    """
    Structured set of smoothed hand features extracted from MediaPipe landmarks.
    All coordinates are normalized to [0.0, 1.0] frame space.
    """
    valid: bool = False          # False when no hand is detected or data is bad

    # Palm center (normalized)
    palm_x: float = 0.0
    palm_y: float = 0.0
    palm_z: float = 0.0

    # Finger extension states
    thumb_extended: bool = False
    index_extended: bool = False
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False

    # Finger joint angles (degrees, at the PIP joint for non-thumb)
    thumb_angle: float = 0.0
    index_angle: float = 0.0
    middle_angle: float = 0.0
    ring_angle: float = 0.0
    pinky_angle: float = 0.0

    # Distance features (normalized by palm size)
    thumb_index_distance: float = 0.0
    palm_size: float = 0.0       # raw palm reference size for debugging

    # Hand orientation
    hand_angle: float = 0.0     # degrees, wrist→middle_MCP direction

    # Movement
    dx: float = 0.0
    dy: float = 0.0
    speed: float = 0.0
    direction: str = DIRECTION_STATIONARY


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def _lm(landmarks, idx):
    """Return the landmark at index as a plain dict with x, y, z."""
    lm = landmarks[idx]
    return {"x": lm.x, "y": lm.y, "z": lm.z}


def calculate_angle(a: dict, b: dict, c: dict) -> float:
    """
    Calculate the angle at point b formed by the vectors b→a and b→c.
    Points are dicts with 'x' and 'y' keys (z ignored for stability).
    Returns angle in degrees in [0, 180].
    """
    ax, ay = a["x"] - b["x"], a["y"] - b["y"]
    cx, cy = c["x"] - b["x"], c["y"] - b["y"]

    dot = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)

    if mag_a < 1e-9 or mag_c < 1e-9:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


def calculate_distance(a: dict, b: dict) -> float:
    """Euclidean distance between two points (using x, y only)."""
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ─── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts raw MediaPipe landmark lists into smoothed HandFeatureSet instances.

    Usage:
        extractor = FeatureExtractor(alpha=0.35)
        features = extractor.extract(landmarks)  # landmarks from HandTracker
    """

    def __init__(
        self,
        alpha: float = 0.35,
        finger_extension_threshold: float = 0.5,
        movement_stationary_threshold: float = 0.008,
    ):
        self.alpha = alpha
        self.finger_extension_threshold = finger_extension_threshold
        self.movement_stationary_threshold = movement_stationary_threshold

        # Smoothers
        self._palm_pos = VectorEMAFilter(3, alpha)      # x, y, z
        self._finger_angles = VectorEMAFilter(5, alpha) # thumb…pinky
        self._thumb_index_dist = EMAFilter(alpha)
        self._hand_angle = EMAFilter(alpha)
        self._dx = EMAFilter(alpha)
        self._dy = EMAFilter(alpha)
        self._speed = EMAFilter(alpha)

        # Previous palm for movement calculation
        self._prev_palm_x: Optional[float] = None
        self._prev_palm_y: Optional[float] = None

    def reset(self):
        """Reset all smoothers (call when hand disappears between frames)."""
        self._palm_pos.reset()
        self._finger_angles.reset()
        self._thumb_index_dist.reset()
        self._hand_angle.reset()
        self._dx.reset()
        self._dy.reset()
        self._speed.reset()
        self._prev_palm_x = None
        self._prev_palm_y = None

    def extract(self, landmarks) -> HandFeatureSet:
        """
        Extract and smooth features from a single hand's landmark list.

        Args:
            landmarks: list of 21 MediaPipe NormalizedLandmark objects, or None.

        Returns:
            HandFeatureSet. If landmarks are None/invalid, returns invalid set.
        """
        if landmarks is None or len(landmarks) < 21:
            self.reset()
            return HandFeatureSet(valid=False)

        try:
            return self._extract_safe(landmarks)
        except Exception:
            # Never crash the render loop due to bad landmark data
            return HandFeatureSet(valid=False)

    def _extract_safe(self, landmarks) -> HandFeatureSet:
        fs = HandFeatureSet(valid=True)

        # ── Palm center ───────────────────────────────────────────────────────
        palm_lm_indices = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        raw_px = sum(landmarks[i].x for i in palm_lm_indices) / len(palm_lm_indices)
        raw_py = sum(landmarks[i].y for i in palm_lm_indices) / len(palm_lm_indices)
        raw_pz = sum(landmarks[i].z for i in palm_lm_indices) / len(palm_lm_indices)

        smoothed_palm = self._palm_pos.update([raw_px, raw_py, raw_pz])
        fs.palm_x, fs.palm_y, fs.palm_z = smoothed_palm

        # ── Palm size reference ───────────────────────────────────────────────
        wrist = _lm(landmarks, WRIST)
        middle_mcp = _lm(landmarks, MIDDLE_MCP)
        fs.palm_size = calculate_distance(wrist, middle_mcp)
        palm_size_safe = max(fs.palm_size, 1e-6)

        # ── Finger angles (at PIP joint for fingers, IP for thumb) ────────────
        raw_angles = [
            calculate_angle(_lm(landmarks, THUMB_MCP), _lm(landmarks, THUMB_IP),  _lm(landmarks, THUMB_TIP)),
            calculate_angle(_lm(landmarks, INDEX_MCP),  _lm(landmarks, INDEX_PIP),  _lm(landmarks, INDEX_TIP)),
            calculate_angle(_lm(landmarks, MIDDLE_MCP), _lm(landmarks, MIDDLE_PIP), _lm(landmarks, MIDDLE_TIP)),
            calculate_angle(_lm(landmarks, RING_MCP),   _lm(landmarks, RING_PIP),   _lm(landmarks, RING_TIP)),
            calculate_angle(_lm(landmarks, PINKY_MCP),  _lm(landmarks, PINKY_PIP),  _lm(landmarks, PINKY_TIP)),
        ]
        smoothed_angles = self._finger_angles.update(raw_angles)
        fs.thumb_angle  = smoothed_angles[0]
        fs.index_angle  = smoothed_angles[1]
        fs.middle_angle = smoothed_angles[2]
        fs.ring_angle   = smoothed_angles[3]
        fs.pinky_angle  = smoothed_angles[4]

        # ── Finger extension (tip vs MCP y-comparison + angle heuristic) ──────
        # For non-thumb fingers: tip should be above MCP (smaller y = higher on screen)
        # We also use the straight-angle heuristic: angle > threshold * 180° means extended
        ANGLE_EXTENDED_DEG = self.finger_extension_threshold * 180.0

        # Thumb: use tip vs IP horizontal/vertical position (more reliable)
        thumb_tip = _lm(landmarks, THUMB_TIP)
        thumb_ip  = _lm(landmarks, THUMB_IP)
        thumb_mcp = _lm(landmarks, THUMB_MCP)
        # Thumb is extended when tip is far from the palm center
        thumb_tip_dist = calculate_distance({"x": thumb_tip["x"], "y": thumb_tip["y"]}, middle_mcp)
        fs.thumb_extended = (thumb_tip_dist / palm_size_safe) > 0.4

        # Index–Pinky: tip y significantly above MCP y and angle relatively open
        def _finger_extended(tip_idx, mcp_idx, pip_idx, angle_deg) -> bool:
            tip = _lm(landmarks, tip_idx)
            mcp = _lm(landmarks, mcp_idx)
            pip = _lm(landmarks, pip_idx)
            # tip above pip AND angle suggests open
            tip_above_pip = tip["y"] < pip["y"]
            return tip_above_pip and (angle_deg > ANGLE_EXTENDED_DEG)

        fs.index_extended  = _finger_extended(INDEX_TIP,  INDEX_MCP,  INDEX_PIP,  fs.index_angle)
        fs.middle_extended = _finger_extended(MIDDLE_TIP, MIDDLE_MCP, MIDDLE_PIP, fs.middle_angle)
        fs.ring_extended   = _finger_extended(RING_TIP,   RING_MCP,   RING_PIP,   fs.ring_angle)
        fs.pinky_extended  = _finger_extended(PINKY_TIP,  PINKY_MCP,  PINKY_PIP,  fs.pinky_angle)

        # ── Thumb-Index distance (normalized) ─────────────────────────────────
        raw_tid = calculate_distance(_lm(landmarks, THUMB_TIP), _lm(landmarks, INDEX_TIP)) / palm_size_safe
        fs.thumb_index_distance = self._thumb_index_dist.update(raw_tid)

        # ── Hand orientation ──────────────────────────────────────────────────
        dx_orient = middle_mcp["x"] - wrist["x"]
        dy_orient = middle_mcp["y"] - wrist["y"]
        raw_hand_angle = math.degrees(math.atan2(-dy_orient, dx_orient))
        fs.hand_angle = self._hand_angle.update(raw_hand_angle)

        # ── Movement ──────────────────────────────────────────────────────────
        if self._prev_palm_x is not None:
            raw_dx = fs.palm_x - self._prev_palm_x
            raw_dy = fs.palm_y - self._prev_palm_y
            raw_speed = math.hypot(raw_dx, raw_dy)
        else:
            raw_dx = raw_dy = raw_speed = 0.0

        self._prev_palm_x = fs.palm_x
        self._prev_palm_y = fs.palm_y

        fs.dx    = self._dx.update(raw_dx)
        fs.dy    = self._dy.update(raw_dy)
        fs.speed = self._speed.update(raw_speed)

        # Direction (geometric only — NOT a gesture label)
        if fs.speed < self.movement_stationary_threshold:
            fs.direction = DIRECTION_STATIONARY
        elif abs(fs.dx) >= abs(fs.dy):
            fs.direction = DIRECTION_RIGHT if fs.dx > 0 else DIRECTION_LEFT
        else:
            fs.direction = DIRECTION_DOWN if fs.dy > 0 else DIRECTION_UP

        return fs
