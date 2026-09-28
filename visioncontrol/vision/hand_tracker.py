"""
vision/hand_tracker.py

Phase 2: Detects hand landmarks using MediaPipe.

Supports both legacy mp.solutions API (mediapipe < 1.0) and the new
Tasks-based API (mediapipe >= 1.0) via runtime version detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
from packaging.version import Version

_MP_VERSION = Version(mp.__version__)
_USE_LEGACY  = _MP_VERSION < Version("1.0.0")

# MediaPipe hand landmark indices (same in both APIs)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Connection pairs (same in both APIs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


# ─── Shared NormalizedLandmark adapter ────────────────────────────────────────

class _LM:
    """Normalised landmark wrapper — same interface for both APIs."""
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


# ─── HandTracker ──────────────────────────────────────────────────────────────

class HandTracker:
    """
    Detects hand landmarks using MediaPipe Hands.
    Works with both mediapipe < 1.0 (legacy solutions API)
    and mediapipe >= 1.0 (Tasks API).
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        self._max_hands = max_num_hands

        if _USE_LEGACY:
            self._init_legacy(max_num_hands, min_detection_confidence, min_tracking_confidence)
        else:
            self._init_tasks(max_num_hands, min_detection_confidence, min_tracking_confidence)

    # ── Legacy API (mediapipe < 1.0) ──────────────────────────────────────────

    def _init_legacy(self, max_hands, det_conf, track_conf):
        self._mp_hands   = mp.solutions.hands
        self._mp_draw    = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles
        self._hands      = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self._process_fn = self._process_legacy

    def _process_legacy(self, frame) -> Tuple:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    annotated,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
                landmarks_list.append(hand_lms.landmark)

        return annotated, landmarks_list

    # ── Tasks API (mediapipe >= 1.0) ──────────────────────────────────────────

    def _init_tasks(self, max_hands, det_conf, track_conf):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os, tempfile

        model_path = os.path.join(
            os.path.dirname(__file__), "hand_landmarker.task"
        )
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand landmarker model (~14 MB)…")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded.")

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker  = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._process_fn  = self._process_tasks

    def _process_tasks(self, frame) -> Tuple:
        import mediapipe as mp_mod
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mod.Image(
            image_format=mp_mod.ImageFormat.SRGB, data=rgb
        )
        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        annotated = frame.copy()
        landmarks_list = []

        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in result.hand_landmarks:
                # Convert to _LM adapter objects
                adapted = [_LM(lm.x, lm.y, lm.z) for lm in hand_lms]
                landmarks_list.append(adapted)
                # Draw skeleton manually
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for i, pt in enumerate(pts):
                    cv2.circle(annotated, pt, 4, (0, 220, 255), -1)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(annotated, pts[a], pts[b], (200, 200, 200), 2)

        return annotated, landmarks_list

    # ── Public interface ──────────────────────────────────────────────────────

    def process(self, frame) -> Tuple:
        """
        Process a BGR frame.
        Returns (annotated_frame, list_of_landmark_lists).
        Each landmark list has 21 objects with .x .y .z in [0,1].
        """
        if frame is None:
            return frame, []
        try:
            return self._process_fn(frame)
        except Exception as e:
            print(f"[HandTracker] Error: {e}")
            return frame, []

    def close(self):
        """Release resources."""
        if _USE_LEGACY:
            self._hands.close()
        else:
            self._landmarker.close()
