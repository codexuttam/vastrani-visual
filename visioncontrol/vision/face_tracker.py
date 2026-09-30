"""
vision/face_tracker.py

Phase 7: Detects face landmarks using MediaPipe FaceMesh.

Supports both legacy mp.solutions API (mediapipe < 1.0) and the new
Tasks-based API (mediapipe >= 1.0) via runtime version detection.

Persistent model initialization happens ONCE during __init__.
Deterministic primary face selection (largest face region).
Draws face landmarks and anchor indicator when debug_face=True.
"""

import time
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
from packaging.version import Version

from vision.hand_tracker import _LM

_MP_VERSION = Version(mp.__version__)
_USE_LEGACY  = _MP_VERSION < Version("1.0.0")

# Key landmark connection pairs for subtle debug rendering
FACEMESH_CONTOURS = [
    # Left Eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
    # Right Eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    # Lips Outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 291),
    # Face Oval (partial outline)
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162), (162, 127), (127, 234),
]


class FaceTracker:
    """
    Detects face landmarks locally using MediaPipe FaceMesh.
    Maintains persistent model instance.
    Selects primary face deterministically.
    """

    def __init__(
        self,
        max_num_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_faces = max_num_faces
        self.inference_ms: float = 0.0

        if _USE_LEGACY:
            self._init_legacy(max_num_faces, min_detection_confidence, min_tracking_confidence)
        else:
            self._init_tasks(max_num_faces, min_detection_confidence, min_tracking_confidence)

    # ── Legacy API (mediapipe < 1.0) ──────────────────────────────────────────

    def _init_legacy(self, max_faces, det_conf, track_conf):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh    = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self._process_fn = self._process_legacy

    def _process_legacy(self, frame, debug_face: bool) -> Tuple[object, Optional[List], List[List]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0 = time.time()
        results = self._face_mesh.process(rgb)
        self.inference_ms = (time.time() - t0) * 1000.0

        rgb.flags.writeable = True
        annotated = frame.copy()

        all_faces_landmarks = []
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                adapted = [_LM(lm.x, lm.y, lm.z) for lm in face_lms.landmark]
                all_faces_landmarks.append(adapted)

        # Primary face selection (Section 17: largest face area)
        primary_lms = self._select_primary_face(all_faces_landmarks)

        if debug_face and primary_lms:
            annotated = self._draw_debug_overlay(annotated, primary_lms)

        return annotated, primary_lms, all_faces_landmarks

    # ── Tasks API (mediapipe >= 1.0) ──────────────────────────────────────────

    def _init_tasks(self, max_faces, det_conf, track_conf):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os

        model_path = os.path.join(
            os.path.dirname(__file__), "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            print("Downloading MediaPipe face landmarker model (~4 MB)…")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded.")

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker  = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._process_fn  = self._process_tasks

    def _process_tasks(self, frame, debug_face: bool) -> Tuple[object, Optional[List], List[List]]:
        import mediapipe as mp_mod
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33

        t0 = time.time()
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self.inference_ms = (time.time() - t0) * 1000.0

        annotated = frame.copy()
        all_faces_landmarks = []

        if result.face_landmarks:
            for face_lms in result.face_landmarks:
                adapted = [_LM(lm.x, lm.y, lm.z) for lm in face_lms]
                all_faces_landmarks.append(adapted)

        primary_lms = self._select_primary_face(all_faces_landmarks)

        if debug_face and primary_lms:
            annotated = self._draw_debug_overlay(annotated, primary_lms)

        return annotated, primary_lms, all_faces_landmarks

    # ── Primary Face Selection ────────────────────────────────────────────────

    def _select_primary_face(self, faces: List[List]) -> Optional[List]:
        """
        Deterministically selects the primary face.
        Rule: Largest face bounding box area (width * height).
        """
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]

        best_face = None
        max_area = -1.0

        for face in faces:
            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > max_area:
                max_area = area
                best_face = face

        return best_face

    # ── Debug Visualization ───────────────────────────────────────────────────

    def _draw_debug_overlay(self, frame, landmarks: List) -> object:
        """
        Draws subtle face landmarks and anchor indicator on frame.
        """
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # 1. Subtle landmark dots & contours
        n_pts = len(pts)
        for a, b in FACEMESH_CONTOURS:
            if a < n_pts and b < n_pts:
                cv2.line(frame, pts[a], pts[b], (180, 220, 100), 1, cv2.LINE_AA)

        # 2. Section 20: Face Anchor Box & Crosshair Indicator
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        cx = int(((min(xs) + max(xs)) / 2.0) * w)
        cy = int(((min(ys) + max(ys)) / 2.0) * h)

        # Anchor Crosshair
        arm = 12
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Anchor label box
        box_w, box_h = 100, 20
        cv2.rectangle(frame, (cx - 50, cy - 30), (cx + 50, cy - 10), (30, 30, 30), -1)
        cv2.rectangle(frame, (cx - 50, cy - 30), (cx + 50, cy - 10), (0, 255, 255), 1)
        cv2.putText(
            frame,
            "FACE ANCHOR",
            (cx - 44, cy - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    # ── Public Interface ──────────────────────────────────────────────────────

    def process(self, frame, debug_face: bool = False) -> Tuple[object, Optional[List], List[List]]:
        """
        Process a BGR camera frame.

        Returns:
            (annotated_frame, primary_landmarks, list_of_all_faces_landmarks)
        """
        if frame is None:
            return frame, None, []
        try:
            return self._process_fn(frame, debug_face)
        except Exception as e:
            print(f"[FaceTracker] Error: {e}")
            return frame, None, []

    def close(self):
        """Release MediaPipe resources."""
        if _USE_LEGACY:
            self._face_mesh.close()
        else:
            self._landmarker.close()
