"""
tests/test_face_tracker.py

Unit tests for Phase 7 FaceTracker initialization and primary face selection logic.
Requires NO camera hardware.
"""

from unittest.mock import MagicMock
import pytest
from vision.hand_tracker import _LM
from vision.face_tracker import FaceTracker


def test_face_tracker_init():
    """Verify FaceTracker initializes without error."""
    tracker = FaceTracker(max_num_faces=2, min_detection_confidence=0.5)
    assert tracker.max_num_faces == 2
    assert tracker.inference_ms >= 0.0
    tracker.close()


def test_primary_face_selection_largest_face():
    """Verify primary face selection chooses the face with the largest bounding area."""
    tracker = FaceTracker(max_num_faces=2)

    # Small face (0.1 x 0.1)
    small_face = [_LM(0.1, 0.1, 0.0), _LM(0.2, 0.2, 0.0)]
    # Large face (0.5 x 0.5)
    large_face = [_LM(0.2, 0.2, 0.0), _LM(0.7, 0.7, 0.0)]

    selected = tracker._select_primary_face([small_face, large_face])
    assert selected == large_face

    tracker.close()


def test_face_tracker_process_none_frame():
    """Verify process returns None on invalid/None frame without crashing."""
    tracker = FaceTracker()
    frame, primary_lms, all_faces = tracker.process(None)
    assert frame is None
    assert primary_lms is None
    assert all_faces == []
    tracker.close()
