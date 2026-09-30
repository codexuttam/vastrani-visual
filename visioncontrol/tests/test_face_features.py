"""
tests/test_face_features.py

Unit tests for Phase 7 FaceFeatureExtractor and FaceState calculations.
Requires NO camera or GPU hardware.
"""

import math
import time
import pytest
from vision.hand_tracker import _LM
from vision.face_features import (
    FaceFeatureExtractor,
    FaceLandmarks,
    FaceAnchor,
    FaceState,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    NOSE_TIP,
    CHIN,
    FOREHEAD,
    LEFT_FACE_EDGE,
    RIGHT_FACE_EDGE,
)


def _create_synthetic_face(
    center_x: float = 0.5,
    center_y: float = 0.5,
    width: float = 0.4,
    height: float = 0.4,
    roll_deg: float = 0.0,
) -> list:
    """Helper to create a synthetic 468-point landmark array with configurable center, scale, and tilt."""
    lms = [_LM(center_x, center_y, 0.0) for _ in range(468)]

    half_w = width / 2.0
    half_h = height / 2.0

    # Boundaries
    lms[LEFT_FACE_EDGE]  = _LM(center_x - half_w, center_y, 0.0)
    lms[RIGHT_FACE_EDGE] = _LM(center_x + half_w, center_y, 0.0)
    lms[FOREHEAD]        = _LM(center_x, center_y - half_h, 0.0)
    lms[CHIN]            = _LM(center_x, center_y + half_h, 0.0)
    lms[NOSE_TIP]        = _LM(center_x, center_y, 0.0)

    # Eyes with optional roll rotation
    rad = math.radians(roll_deg)
    eye_offset_x = half_w * 0.5
    eye_offset_y = -half_h * 0.3

    lx_rot = eye_offset_x * math.cos(rad) - eye_offset_y * math.sin(rad)
    ly_rot = eye_offset_x * math.sin(rad) + eye_offset_y * math.cos(rad)

    rx_rot = (-eye_offset_x) * math.cos(rad) - eye_offset_y * math.sin(rad)
    ry_rot = (-eye_offset_x) * math.sin(rad) + eye_offset_y * math.cos(rad)

    lms[LEFT_EYE_OUTER]  = _LM(center_x - lx_rot, center_y - ly_rot, 0.0)
    lms[RIGHT_EYE_OUTER] = _LM(center_x - rx_rot, center_y - ry_rot, 0.0)

    return lms


def test_face_features_center_and_size():
    """Verify calculated center, width, height, and scale."""
    extractor = FaceFeatureExtractor(smoothing_alpha=1.0)  # 1.0 for instant response
    lms = _create_synthetic_face(center_x=0.5, center_y=0.5, width=0.4, height=0.4)

    state = extractor.extract(lms, timestamp=1.0)

    assert state.detected is True
    assert pytest.approx(state.center_x, abs=0.01) == 0.5
    assert pytest.approx(state.center_y, abs=0.01) == 0.5
    assert pytest.approx(state.width, abs=0.01) == 0.4
    assert pytest.approx(state.height, abs=0.01) == 0.4
    assert state.scale > 0.0


def test_face_features_scale_increases_when_closer():
    """Verify scale increases when face size becomes larger (moving closer)."""
    extractor = FaceFeatureExtractor(smoothing_alpha=1.0)

    small_face = _create_synthetic_face(width=0.2, height=0.2)
    large_face = _create_synthetic_face(width=0.6, height=0.6)

    s1 = extractor.extract(small_face, timestamp=1.0)
    s2 = extractor.extract(large_face, timestamp=1.1)

    assert s2.scale > s1.scale


def test_face_features_roll():
    """Verify head roll angle calculation on tilted landmarks."""
    extractor = FaceFeatureExtractor(smoothing_alpha=1.0)
    tilted_lms = _create_synthetic_face(roll_deg=15.0)

    state = extractor.extract(tilted_lms, timestamp=1.0)
    assert pytest.approx(state.roll, abs=2.0) == 15.0


def test_face_features_movement_dx_dy():
    """Verify horizontal/vertical movement dx, dy, and speed."""
    extractor = FaceFeatureExtractor(smoothing_alpha=1.0)

    lms1 = _create_synthetic_face(center_x=0.4, center_y=0.5)
    lms2 = _create_synthetic_face(center_x=0.5, center_y=0.5)

    _ = extractor.extract(lms1, timestamp=1.0)
    state2 = extractor.extract(lms2, timestamp=2.0)

    assert state2.dx > 0.0
    assert state2.speed > 0.0


def test_face_features_face_loss_and_reacquisition():
    """Verify grace period when face is lost, timeout, and reacquisition."""
    extractor = FaceFeatureExtractor(smoothing_alpha=1.0, lost_timeout_ms=500.0)
    lms = _create_synthetic_face(center_x=0.5, center_y=0.5)

    # 1. Active detection
    s1 = extractor.extract(lms, timestamp=1.0)
    assert s1.detected is True

    # 2. Loss within grace period (200ms < 500ms)
    s2 = extractor.extract(None, timestamp=1.2)
    assert s2.detected is False
    assert s2.center_x == 0.5  # Held during grace period

    # 3. Timeout exceeded (700ms > 500ms)
    s3 = extractor.extract(None, timestamp=1.8)
    assert s3.detected is False
    assert s3.center_x == 0.0  # Reset after timeout

    # 4. Reacquisition
    s4 = extractor.extract(lms, timestamp=2.0)
    assert s4.detected is True
    assert s4.center_x == 0.5
