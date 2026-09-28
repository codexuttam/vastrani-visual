"""
tests/test_features.py

Unit tests for Phase 3 hand feature extraction.
Tests geometry helpers, normalization, smoothing, and missing-landmark safety.
"""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gestures.features import (
    calculate_angle,
    calculate_distance,
    FeatureExtractor,
    HandFeatureSet,
    DIRECTION_STATIONARY,
)
from gestures.smoothing import EMAFilter, VectorEMAFilter


# ─── calculate_angle ──────────────────────────────────────────────────────────

def test_angle_90():
    a = {"x": 0.0, "y": 1.0}
    b = {"x": 0.0, "y": 0.0}
    c = {"x": 1.0, "y": 0.0}
    assert abs(calculate_angle(a, b, c) - 90.0) < 0.5


def test_angle_180():
    a = {"x": -1.0, "y": 0.0}
    b = {"x":  0.0, "y": 0.0}
    c = {"x":  1.0, "y": 0.0}
    assert abs(calculate_angle(a, b, c) - 180.0) < 0.5


def test_angle_45():
    a = {"x": 1.0, "y": 0.0}
    b = {"x": 0.0, "y": 0.0}
    c = {"x": 1.0, "y": 1.0}
    assert abs(calculate_angle(a, b, c) - 45.0) < 1.0


def test_angle_degenerate_same_point():
    """Should return 0.0 without raising when points coincide."""
    a = {"x": 0.0, "y": 0.0}
    result = calculate_angle(a, a, a)
    assert result == 0.0


# ─── calculate_distance ───────────────────────────────────────────────────────

def test_distance_known_points():
    a = {"x": 0.0, "y": 0.0}
    b = {"x": 3.0, "y": 4.0}
    assert abs(calculate_distance(a, b) - 5.0) < 1e-6


def test_distance_same_point():
    a = {"x": 0.5, "y": 0.5}
    assert calculate_distance(a, a) == 0.0


# ─── EMAFilter ────────────────────────────────────────────────────────────────

def test_ema_first_value_passthrough():
    f = EMAFilter(alpha=0.5)
    result = f.update(10.0)
    assert result == 10.0


def test_ema_smoothing():
    f = EMAFilter(alpha=0.5)
    f.update(0.0)           # init = 0
    result = f.update(10.0) # 0.5*10 + 0.5*0 = 5
    assert abs(result - 5.0) < 1e-6


def test_ema_reset():
    f = EMAFilter(alpha=0.5)
    f.update(100.0)
    f.reset()
    result = f.update(1.0)
    assert result == 1.0  # First value after reset is passed through


def test_ema_invalid_alpha():
    with pytest.raises(ValueError):
        EMAFilter(alpha=0.0)
    with pytest.raises(ValueError):
        EMAFilter(alpha=1.5)


# ─── VectorEMAFilter ──────────────────────────────────────────────────────────

def test_vector_ema_size_mismatch():
    f = VectorEMAFilter(3, alpha=0.5)
    with pytest.raises(ValueError):
        f.update([1.0, 2.0])  # wrong length


def test_vector_ema_correct_update():
    f = VectorEMAFilter(2, alpha=0.5)
    f.update([0.0, 0.0])
    result = f.update([10.0, 20.0])
    assert abs(result[0] - 5.0) < 1e-6
    assert abs(result[1] - 10.0) < 1e-6


# ─── FeatureExtractor ─────────────────────────────────────────────────────────

def test_extract_none_landmarks_returns_invalid():
    ex = FeatureExtractor()
    fs = ex.extract(None)
    assert fs.valid is False


def test_extract_empty_list_returns_invalid():
    ex = FeatureExtractor()
    fs = ex.extract([])
    assert fs.valid is False


def test_extract_incomplete_landmarks_returns_invalid():
    """Only 5 landmarks — should return invalid without crashing."""
    ex = FeatureExtractor()

    class FakeLM:
        def __init__(self):
            self.x = self.y = self.z = 0.0

    fs = ex.extract([FakeLM()] * 5)
    assert fs.valid is False


def test_palm_center_calculation():
    """Palm center should be the average of wrist + 4 MCP landmarks."""
    ex = FeatureExtractor()

    class FakeLM:
        def __init__(self, x=0.5, y=0.5, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    # All landmarks at (0.5, 0.5, 0.0) → palm center should be (0.5, 0.5)
    lms = [FakeLM()] * 21
    fs = ex.extract(lms)
    assert fs.valid is True
    assert abs(fs.palm_x - 0.5) < 0.01
    assert abs(fs.palm_y - 0.5) < 0.01


def test_no_crash_on_repeated_missing_hand():
    """Calling extract with None repeatedly should never raise."""
    ex = FeatureExtractor()
    for _ in range(30):
        fs = ex.extract(None)
        assert fs.valid is False


# ─── Config defaults ──────────────────────────────────────────────────────────

def test_config_defaults():
    from config import Config
    cfg = Config()
    assert cfg.CAMERA_INDEX >= 0
    assert cfg.TARGET_FPS == 30
    assert cfg.FRAME_WIDTH == 1280
    assert cfg.FRAME_HEIGHT == 720
    assert 0.0 < cfg.SMOOTHING_ALPHA <= 1.0


# ─── Camera initialization structure ─────────────────────────────────────────

def test_camera_init_structure():
    from vision.camera import Camera
    cam = Camera(camera_index=1, target_fps=60, width=1920, height=1080)
    assert cam.camera_index == 1
    assert cam.target_fps == 60
    assert cam.width == 1920
    assert cam.height == 1080
    assert cam.cap is None
