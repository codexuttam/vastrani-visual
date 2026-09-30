"""
tests/test_face_smoothing.py

Unit tests for Phase 7 ExponentialSmoother, PointSmoother, and AngleSmoother.
Requires NO camera or GPU hardware.
"""

import pytest
from vision.smoothing import ExponentialSmoother, PointSmoother, AngleSmoother


def test_exponential_smoother_gradual_step():
    """Verify ExponentialSmoother moves gradually toward new values."""
    smoother = ExponentialSmoother(alpha=0.5)

    # Initial value
    v1 = smoother.update(10.0)
    assert v1 == 10.0

    # Step to 20.0
    v2 = smoother.update(20.0)
    assert v2 == 15.0  # 0.5 * 20 + 0.5 * 10

    v3 = smoother.update(20.0)
    assert v3 == 17.5  # 0.5 * 20 + 0.5 * 15


def test_point_smoother():
    """Verify PointSmoother smoothes 2D point coordinates."""
    smoother = PointSmoother(alpha=0.5)

    p1 = smoother.update(0.0, 0.0)
    assert p1 == (0.0, 0.0)

    p2 = smoother.update(1.0, 2.0)
    assert p2 == (0.5, 1.0)


def test_angle_smoother_wrapping():
    """Verify AngleSmoother handles angle wrapping correctly."""
    smoother = AngleSmoother(alpha=0.5)

    a1 = smoother.update(170.0)
    assert a1 == 170.0

    # Crossing boundary: 170 to -170 (diff is -20 deg)
    a2 = smoother.update(-170.0)
    assert round(a2, 2) == 180.0 or round(a2, 2) == -180.0

    smoother.reset()
    assert smoother.value is None
