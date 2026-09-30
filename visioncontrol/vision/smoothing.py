"""
vision/smoothing.py

Phase 7: Reusable exponential smoothing helper for scalar values, 2D points, and angles.
"""

import math
from typing import Tuple, Optional


class ExponentialSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing:
        smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha: float = alpha
        self._value: Optional[float] = None

    def update(self, current: float) -> float:
        if self._value is None:
            self._value = current
        else:
            self._value = self.alpha * current + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self) -> Optional[float]:
        return self._value


class PointSmoother:
    """Smoothes 2D points (x, y)."""

    def __init__(self, alpha: float = 0.35):
        self.x_smoother = ExponentialSmoother(alpha)
        self.y_smoother = ExponentialSmoother(alpha)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        sx = self.x_smoother.update(x)
        sy = self.y_smoother.update(y)
        return sx, sy

    def reset(self):
        self.x_smoother.reset()
        self.y_smoother.reset()


class AngleSmoother:
    """Smoothes angles in degrees while handling continuous wrapping correctly."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._angle: Optional[float] = None

    def update(self, current_deg: float) -> float:
        if self._angle is None:
            self._angle = current_deg
            return self._angle

        # Compute shortest angle difference
        diff = (current_deg - self._angle + 180.0) % 360.0 - 180.0
        self._angle = (self._angle + self.alpha * diff) % 360.0
        # Normalize back to [-180, 180]
        if self._angle > 180.0:
            self._angle -= 360.0
        return self._angle

    def reset(self):
        self._angle = None

    @property
    def value(self) -> Optional[float]:
        return self._angle
