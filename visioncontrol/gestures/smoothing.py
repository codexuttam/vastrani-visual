"""
gestures/smoothing.py

Exponential moving average (EMA) smoother.
Works for any scalar or list of scalars.
Alpha controls responsiveness vs. stability:
  - closer to 1.0 → faster response, more jitter
  - closer to 0.0 → smoother, more lag
"""

from typing import Union, List


class EMAFilter:
    """
    Exponential moving average filter for a single scalar value.

    smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, alpha: float = 0.35):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value: float = 0.0
        self._initialized: bool = False

    def update(self, value: float) -> float:
        """Push a new value and return the smoothed result."""
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        """Reset filter state (e.g. when hand disappears)."""
        self._initialized = False
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value


class VectorEMAFilter:
    """
    Applies an EMA filter independently to each element of a fixed-size vector.
    Useful for smoothing palm positions, finger angles, etc.
    """

    def __init__(self, size: int, alpha: float = 0.35):
        self.filters = [EMAFilter(alpha) for _ in range(size)]

    def update(self, values: List[float]) -> List[float]:
        """
        Push a list of values and return the smoothed list.
        Raises ValueError if length does not match initialised size.
        """
        if len(values) != len(self.filters):
            raise ValueError(
                f"Expected {len(self.filters)} values, got {len(values)}"
            )
        return [f.update(v) for f, v in zip(self.filters, values)]

    def reset(self):
        for f in self.filters:
            f.reset()
