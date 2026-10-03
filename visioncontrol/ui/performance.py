"""
ui/performance.py

Real-time system performance monitor for VisionControl HUD.
Tracks FPS, pipeline latencies (vision, face, intent, serial), CPU %, and Memory MB.
"""

import time
import os
import resource
from typing import Optional
from ui.contracts import PerformanceMetrics


class PerformanceMonitor:
    """Calculates and stores system performance metrics."""

    def __init__(self, smoothing_alpha: float = 0.2):
        self.alpha = smoothing_alpha
        self.metrics = PerformanceMetrics()
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._fps_accumulator = 0.0

    def update_frame(
        self,
        fps: float,
        vision_ms: float = 0.0,
        face_ms: float = 0.0,
        intent_ms: float = 0.0,
        serial_ms: float = 0.0,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> PerformanceMetrics:
        """Updates frame metrics with exponential smoothing."""
        self.metrics.fps = fps if self.metrics.fps == 0.0 else (self.alpha * fps + (1 - self.alpha) * self.metrics.fps)
        self.metrics.vision_ms = vision_ms
        self.metrics.face_ms = face_ms
        self.metrics.intent_ms = intent_ms
        self.metrics.serial_ms = serial_ms
        self.metrics.frame_width = frame_width
        self.metrics.frame_height = frame_height

        # Fetch Memory usage in MB safely
        try:
            # ru_maxrss is in bytes on mac, kbytes on linux
            rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == "Darwin":
                self.metrics.memory_mb = float(rusage) / (1024 * 1024)
            else:
                self.metrics.memory_mb = float(rusage) / 1024
        except Exception:
            self.metrics.memory_mb = None

        return self.metrics
