"""
ui/components/performance_panel.py

System Performance & Telemetry Monitor Panel component.
Displays FPS, pipeline latencies (vision, face, intent, serial), CPU %, and Memory MB.
"""

import cv2
import numpy as np
from typing import Optional
from ui.contracts import PerformanceMetrics
from ui.design_system import (
    CYAN, VIOLET, GREEN_SUCCESS, TEXT_PRIMARY, TEXT_MUTED, DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel,
)


class PerformancePanelComponent:
    """Renders real-time performance telemetry."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        metrics: Optional[PerformanceMetrics] = None,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="PERFORMANCE MONITOR")

        curr_y = y + 40
        if metrics is None:
            cv2.putText(frame, "Metrics N/A", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        fps_val = f"{metrics.fps:.1f}" if metrics.fps > 0 else "N/A"
        cv2.putText(frame, f"FPS       : {fps_val}", (x + 12, curr_y), FONT_MONO, 0.7, GREEN_SUCCESS, 1, cv2.LINE_AA)
        curr_y += 22

        vis_val = f"{metrics.vision_ms:.1f}ms" if metrics.vision_ms > 0 else "N/A"
        cv2.putText(frame, f"Vision    : {vis_val}", (x + 12, curr_y), FONT_MONO, 0.7, CYAN, 1, cv2.LINE_AA)
        curr_y += 22

        if metrics.face_ms > 0:
            cv2.putText(frame, f"Face      : {metrics.face_ms:.1f}ms", (x + 12, curr_y), FONT_MONO, 0.7, VIOLET, 1, cv2.LINE_AA)
            curr_y += 22

        intent_val = f"{metrics.intent_ms:.1f}ms" if metrics.intent_ms > 0 else "N/A"
        cv2.putText(frame, f"Intent    : {intent_val}", (x + 12, curr_y), FONT_MONO, 0.7, TEXT_PRIMARY, 1, cv2.LINE_AA)
        curr_y += 22

        mem_val = f"{metrics.memory_mb:.1f} MB" if metrics.memory_mb is not None else "N/A"
        cv2.putText(frame, f"Memory    : {mem_val}", (x + 12, curr_y), FONT_MONO, 0.7, TEXT_MUTED, 1, cv2.LINE_AA)
