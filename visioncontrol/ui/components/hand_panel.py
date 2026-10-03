"""
ui/components/hand_panel.py

Hand Tracking Status & Gesture Visualizer component.
Displays real-time hand tracking status, recognized gesture, confidence %,
and individual finger extension matrix (INDEX, THUMB, MIDDLE, RING, PINKY).
"""

import cv2
import numpy as np
from typing import Optional
from gestures.features import HandFeatureSet
from gestures.types import GestureResult, ControlMode
from ui.design_system import (
    CYAN, GREEN_SUCCESS, RED_ERROR, WHITE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, draw_badge,
)


class HandPanelComponent:
    """Renders hand tracking diagnostics and finger state matrix."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        features: Optional[HandFeatureSet] = None,
        result: Optional[GestureResult] = None,
        mode: ControlMode = ControlMode.IDLE,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="HAND TRACKING")

        curr_y = y + 40
        if features is None or not features.valid:
            cv2.putText(frame, "● INACTIVE", (x + 12, curr_y), FONT_PRIMARY, 0.55, RED_ERROR, 1, cv2.LINE_AA)
            cv2.putText(frame, "No hand in frame", (x + 12, curr_y + 24), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        # Hand active state
        cv2.putText(frame, "● ACTIVE", (x + 12, curr_y), FONT_PRIMARY, 0.55, GREEN_SUCCESS, 1, cv2.LINE_AA)
        curr_y += 24

        # Active Gesture
        g_name = result.gesture.value if result else "IDLE"
        conf = int((result.confidence if result else 0.0) * 100)
        cv2.putText(frame, f"Gesture: {g_name}", (x + 12, curr_y), FONT_PRIMARY, 0.5, CYAN, 1, cv2.LINE_AA)
        curr_y += 20
        cv2.putText(frame, f"Confidence: {conf}%", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
        curr_y += 24

        # Finger State Matrix
        cv2.putText(frame, "FINGERS:", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
        curr_y += 18

        fingers = [
            ("THU", getattr(features, "thumb_extended", False)),
            ("IND", getattr(features, "index_extended", False)),
            ("MID", getattr(features, "middle_extended", False)),
            ("RNG", getattr(features, "ring_extended", False)),
            ("PNK", getattr(features, "pinky_extended", False)),
        ]

        fx = x + 12
        for name, ext in fingers:
            symbol = "[v]" if ext else "[-]"
            f_color = GREEN_SUCCESS if ext else (100, 105, 115)
            cv2.putText(frame, f"{name}{symbol}", (fx, curr_y), FONT_MONO, 0.7, f_color, 1, cv2.LINE_AA)
            fx += 52
