"""
ui/components/face_panel.py

Face Tracking Status Panel component.
Displays real-time face tracking state, face detection count, and stability.
"""

import cv2
import numpy as np
from typing import Optional
from ui.design_system import (
    CYAN, GREEN_SUCCESS, RED_ERROR, VIOLET, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    DARK_BG, PANEL_BORDER, FONT_PRIMARY,
    draw_glass_panel,
)


class FacePanelComponent:
    """Renders face tracking diagnostics."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        face_state: Optional[object] = None,
        face_infer_ms: float = 0.0,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="FACE TRACKING")

        curr_y = y + 40
        detected = getattr(face_state, "detected", False) if face_state else False

        if not detected:
            cv2.putText(frame, "● NOT DETECTED", (x + 12, curr_y), FONT_PRIMARY, 0.55, RED_ERROR, 1, cv2.LINE_AA)
            cv2.putText(frame, "Face out of view", (x + 12, curr_y + 24), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        # Face active state
        cv2.putText(frame, "● ACTIVE", (x + 12, curr_y), FONT_PRIMARY, 0.55, GREEN_SUCCESS, 1, cv2.LINE_AA)
        curr_y += 24

        cv2.putText(frame, "Faces: 1", (x + 12, curr_y), FONT_PRIMARY, 0.5, CYAN, 1, cv2.LINE_AA)
        curr_y += 20

        stability = getattr(face_state, "stability", "STABLE") if face_state else "STABLE"
        cv2.putText(frame, f"Tracking: {stability}", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
        curr_y += 20

        if face_infer_ms > 0:
            cv2.putText(frame, f"Inference: {face_infer_ms:.1f}ms", (x + 12, curr_y), FONT_PRIMARY, 0.45, VIOLET, 1, cv2.LINE_AA)
