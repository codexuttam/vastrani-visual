"""
ui/components/vision_panel.py

Camera & Primary Vision Panel component.
Handles live camera view annotations, camera offline overlays, and recovery prompts.
"""

import cv2
import numpy as np
from ui.design_system import (
    CYAN, RED_ERROR, WHITE, DARK_BG, PANEL_BORDER, FONT_PRIMARY,
    draw_glass_panel,
)


class VisionPanelComponent:
    """Renders vision viewport hints and recovery state overlays."""

    def render_offline_overlay(self, frame: np.ndarray, message: str = "Camera could not be initialized."):
        h, w, _ = frame.shape

        # Translucent dark backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Center card
        cw, ch = 480, 200
        cx, cy = (w - cw) // 2, (h - ch) // 2
        draw_glass_panel(frame, cx, cy, cw, ch, bg_color=DARK_BG, border_color=RED_ERROR, alpha=0.95, title="CAMERA ERROR")

        cv2.putText(frame, "CAMERA OFFLINE", (cx + 140, cy + 60), FONT_PRIMARY, 0.8, RED_ERROR, 2, cv2.LINE_AA)
        cv2.putText(frame, message, (cx + 40, cy + 105), FONT_PRIMARY, 0.5, WHITE, 1, cv2.LINE_AA)

        # Retry hint button
        btn_x, btn_y, btn_w, btn_h = cx + 160, cy + 135, 160, 36
        cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), RED_ERROR, -1)
        cv2.putText(frame, "[ RETRY (C) ]", (btn_x + 24, btn_y + 24), FONT_PRIMARY, 0.55, WHITE, 1, cv2.LINE_AA)

    def render_no_hand_hint(self, frame: np.ndarray):
        h, w, _ = frame.shape
        hint = "Raise your hand to begin gesture control"
        (tw, th), _ = cv2.getTextSize(hint, FONT_PRIMARY, 0.55, 1)
        draw_glass_panel(frame, (w - tw) // 2 - 14, (h // 2) - 20, tw + 28, 40, bg_color=DARK_BG, border_color=PANEL_BORDER, alpha=0.7)
        cv2.putText(frame, hint, ((w - tw) // 2, (h // 2) + 6), FONT_PRIMARY, 0.55, CYAN, 1, cv2.LINE_AA)
