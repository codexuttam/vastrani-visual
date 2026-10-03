"""
ui/components/system_panel.py

System Status & Header Banner component.
Renders title, system status, mode, camera indicator, and header controls.
"""

import cv2
import numpy as np
from typing import Optional
from ui.contracts import SystemStatus, HUDMode
from ui.design_system import (
    CYAN, VIOLET, WHITE, PANEL_BG, PANEL_BORDER, DARK_BG,
    GREEN_SUCCESS, RED_ERROR, ORANGE_WARN, YELLOW_DEGRADED, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, draw_status_badge, draw_badge, sanitize_text,
)


class SystemPanelComponent:
    """Renders persistent header banner and system status indicators."""

    def render(
        self,
        frame: np.ndarray,
        status: SystemStatus,
        mode: HUDMode,
        camera_live: bool = True,
        reduced_motion: bool = False,
    ):
        h, w, _ = frame.shape

        # Top Header Bar background
        draw_glass_panel(frame, 10, 10, w - 20, 42, bg_color=DARK_BG, border_color=PANEL_BORDER, alpha=0.85)

        # Title: VATSRANI VISION
        cv2.putText(frame, "VATSRANI VISION", (24, 38), FONT_PRIMARY, 0.75, CYAN, 2, cv2.LINE_AA)

        # Mode badge (STANDARD vs DEVELOPER)
        mode_color = VIOLET if mode == HUDMode.DEVELOPER else CYAN
        draw_badge(frame, 250, 20, mode.value, color=mode_color, text_color=DARK_BG, scale=0.45)

        # Reduced motion indicator if active
        if reduced_motion:
            draw_badge(frame, 350, 20, "NO ANIM", color=PANEL_BORDER, text_color=WHITE, scale=0.4)

        # Persistent System Status Badge
        draw_status_badge(frame, w - 380, 19, status)

        # Camera Live Indicator
        cam_color = GREEN_SUCCESS if camera_live else RED_ERROR
        cam_text = "● CAM LIVE" if camera_live else "✕ CAM OFFLINE"
        cv2.putText(frame, cam_text, (w - 150, 37), FONT_PRIMARY, 0.5, cam_color, 1, cv2.LINE_AA)
