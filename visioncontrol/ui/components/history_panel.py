"""
ui/components/history_panel.py

Recent Command History Panel component.
Displays log of recently dispatched actions with timestamps and execution status.
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from devices.models import CommandRecord
from ui.design_system import (
    GREEN_SUCCESS, RED_ERROR, TEXT_PRIMARY, TEXT_MUTED, DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, sanitize_text,
)


class HistoryPanelComponent:
    """Renders command history log."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        command_history: Optional[List[CommandRecord]] = None,
        max_items: int = 5,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="COMMAND HISTORY")

        curr_y = y + 40
        if not command_history:
            cv2.putText(frame, "No commands recorded", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        # Render recent items top to bottom
        recent = command_history[-max_items:]
        for item in reversed(recent):
            t_str = time.strftime("%H:%M:%S", time.localtime(item.timestamp))
            action_text = getattr(item, "action", str(item))
            success = getattr(item, "success", True)

            icon = "✓" if success else "✕"
            icon_color = GREEN_SUCCESS if success else RED_ERROR

            line_time = f"{t_str} {icon}"
            cv2.putText(frame, line_time, (x + 12, curr_y), FONT_MONO, 0.7, icon_color, 1, cv2.LINE_AA)

            cmd_desc = sanitize_text(str(action_text), 28)
            cv2.putText(frame, cmd_desc, (x + 95, curr_y), FONT_PRIMARY, 0.42, TEXT_PRIMARY, 1, cv2.LINE_AA)

            curr_y += 22
            if curr_y > y + h - 16:
                break
