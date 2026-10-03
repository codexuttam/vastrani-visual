"""
ui/components/event_stream_panel.py

Real-Time Event Stream Log Panel component.
Renders real-time system events feed with level color coding.
"""

import cv2
import numpy as np
from typing import List
from ui.contracts import EventItem
from ui.design_system import (
    CYAN, GREEN_SUCCESS, RED_ERROR, ORANGE_WARN, TEXT_PRIMARY, TEXT_MUTED, DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, sanitize_text,
)


class EventStreamPanelComponent:
    """Renders real-time system event log."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        events: List[EventItem],
        max_display: int = 6,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="REAL-TIME EVENT STREAM")

        curr_y = y + 40
        if not events:
            cv2.putText(frame, "Waiting for system events...", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        display_items = events[-max_display:]
        for ev in reversed(display_items):
            lvl_color = CYAN
            if ev.level == "SUCCESS":
                lvl_color = GREEN_SUCCESS
            elif ev.level == "WARN":
                lvl_color = ORANGE_WARN
            elif ev.level == "ERROR":
                lvl_color = RED_ERROR

            prefix = f"{ev.timestamp_str} [{ev.source}]"
            cv2.putText(frame, prefix, (x + 12, curr_y), FONT_MONO, 0.65, lvl_color, 1, cv2.LINE_AA)

            msg_text = sanitize_text(ev.message, 32)
            cv2.putText(frame, msg_text, (x + 150, curr_y), FONT_PRIMARY, 0.42, TEXT_PRIMARY, 1, cv2.LINE_AA)

            curr_y += 22
            if curr_y > y + h - 14:
                break
