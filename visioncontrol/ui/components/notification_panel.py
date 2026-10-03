"""
ui/components/notification_panel.py

Centralized Toast Notification Overlay Renderer component.
Renders active notifications with category color accents and auto-dismiss progress indicators.
"""

import cv2
import numpy as np
from typing import List
from ui.contracts import NotificationItem, NotificationCategory
from ui.design_system import (
    CYAN, GREEN_SUCCESS, RED_ERROR, ORANGE_WARN, WHITE, TEXT_PRIMARY, TEXT_MUTED, DARK_BG, PANEL_BORDER, FONT_PRIMARY,
    draw_glass_panel, draw_progress_bar, sanitize_text,
)


class NotificationPanelComponent:
    """Renders active toast notifications overlay."""

    def render(
        self,
        frame: np.ndarray,
        notifications: List[NotificationItem],
        top_y: int = 60,
        right_margin: int = 20,
    ):
        h, w, _ = frame.shape
        if not notifications:
            return

        nw, nh = 320, 65
        curr_y = top_y

        for item in notifications:
            nx = w - nw - right_margin
            if curr_y + nh > h - 80:
                break

            cat_color = CYAN
            if item.category == NotificationCategory.SUCCESS:
                cat_color = GREEN_SUCCESS
            elif item.category == NotificationCategory.WARNING:
                cat_color = ORANGE_WARN
            elif item.category == NotificationCategory.ERROR:
                cat_color = RED_ERROR

            # Draw card
            draw_glass_panel(frame, nx, curr_y, nw, nh, bg_color=DARK_BG, border_color=cat_color, alpha=0.90)

            # Left accent stripe
            cv2.rectangle(frame, (nx, curr_y), (nx + 6, curr_y + nh), cat_color, -1)

            # Title
            cv2.putText(frame, sanitize_text(item.title, 30), (nx + 16, curr_y + 24), FONT_PRIMARY, 0.48, WHITE, 1, cv2.LINE_AA)

            # Message
            cv2.putText(frame, sanitize_text(item.message, 36), (nx + 16, curr_y + 45), FONT_PRIMARY, 0.42, TEXT_PRIMARY, 1, cv2.LINE_AA)

            # Bottom countdown progress bar
            progress = max(0.0, min(1.0, item.remaining_time / max(0.1, item.duration_sec)))
            draw_progress_bar(frame, nx + 6, curr_y + nh - 4, int((nw - 6) * progress), 3, progress=1.0, color=cat_color)

            curr_y += nh + 10
