"""
ui/components/confirmation_panel.py

Confirmation Dialog Modal overlay card.
Displays high-priority confirmation requests from Intent Engine,
prompting the user with keyboard options [Y/ENTER] or [N/ESC].
"""

import cv2
import numpy as np
from typing import Optional
from ui.design_system import (
    AMBER, DARK_BG, WHITE, GREEN_SUCCESS, RED_ERROR, PANEL_BORDER, FONT_PRIMARY,
    draw_glass_panel, sanitize_text,
)


class ConfirmationDialogComponent:
    """Renders high-visibility modal card for user action confirmation."""

    def render(
        self,
        frame: np.ndarray,
        intent_result: Optional[object] = None,
        pending_action: Optional[str] = None,
    ):
        h, w, _ = frame.shape

        # Dim background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Modal dimensions & center calculation
        mw, mh = 500, 220
        mx, my = (w - mw) // 2, (h - mh) // 2

        draw_glass_panel(frame, mx, my, mw, mh, bg_color=DARK_BG, border_color=AMBER, alpha=0.95, title="CONFIRM ACTION REQUIRED")

        # Determine prompt text
        prompt_str = "Execute pending action?"
        if intent_result and hasattr(intent_result, "message") and intent_result.message:
            prompt_str = intent_result.message
        elif pending_action:
            prompt_str = f"Execute action: {pending_action}?"

        cv2.putText(frame, sanitize_text(prompt_str, 48), (mx + 24, my + 70), FONT_PRIMARY, 0.6, AMBER, 2, cv2.LINE_AA)
        cv2.putText(frame, "Please confirm or cancel using keyboard or controls.", (mx + 24, my + 105), FONT_PRIMARY, 0.45, WHITE, 1, cv2.LINE_AA)

        # Buttons
        # [ CONFIRM (Y/ENTER) ]
        btn1_x, btn1_y, btn1_w, btn1_h = mx + 40, my + 145, 200, 42
        cv2.rectangle(frame, (btn1_x, btn1_y), (btn1_x + btn1_w, btn1_y + btn1_h), GREEN_SUCCESS, -1)
        cv2.putText(frame, "[ CONFIRM (Y/ENTER) ]", (btn1_x + 10, btn1_y + 26), FONT_PRIMARY, 0.45, DARK_BG, 2, cv2.LINE_AA)

        # [ CANCEL (N/ESC) ]
        btn2_x, btn2_y, btn2_w, btn2_h = mx + 260, my + 145, 200, 42
        cv2.rectangle(frame, (btn2_x, btn2_y), (btn2_x + btn2_w, btn2_y + btn2_h), RED_ERROR, -1)
        cv2.putText(frame, "[ CANCEL (N/ESC) ]", (btn2_x + 20, btn2_y + 26), FONT_PRIMARY, 0.45, WHITE, 2, cv2.LINE_AA)
