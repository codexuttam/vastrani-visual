"""
ui/components/intent_panel.py

AI Intent & Natural Language Interpretation Panel.
Renders raw user prompt, parsed intent details, confidence %, and execution status.
"""

import cv2
import numpy as np
from typing import Optional
from ui.design_system import (
    CYAN, MAGENTA, GREEN_SUCCESS, RED_ERROR, AMBER, WHITE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, draw_badge, sanitize_text,
)


class IntentPanelComponent:
    """Renders AI intent state and parsed parameters."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        intent_result: Optional[object] = None,
        intent_pending: bool = False,
        provider_name: str = "auto",
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title=f"AI INTENT ({provider_name.upper()})")

        curr_y = y + 40
        if intent_result is None:
            cv2.putText(frame, 'Prompt: "Waiting for user input..."', (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, "Status: IDLE", (x + 12, curr_y + 24), FONT_PRIMARY, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            return

        # Fetch properties safely
        raw_text = getattr(intent_result, "raw_input", "") or getattr(intent_result, "message", "")
        status = getattr(intent_result, "status", "EXECUTED")
        cmd = getattr(intent_result, "command", None)

        # 1. Raw user prompt (distinguished in Magenta/Cyan)
        cv2.putText(frame, f'"{sanitize_text(raw_text, 36)}"', (x + 12, curr_y), FONT_PRIMARY, 0.48, MAGENTA, 1, cv2.LINE_AA)
        curr_y += 24

        # 2. Parsed action & target
        if cmd and hasattr(cmd, "intent"):
            intent_name = cmd.intent.value if hasattr(cmd.intent, "value") else str(cmd.intent)
            cv2.putText(frame, f"Intent: {intent_name}", (x + 12, curr_y), FONT_PRIMARY, 0.45, CYAN, 1, cv2.LINE_AA)
            curr_y += 20

            target = getattr(cmd, "target", None) or (cmd.entities.get("target") or cmd.entities.get("device") if hasattr(cmd, "entities") else None)
            action_val = cmd.action.value if hasattr(cmd.action, 'value') else cmd.action
            action_str = f"Action: {action_val}"
            if target:
                action_str += f" -> {target}"
            cv2.putText(frame, sanitize_text(action_str, 38), (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_PRIMARY, 1, cv2.LINE_AA)
            curr_y += 20

            conf = int((cmd.confidence if hasattr(cmd, "confidence") else 1.0) * 100)
            cv2.putText(frame, f"Confidence: {conf}%", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            curr_y += 22
        else:
            cv2.putText(frame, f"Result: {sanitize_text(str(status), 30)}", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            curr_y += 24

        # 3. Status badge
        if intent_pending or status == "PENDING_CONFIRMATION":
            draw_badge(frame, x + 12, curr_y, "CONFIRMATION REQ", color=AMBER, text_color=DARK_BG, scale=0.42)
        elif status == "EXECUTED" or status == "SUCCESS":
            draw_badge(frame, x + 12, curr_y, "✓ EXECUTED", color=GREEN_SUCCESS, text_color=DARK_BG, scale=0.42)
        elif status == "REJECTED" or status == "BLOCKED" or status == "FAILED":
            draw_badge(frame, x + 12, curr_y, "✕ REJECTED", color=RED_ERROR, text_color=WHITE, scale=0.42)
        else:
            draw_badge(frame, x + 12, curr_y, str(status).upper(), color=PANEL_BORDER, text_color=WHITE, scale=0.42)
