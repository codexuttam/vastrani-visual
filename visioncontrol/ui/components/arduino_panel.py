"""
ui/components/arduino_panel.py

Arduino Hardware Connection Telemetry Panel component.
Displays connection status, configured USB port, baud rate, telemetry latency,
last command transmitted, and last ACK response.
"""

import cv2
import numpy as np
from typing import Optional
from ui.design_system import (
    TEAL, GREEN_SUCCESS, RED_ERROR, ORANGE_WARN, GRAY_DISABLED, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel,
)


class ArduinoPanelComponent:
    """Renders Arduino connection telemetry panel."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        connected: bool = False,
        unresponsive: bool = False,
        port: str = "",
        baud_rate: int = 115200,
        device_mode: str = "VIRTUAL",
        last_command: str = "N/A",
        last_response: str = "N/A",
        latency_ms: float = 0.0,
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title="ARDUINO HARDWARE")

        curr_y = y + 40
        if device_mode == "VIRTUAL":
            cv2.putText(frame, "● VIRTUAL MODE", (x + 12, curr_y), FONT_PRIMARY, 0.55, GRAY_DISABLED, 1, cv2.LINE_AA)
            cv2.putText(frame, "Serial transport disabled", (x + 12, curr_y + 24), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        if connected:
            cv2.putText(frame, "● CONNECTED", (x + 12, curr_y), FONT_PRIMARY, 0.55, GREEN_SUCCESS, 1, cv2.LINE_AA)
        elif unresponsive:
            cv2.putText(frame, "! UNRESPONSIVE", (x + 12, curr_y), FONT_PRIMARY, 0.55, ORANGE_WARN, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "✕ DISCONNECTED", (x + 12, curr_y), FONT_PRIMARY, 0.55, RED_ERROR, 1, cv2.LINE_AA)

        curr_y += 24
        port_label = port if port else "AUTO DETECT"
        cv2.putText(frame, f"Port: {port_label}", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEAL, 1, cv2.LINE_AA)
        curr_y += 20

        if latency_ms > 0:
            cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (x + 12, curr_y), FONT_MONO, 0.7, TEXT_SECONDARY, 1, cv2.LINE_AA)
            curr_y += 20

        cv2.putText(frame, f"Last Cmd: {last_command}", (x + 12, curr_y), FONT_PRIMARY, 0.42, TEXT_SECONDARY, 1, cv2.LINE_AA)
        curr_y += 18
        cv2.putText(frame, f"Last Resp: {last_response}", (x + 12, curr_y), FONT_PRIMARY, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)
