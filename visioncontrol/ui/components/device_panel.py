"""
ui/components/device_panel.py

Connected Devices Status & Device Rail component.
Dynamically queries the DeviceRegistry to display connected devices,
their states (ON/OFF), levels, and bottom navigation rail.
"""

import cv2
import numpy as np
from typing import Optional
from devices.registry import DeviceRegistry
from devices.models import DeviceType
from ui.design_system import (
    CYAN, GREEN_SUCCESS, RED_ERROR, WHITE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    DARK_BG, PANEL_BORDER, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, draw_badge,
)

DEVICE_LABELS = {
    DeviceType.LIGHT: "LIGHT",
    DeviceType.FAN:   "FAN",
    DeviceType.MUSIC: "MUSIC",
    DeviceType.SERVO: "SERVO",
}


class DevicePanelComponent:
    """Renders connected devices panel and bottom device rail."""

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        registry: Optional[DeviceRegistry] = None,
        pending_action: Optional[str] = None,
        device_mode: str = "VIRTUAL",
    ):
        draw_glass_panel(frame, x, y, w, h, bg_color=DARK_BG, border_color=PANEL_BORDER, title=f"CONNECTED DEVICES ({device_mode})")

        curr_y = y + 40
        if registry is None:
            cv2.putText(frame, "No devices registered", (x + 12, curr_y), FONT_PRIMARY, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        devices = registry.list_devices() if hasattr(registry, "list_devices") else []
        selected = registry.current() if hasattr(registry, "current") else None

        for dev in devices:
            is_sel = (selected and selected.state.device_id == dev.state.device_id)
            s = dev.state
            status_text = "● ON" if s.power else "○ OFF"
            status_color = GREEN_SUCCESS if s.power else RED_ERROR

            name_color = CYAN if is_sel else TEXT_PRIMARY
            sel_prefix = "► " if is_sel else "  "

            dev_type = getattr(dev, "type", s.device_type)
            label = DEVICE_LABELS.get(dev_type, s.device_id.upper())
            line = f"{sel_prefix}{label:<8}"
            cv2.putText(frame, line, (x + 12, curr_y), FONT_PRIMARY, 0.48, name_color, 1, cv2.LINE_AA)

            cv2.putText(frame, status_text, (x + 140, curr_y), FONT_PRIMARY, 0.48, status_color, 1, cv2.LINE_AA)

            # Show level if available
            level_str = f"{s.level}{s.level_unit}" if hasattr(s, "level_unit") else f"{s.level}"
            if level_str:
                cv2.putText(frame, level_str, (x + 215, curr_y), FONT_MONO, 0.7, TEXT_SECONDARY, 1, cv2.LINE_AA)

            curr_y += 24

        if pending_action:
            cv2.putText(frame, f"Pending: {pending_action}", (x + 12, y + h - 12), FONT_PRIMARY, 0.42, CYAN, 1, cv2.LINE_AA)

    def render_rail(self, frame: np.ndarray, registry: Optional[DeviceRegistry]):
        """Renders bottom device rail strip."""
        if registry is None:
            return
        h, w, _ = frame.shape
        devices = registry.list_devices() if hasattr(registry, "list_devices") else []
        selected = registry.current() if hasattr(registry, "current") else None
        if not devices:
            return

        rail_w = len(devices) * 110 + 20
        rail_x = (w - rail_w) // 2
        rail_y = h - 45

        draw_glass_panel(frame, rail_x, rail_y, rail_w, 36, bg_color=DARK_BG, border_color=PANEL_BORDER, alpha=0.85)

        dx = rail_x + 10
        for dev in devices:
            s = dev.state
            is_sel = (selected and selected.state.device_id == s.device_id)
            dev_type = getattr(dev, "type", s.device_type)
            label = DEVICE_LABELS.get(dev_type, s.device_id[:4].upper())
            icon_status = "●" if s.power else "○"

            bg = CYAN if is_sel else DARK_BG
            fg = DARK_BG if is_sel else TEXT_PRIMARY

            draw_badge(frame, dx, rail_y + 6, f"{icon_status} {label}", color=bg, text_color=fg, scale=0.42)
            dx += 110
