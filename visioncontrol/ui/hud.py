"""
ui/hud.py

VisionControl HUD renderer.

Renders:
  - Main overlay  : title, camera status, FPS, mode
  - Gesture panel : gesture engine state (Phase 4, debug only)
  - Feature panel : hand diagnostics (Phase 3, debug only)
  - Device panel  : device control state (Phase 5)
  - Device rail   : bottom navigation strip for all registered devices (Phase 5)
"""

import cv2
from typing import Optional, List

from gestures.features import HandFeatureSet
from gestures.types import (
    GestureResult, GestureEvent, GestureType,
    StateMachineState, ControlMode,
)
from devices.models import DeviceState, DeviceType, CommandRecord
from devices.virtual_device import VirtualDevice
from devices.registry import DeviceRegistry

# ─── Color palette (BGR) ──────────────────────────────────────────────────────
CYAN      = (255, 220,   0)
VIOLET    = (211,   0, 148)
WHITE     = (255, 255, 255)
DARK_GRAY = ( 25,  25,  25)
GREEN     = (  0, 200, 100)
RED       = ( 60,  60, 220)
YELLOW    = (  0, 200, 220)
ORANGE    = (  0, 140, 255)
TEAL      = (200, 200,   0)
PURPLE    = (180,  60, 220)
BLUE      = (220, 100,   0)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# Device type icon labels (text fallback — no emoji dependency)
DEVICE_ICONS = {
    DeviceType.LIGHT: "LT",
    DeviceType.FAN:   "FN",
    DeviceType.MUSIC: "MU",
    DeviceType.SERVO: "SV",
}

DEVICE_LABELS = {
    DeviceType.LIGHT: "LIGHT",
    DeviceType.FAN:   "FAN",
    DeviceType.MUSIC: "MUSIC",
    DeviceType.SERVO: "SERVO",
}


class HUD:
    """Renders the VisionControl HUD, gesture panels, and Phase 5 device panels."""

    def render(
        self,
        frame,
        fps: float,
        features: Optional[HandFeatureSet]   = None,
        result:   Optional[GestureResult]    = None,
        event:    Optional[GestureEvent]     = None,
        mode:     ControlMode                = ControlMode.IDLE,
        debug_features: bool                 = False,
        debug_gestures: bool                 = False,
        # ── Phase 5 & 6 ──────────────────────────────────────────────────────
        registry:       Optional[DeviceRegistry] = None,
        pending_action: Optional[str]            = None,
        debug_devices:  bool                     = True,
        command_history: Optional[List[CommandRecord]] = None,
        device_mode:    str                      = "VIRTUAL",
        arduino_connected: bool                  = False,
        arduino_unresponsive: bool               = False,
        # ── Phase 7 ──────────────────────────────────────────────────────────
        face_state:     Optional[object]         = None,
        debug_face:     bool                     = False,
        face_inference_ms: float                 = 0.0,
    ):
        h, w, _ = frame.shape

        # ── Top-left: title ───────────────────────────────────────────────────
        cv2.putText(frame, "VISIONCONTROL", (20, 40), FONT, 0.8, CYAN, 2, cv2.LINE_AA)

        # ── Top-right: camera live indicator ─────────────────────────────────
        cam_text = "o CAMERA LIVE"
        (sw, _), _ = cv2.getTextSize(cam_text, FONT, 0.65, 2)
        cv2.putText(frame, cam_text, (w - sw - 20, 40), FONT, 0.65, VIOLET, 2, cv2.LINE_AA)

        # ── Bottom-left: FPS, mode & Arduino status HUD ───────────────────────
        mode_color = GREEN if mode == ControlMode.CONTROL else WHITE
        fps_str = f"FPS: {int(fps)}"
        if face_inference_ms > 0:
            fps_str += f" | Face: {face_inference_ms:.1f}ms"
        cv2.putText(frame, fps_str, (20, h - 110), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"MODE: {mode.value}", (20, h - 85), FONT, 0.6, mode_color, 2, cv2.LINE_AA)

        # Section 25: Arduino status & Device Mode HUD
        dev_mode_str = f"DEV MODE: {device_mode}"
        cv2.putText(frame, dev_mode_str, (20, h - 60), FONT, 0.55, TEAL, 1, cv2.LINE_AA)

        if device_mode == "VIRTUAL":
            ard_text  = "Arduino   - DISABLED"
            ard_color = (150, 150, 150)
        elif arduino_connected:
            ard_text  = "Arduino   * CONNECTED"
            ard_color = GREEN
        elif arduino_unresponsive:
            ard_text  = "Arduino   ! UNRESPONSIVE"
            ard_color = ORANGE
        else:
            ard_text  = "Arduino   o DISCONNECTED"
            ard_color = RED

        cv2.putText(frame, ard_text, (20, h - 38), FONT, 0.55, ard_color, 1, cv2.LINE_AA)

        # ── Center hint when no hand ──────────────────────────────────────────
        if features is None or not features.valid:
            hint = "Raise your hand to begin"
            (tw, th), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
            cv2.putText(frame, hint, ((w - tw) // 2, (h + th) // 2),
                        FONT, 0.55, CYAN, 1, cv2.LINE_AA)

        # ── Phase 5: Device Rail ──────────────────────────────────────────────
        if registry is not None:
            self._render_device_rail(frame, registry)

        # ── Phase 5/6: Device Control Panel ─────────────────────────────────
        if debug_devices and registry is not None:
            self._render_device_panel(
                frame, registry, pending_action, command_history,
                device_mode, arduino_connected
            )

        # ── Gesture debug panel ───────────────────────────────────────────────
        if debug_gestures and result is not None:
            self._render_gesture_panel(frame, result, event, mode)

        # ── Feature debug panel ───────────────────────────────────────────────
        if debug_features and features is not None and features.valid:
            self._render_feature_panel(frame, features)

        # ── Phase 7: Face tracking debug panel ────────────────────────────────
        if debug_face and face_state is not None:
            self._render_face_panel(frame, face_state, face_inference_ms)

        return frame

    # ─── Phase 5/6: Device Panel ─────────────────────────────────────────────

    def _render_device_panel(
        self,
        frame,
        registry: DeviceRegistry,
        pending_action: Optional[str],
        history: Optional[List[CommandRecord]],
        device_mode: str = "VIRTUAL",
        arduino_connected: bool = False,
    ):
        h, w, _ = frame.shape
        device = registry.current()
        if device is None:
            return

        s = device.state
        px, py = w - 300, 60
        pw, ph  = 285, 235

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 8, py - 4), (px + pw, py + ph), DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Header bar
        cv2.rectangle(frame, (px - 8, py - 4), (px + pw, py + 22), TEAL, -1)
        cv2.putText(frame, "DEVICE CONTROL", (px, py + 14), FONT, 0.48, DARK_GRAY, 1, cv2.LINE_AA)

        lh = 20
        y  = py + 36

        def row(label, value, color=WHITE):
            nonlocal y
            cv2.putText(frame, f"{label:<10}{value}", (px, y),
                        FONT_MONO, 1.0, color, 1, cv2.LINE_AA)
            y += lh

        # Nav arrows around device name
        name_str = s.name.upper()
        if len(name_str) > 18:
            name_str = name_str[:17] + "."
        cv2.putText(frame, f"< {name_str} >", (px, y), FONT_MONO, 1.05, CYAN, 1, cv2.LINE_AA)
        y += lh + 2

        # Separator
        cv2.line(frame, (px, y), (px + pw - 16, y), TEAL, 1)
        y += 6

        row("ID", s.device_id, CYAN)

        power_str   = "ON" if s.power else "OFF"
        power_color = GREEN if s.power else RED

        # Section 26: Virtual vs Hardware status display
        row("VIRT STATE", f"{power_str} ({s.level}{s.level_unit})", power_color)

        if device_mode == "HARDWARE":
            if arduino_connected:
                hw_str = f"{power_str} ({s.level}{s.level_unit})"
                hw_color = GREEN if s.power else RED
            else:
                hw_str = "DISCONNECTED"
                hw_color = RED
        else:
            hw_str = "VIRTUAL ONLY"
            hw_color = (160, 160, 160)

        row("HW STATE", hw_str, hw_color)
        row("TYPE", s.device_type.value, PURPLE)

        # Pending action
        if pending_action:
            pend_color = ORANGE
            pend_str   = pending_action
        else:
            pend_color = WHITE
            pend_str   = "none"
        row("PENDING", pend_str, pend_color)

        # [ SELECTED ] badge
        y += 2
        badge = "[ SELECTED ]"
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.45, 1)
        cv2.putText(frame, badge, (px + (pw - bw) // 2 - 8, y), FONT, 0.45, GREEN, 1, cv2.LINE_AA)

        # ── Command history (last 3) ───────────────────────────────────────────
        if history:
            hy = py + ph + 20
            cv2.putText(frame, "RECENT COMMANDS", (px, hy), FONT, 0.42, TEAL, 1, cv2.LINE_AA)
            hy += 18
            for rec in list(history)[-3:]:
                val_str = f" {rec.value}" if rec.value is not None else ""
                
                if rec.hardware_sent:
                    ack_str = "ACK:OK" if rec.hardware_ack else "ACK:ERR"
                    line = f"{rec.formatted_time()} {rec.device_id} {rec.action}{val_str} [{ack_str}]"
                else:
                    line = f"{rec.formatted_time()} {rec.device_id} {rec.action}{val_str}"

                ok_color = GREEN if rec.virtual_success else RED
                cv2.putText(frame, line[:42], (px, hy), FONT_MONO, 0.85, ok_color, 1, cv2.LINE_AA)
                hy += 16

    # ─── Phase 5: Device Rail ────────────────────────────────────────────────

    def _render_device_rail(self, frame, registry: DeviceRegistry):
        h, w, _ = frame.shape
        devices   = registry.list_devices()
        n         = len(devices)
        if n == 0:
            return

        rail_h    = 36
        rail_y    = h - rail_h
        cell_w    = w // n
        sel_idx   = registry.selected_index

        # Rail background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, rail_y), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        for i, device in enumerate(devices):
            s       = device.state
            x_start = i * cell_w
            x_end   = x_start + cell_w
            cx      = x_start + cell_w // 2

            is_sel = (i == sel_idx)

            # Highlight selected cell
            if is_sel:
                cv2.rectangle(frame, (x_start, rail_y), (x_end, h), TEAL, -1)
                label_color = DARK_GRAY
                icon_color  = DARK_GRAY
            else:
                label_color = WHITE
                icon_color  = CYAN

            # Icon (device type abbreviation)
            icon = DEVICE_ICONS.get(s.device_type, "??")
            (iw, _), _ = cv2.getTextSize(icon, FONT, 0.5, 2)
            cv2.putText(frame, icon, (cx - iw // 2, rail_y + 15), FONT, 0.5, icon_color, 2, cv2.LINE_AA)

            # Label
            label = DEVICE_LABELS.get(s.device_type, s.device_type.value)
            (lw, _), _ = cv2.getTextSize(label, FONT_MONO, 0.85, 1)
            cv2.putText(frame, label, (cx - lw // 2, rail_y + 30), FONT_MONO, 0.85, label_color, 1, cv2.LINE_AA)

            # Power dot
            dot_color = GREEN if s.power else (80, 80, 80)
            cv2.circle(frame, (x_start + 8, rail_y + 8), 4, dot_color, -1)

            # Cell separator
            if i > 0:
                cv2.line(frame, (x_start, rail_y), (x_start, h), (60, 60, 60), 1)

    # ─── Gesture panel (top-left, below title) ────────────────────────────────

    def _render_gesture_panel(
        self, frame, result: GestureResult,
        event: Optional[GestureEvent], mode: ControlMode
    ):
        h, w, _ = frame.shape
        px, py   = 20, 60
        pw, ph   = 280, 180

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 6, py - 4), (px + pw, py + ph), DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lh = 22
        y  = py + 18

        def row(label, value, color=WHITE):
            nonlocal y
            cv2.putText(frame, f"{label:<12}{value}", (px, y),
                        FONT_MONO, 1.05, color, 1, cv2.LINE_AA)
            y += lh

        cv2.putText(frame, "GESTURE ENGINE", (px, y - 2), FONT, 0.52, CYAN, 1, cv2.LINE_AA)
        y += lh - 4

        gest_str   = result.gesture.value
        gest_color = GREEN if result.active else YELLOW
        row("Current:", gest_str, gest_color)

        event_str = event.gesture.value if event else "NONE"
        row("Event:", event_str, ORANGE if event else WHITE)

        action_str = event.action.value if event and event.action else "NONE"
        row("Action:", action_str, ORANGE if event else WHITE)

        row("Confidence:", f"{result.confidence:.2f}")

        m_color = GREEN if mode == ControlMode.CONTROL else WHITE
        row("Mode:", mode.value, m_color)

        state_colors = {
            StateMachineState.NO_GESTURE: WHITE,
            StateMachineState.CANDIDATE:  YELLOW,
            StateMachineState.ACTIVE:     GREEN,
            StateMachineState.COOLDOWN:   ORANGE,
        }
        row("State:", result.state.value, state_colors.get(result.state, WHITE))

    # ─── Feature panel (top-right) ────────────────────────────────────────────

    def _render_feature_panel(self, frame, fs: HandFeatureSet):
        h, w, _ = frame.shape
        pw, ph   = 260, 265
        margin   = 16
        px       = w - pw - margin
        py       = 290   # moved down to make room for device panel

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 8, py - 4), (px + pw, py + ph), DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "HAND FEATURES", (px, py + 16), FONT, 0.52, CYAN, 1, cv2.LINE_AA)

        lh = 22
        y  = py + 16 + lh

        def line(label, value, color=WHITE):
            nonlocal y
            cv2.putText(frame, f"{label:<12}{value}", (px, y),
                        FONT_MONO, 1.0, color, 1, cv2.LINE_AA)
            y += lh

        line("Palm:", f"{fs.palm_x:.2f}, {fs.palm_y:.2f}")
        dir_color = YELLOW if fs.direction != "STATIONARY" else WHITE
        line("Movement:", fs.direction, dir_color)
        line("Speed:", f"{fs.speed:.4f}")
        line("Angle:", f"{fs.hand_angle:.1f}d")
        pinch_color = GREEN if fs.thumb_index_distance < 0.25 else WHITE
        line("Pinch:", f"{fs.thumb_index_distance:.2f}", pinch_color)

        y += 4
        cv2.line(frame, (px, y), (px + pw - 16, y), CYAN, 1)
        y += 10

        cv2.putText(frame, "Fingers", (px, y), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
        y += lh

        for name, ext in [
            ("THUMB",  fs.thumb_extended),
            ("INDEX",  fs.index_extended),
            ("MIDDLE", fs.middle_extended),
            ("RING",   fs.ring_extended),
            ("PINKY",  fs.pinky_extended),
        ]:
            label = "OPEN  " if ext else "CLOSED"
            color = GREEN if ext else RED
            cv2.putText(frame, f"{name:<8}{label}", (px, y),
                        FONT_MONO, 1.0, color, 1, cv2.LINE_AA)
            y += lh

    # ─── Phase 7: Face Tracking Panel (Section 18) ────────────────────────────

    def _render_face_panel(self, frame, fs, inference_ms: float = 0.0):
        h, w, _ = frame.shape
        px, py   = 20, 250
        pw, ph   = 280, 160

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 6, py - 4), (px + pw, py + ph), DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lh = 20
        y  = py + 16

        def row(label, value, color=WHITE):
            nonlocal y
            cv2.putText(frame, f"{label:<10}{value}", (px, y),
                        FONT_MONO, 1.0, color, 1, cv2.LINE_AA)
            y += lh

        cv2.putText(frame, "FACE TRACKING", (px, y - 2), FONT, 0.52, CYAN, 1, cv2.LINE_AA)
        y += lh - 2

        status_str   = "TRACKING" if fs.detected else "NO FACE"
        status_color = GREEN if fs.detected else RED
        row("Status:", status_str, status_color)

        if fs.detected:
            row("Center:", f"{fs.center_x:.2f}, {fs.center_y:.2f}", WHITE)
            row("Scale:", f"{fs.scale:.2f}", YELLOW)
            row("Yaw:", f"{fs.yaw:+.1f}d", TEAL)
            row("Pitch:", f"{fs.pitch:+.1f}d", TEAL)
            row("Roll:", f"{fs.roll:+.1f}d", TEAL)
        else:
            row("Center:", "—", (120, 120, 120))
            row("Scale:", "—", (120, 120, 120))
            row("Yaw/Pt/Rl:", "—", (120, 120, 120))

        if inference_ms > 0:
            row("Infer MS:", f"{inference_ms:.1f}ms", PURPLE)
