"""
ui/hud.py

VisionControl HUD renderer.

Renders:
  - Main overlay  : title, camera status, FPS
  - Feature panel : hand diagnostics (Phase 3, debug only)
  - Gesture panel : gesture engine state (Phase 4, debug only)
"""

import cv2
from typing import Optional

from gestures.features import HandFeatureSet
from gestures.types import (
    GestureResult, GestureEvent, GestureType,
    StateMachineState, ControlMode,
)

# ─── Color palette (BGR) ──────────────────────────────────────────────────────
CYAN      = (255, 220,   0)
VIOLET    = (211,   0, 148)
WHITE     = (255, 255, 255)
DARK_GRAY = ( 25,  25,  25)
GREEN     = (  0, 200, 100)
RED       = ( 60,  60, 220)
YELLOW    = (  0, 200, 220)
ORANGE    = (  0, 140, 255)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


class HUD:
    """Renders the VisionControl HUD and optional debug panels."""

    def render(
        self,
        frame,
        fps: float,
        features: Optional[HandFeatureSet] = None,
        result:   Optional[GestureResult]  = None,
        event:    Optional[GestureEvent]   = None,
        mode:     ControlMode              = ControlMode.IDLE,
        debug_features: bool               = False,
        debug_gestures: bool               = False,
    ):
        h, w, _ = frame.shape

        # ── Top-left: title ───────────────────────────────────────────────────
        cv2.putText(frame, "VISIONCONTROL", (20, 40), FONT, 0.8, CYAN, 2, cv2.LINE_AA)

        # ── Top-right: camera live indicator ─────────────────────────────────
        cam_text = "o CAMERA LIVE"
        (sw, _), _ = cv2.getTextSize(cam_text, FONT, 0.65, 2)
        cv2.putText(frame, cam_text, (w - sw - 20, 40), FONT, 0.65, VIOLET, 2, cv2.LINE_AA)

        # ── Bottom-left: FPS + mode ───────────────────────────────────────────
        mode_color = GREEN if mode == ControlMode.CONTROL else WHITE
        cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 45), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"MODE: {mode.value}", (20, h - 20), FONT, 0.65, mode_color, 2, cv2.LINE_AA)

        # ── Center hint when no hand ──────────────────────────────────────────
        if features is None or not features.valid:
            hint = "Raise your hand to begin"
            (tw, th), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
            cv2.putText(frame, hint, ((w - tw) // 2, (h + th) // 2),
                        FONT, 0.55, CYAN, 1, cv2.LINE_AA)

        # ── Gesture debug panel ───────────────────────────────────────────────
        if debug_gestures and result is not None:
            self._render_gesture_panel(frame, result, event, mode)

        # ── Feature debug panel ───────────────────────────────────────────────
        if debug_features and features is not None and features.valid:
            self._render_feature_panel(frame, features)

        return frame

    # ─── Gesture panel (top-left, below title) ────────────────────────────────

    def _render_gesture_panel(
        self, frame, result: GestureResult,
        event: Optional[GestureEvent], mode: ControlMode
    ):
        h, w, _ = frame.shape
        px, py   = 20, 60
        pw, ph   = 280, 180

        # Semi-transparent background
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

        # Header
        cv2.putText(frame, "GESTURE ENGINE", (px, y - 2), FONT, 0.52, CYAN, 1, cv2.LINE_AA)
        y += lh - 4

        # Current gesture
        gest_str = result.gesture.value
        gest_color = GREEN if result.active else YELLOW
        row("Current:", gest_str, gest_color)

        # Event
        event_str = event.gesture.value if event else "NONE"
        row("Event:", event_str, ORANGE if event else WHITE)

        # Action
        action_str = event.action.value if event and event.action else "NONE"
        row("Action:", action_str, ORANGE if event else WHITE)

        # Confidence
        conf_val = f"{result.confidence:.2f}"
        row("Confidence:", conf_val)

        # Mode
        m_color = GREEN if mode == ControlMode.CONTROL else WHITE
        row("Mode:", mode.value, m_color)

        # State machine state
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
        py       = 60

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
