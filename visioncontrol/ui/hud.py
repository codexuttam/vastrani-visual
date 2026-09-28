"""
ui/hud.py

VisionControl HUD renderer.
Displays the main overlay (FPS, status) and the optional debug feature panel.
"""

import cv2
from gestures.features import HandFeatureSet


# ─── Color palette (BGR) ───────────────────────────────────────────────────────
CYAN        = (255, 220, 0)
VIOLET      = (211, 0, 148)
WHITE       = (255, 255, 255)
DARK_GRAY   = (30, 30, 30)
GREEN       = (0, 200, 100)
RED         = (60, 60, 220)
YELLOW      = (0, 200, 220)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN


class HUD:
    """Renders the VisionControl HUD and optional debug diagnostics panel."""

    def render(self, frame, fps: float, features: HandFeatureSet = None, debug: bool = False):
        """
        Draw HUD elements onto the frame.

        Args:
            frame:    BGR numpy array from the camera.
            fps:      Current FPS value.
            features: HandFeatureSet from the feature extractor (may be None).
            debug:    If True, renders the feature diagnostics panel.
        """
        h, w, _ = frame.shape

        # ── Top-left: VISIONCONTROL ───────────────────────────────────────────
        cv2.putText(frame, "VISIONCONTROL", (20, 40), FONT, 0.8, CYAN, 2, cv2.LINE_AA)

        # ── Top-right: camera status ──────────────────────────────────────────
        status_text = "o CAMERA LIVE"
        (sw, _), _ = cv2.getTextSize(status_text, FONT, 0.65, 2)
        cv2.putText(frame, status_text, (w - sw - 20, 40), FONT, 0.65, VIOLET, 2, cv2.LINE_AA)

        # ── Bottom-left: FPS ─────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 20), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

        # ── Center hint (only when no hand present) ───────────────────────────
        if features is None or not features.valid:
            hint = "Raise your hand to begin"
            (tw, th), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
            cv2.putText(frame, hint, ((w - tw) // 2, (h + th) // 2),
                        FONT, 0.55, CYAN, 1, cv2.LINE_AA)

        # ── Debug panel ───────────────────────────────────────────────────────
        if debug and features is not None and features.valid:
            self._render_debug_panel(frame, features)

        return frame

    # ─── Private ──────────────────────────────────────────────────────────────

    def _render_debug_panel(self, frame, fs: HandFeatureSet):
        """Draw the hand-feature diagnostics panel in the top-right area."""
        h, w, _ = frame.shape

        panel_w = 260
        panel_h = 260
        margin   = 16
        px       = w - panel_w - margin
        py       = 60

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 8, py - 4), (px + panel_w, py + panel_h), DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Title
        cv2.putText(frame, "HAND FEATURES", (px, py + 16), FONT, 0.55, CYAN, 1, cv2.LINE_AA)

        lh = 22  # line height
        y  = py + 16 + lh

        def line(label, value, color=WHITE):
            cv2.putText(frame, f"{label:<12}{value}", (px, y), FONT_MONO, 1.0, color, 1, cv2.LINE_AA)

        # Palm position
        line("Palm:", f"{fs.palm_x:.2f}, {fs.palm_y:.2f}")
        y += lh

        # Movement
        dir_color = YELLOW if fs.direction != "STATIONARY" else WHITE
        line("Movement:", fs.direction, dir_color)
        y += lh
        line("Speed:", f"{fs.speed:.4f}")
        y += lh

        # Hand angle
        line("Angle:", f"{fs.hand_angle:.1f} deg")
        y += lh

        # Pinch distance
        pinch_color = GREEN if fs.thumb_index_distance < 0.3 else WHITE
        line("Pinch:", f"{fs.thumb_index_distance:.2f}", pinch_color)
        y += lh + 4

        # Separator
        cv2.line(frame, (px, y), (px + panel_w - 16, y), CYAN, 1)
        y += 10

        # Finger states header
        cv2.putText(frame, "Fingers", (px, y), FONT, 0.48, CYAN, 1, cv2.LINE_AA)
        y += lh

        finger_states = [
            ("THUMB",  fs.thumb_extended),
            ("INDEX",  fs.index_extended),
            ("MIDDLE", fs.middle_extended),
            ("RING",   fs.ring_extended),
            ("PINKY",  fs.pinky_extended),
        ]

        for name, extended in finger_states:
            label  = "OPEN  " if extended else "CLOSED"
            color  = GREEN if extended else RED
            cv2.putText(frame, f"{name:<8}{label}", (px, y), FONT_MONO, 1.0, color, 1, cv2.LINE_AA)
            y += lh
