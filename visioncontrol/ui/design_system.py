"""
ui/design_system.py

Centralized design system for VisionControl UI / HUD.
Provides palettes, typography, theme tokens, glassmorphism overlays,
status badge renderers, glowing borders, and animation timing functions.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional
from ui.contracts import SystemStatus, NotificationCategory


# ─── Color Palette (BGR for OpenCV) ───────────────────────────────────────────
DARK_BG        = ( 18,  20,  24)   # Deep dark blue/gray backdrop
PANEL_BG       = ( 32,  35,  44)   # Glass panel overlay background
PANEL_BORDER   = ( 70,  78,  95)   # Subtle panel border
PANEL_BORDER_HI= (130, 140, 165)   # Highlight border

CYAN           = (255, 215,   0)   # #00D7FF Primary accent
VIOLET         = (211,   0, 148)   # #9400D3 Secondary accent
TEAL           = (200, 200,   0)   # #00C8C8 Hardware / serial accent
AMBER          = (  0, 180, 255)   # #FFB400 Warning accent
MAGENTA        = (220,  60, 180)   # AI intent accent

GREEN_SUCCESS  = ( 80, 220, 100)   # Active / Success
RED_ERROR      = ( 70,  70, 235)   # Error / Offline
ORANGE_WARN    = (  0, 140, 255)   # Warning / Unresponsive
YELLOW_DEGRADED= (  0, 220, 240)   # Degraded state
GRAY_DISABLED  = (120, 125, 135)   # Disabled / Offline

WHITE          = (255, 255, 255)
TEXT_PRIMARY   = (240, 242, 245)
TEXT_SECONDARY = (175, 182, 195)
TEXT_MUTED     = (115, 122, 135)

FONT_PRIMARY   = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO      = cv2.FONT_HERSHEY_PLAIN


def sanitize_text(text: str, max_length: int = 60) -> str:
    """Sanitize and truncate text strings for safe UI display (prevents secret leaks)."""
    if not text:
        return ""
    if "sk-" in text:
        idx = text.find("sk-")
        text = text[:idx + 3] + "***"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: Tuple[int, int, int] = PANEL_BG,
    border_color: Tuple[int, int, int] = PANEL_BORDER,
    alpha: float = 0.75,
    title: Optional[str] = None,
    corner_accent: bool = True,
) -> np.ndarray:
    """
    Draws a translucent glassmorphism card panel with subtle border and optional title.
    """
    fh, fw, _ = frame.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return frame

    # Extract ROI & apply alpha blend
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, bg_color, dtype=np.uint8)
    blended = cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0)
    frame[y1:y2, x1:x2] = blended

    # Draw outer border
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    # Optional top title bar
    if title:
        cv2.rectangle(frame, (x1, y1), (x2, min(y2, y1 + 24)), border_color, -1)
        cv2.putText(
            frame,
            title.upper(),
            (x1 + 8, y1 + 16),
            FONT_PRIMARY,
            0.45,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

    # Subtle tech corner accents
    if corner_accent and (x2 - x1) > 20 and (y2 - y1) > 20:
        c_len = 6
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), CYAN, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), CYAN, 2)
        cv2.line(frame, (x2 - 1, y2 - 1), (x2 - 1 - c_len, y2 - 1), CYAN, 2)
        cv2.line(frame, (x2 - 1, y2 - 1), (x2 - 1, y2 - 1 - c_len), CYAN, 2)

    return frame


def get_status_color_and_symbol(status: SystemStatus) -> Tuple[Tuple[int, int, int], str]:
    """Returns color and bullet symbol for system status state."""
    if status == SystemStatus.ONLINE:
        return GREEN_SUCCESS, "[●]"
    elif status == SystemStatus.INITIALIZING:
        return CYAN, "[○]"
    elif status == SystemStatus.CAMERA_OFFLINE:
        return RED_ERROR, "[✕]"
    elif status == SystemStatus.AI_OFFLINE:
        return ORANGE_WARN, "[!]"
    elif status == SystemStatus.ARDUINO_DISCONNECTED:
        return ORANGE_WARN, "[!]"
    elif status == SystemStatus.DEGRADED:
        return YELLOW_DEGRADED, "[▲]"
    else:
        return RED_ERROR, "[✕]"


def draw_status_badge(
    frame: np.ndarray,
    x: int,
    y: int,
    status: SystemStatus,
    custom_label: Optional[str] = None,
) -> int:
    """Draws persistent system status badge (symbol + readable text). Returns badge width."""
    color, symbol = get_status_color_and_symbol(status)
    label = custom_label or status.value
    text = f"{symbol} SYSTEM: {label}"

    (tw, th), _ = cv2.getTextSize(text, FONT_PRIMARY, 0.5, 1)
    badge_w = tw + 16
    badge_h = 24

    draw_glass_panel(frame, x, y, badge_w, badge_h, bg_color=DARK_BG, border_color=color, alpha=0.85)
    cv2.putText(frame, text, (x + 8, y + 16), FONT_PRIMARY, 0.5, color, 1, cv2.LINE_AA)
    return badge_w


def draw_badge(
    frame: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = WHITE,
    scale: float = 0.45,
) -> int:
    """Draws a compact badge pill. Returns badge width."""
    (tw, th), _ = cv2.getTextSize(text, FONT_PRIMARY, scale, 1)
    pw, ph = tw + 12, th + 8
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), color, -1)
    cv2.putText(frame, text, (x + 6, y + ph - 4), FONT_PRIMARY, scale, text_color, 1, cv2.LINE_AA)
    return pw


def draw_progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    progress: float,
    color: Tuple[int, int, int] = CYAN,
    bg_color: Tuple[int, int, int] = DARK_BG,
):
    """Draws a sleek progress bar."""
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL_BORDER, 1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)


def get_pulse_alpha(cycle_sec: float = 1.5, reduced_motion: bool = False) -> float:
    """Computes a smooth pulsing alpha factor [0.5, 1.0]."""
    if reduced_motion:
        return 1.0
    t = time.time()
    val = (np.sin(t * (2.0 * np.pi / cycle_sec)) + 1.0) / 2.0
    return 0.5 + 0.5 * val
