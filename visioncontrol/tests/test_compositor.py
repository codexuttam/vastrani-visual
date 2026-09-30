"""
tests/test_compositor.py

Unit tests for Phase 8 ARCompositor, image rotation, resizing, and alpha blending.
Requires NO camera hardware.
"""

import pytest
import numpy as np
from ar.compositor import ARCompositor, rotate_overlay, resize_overlay


def test_resize_overlay():
    """Verify resizing BGRA image."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    resized = resize_overlay(img, 50, 50)
    assert resized.shape == (50, 50, 4)


def test_rotate_overlay_preserves_alpha():
    """Verify overlay rotation preserves alpha channel and increases bounds if needed."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # Fully opaque

    rotated_45 = rotate_overlay(img, 45.0)
    assert rotated_45.ndim == 3
    assert rotated_45.shape[2] == 4
    # 45 deg rotation expands bounding box size
    assert rotated_45.shape[0] > 100
    assert rotated_45.shape[1] > 100


def test_alpha_blending_compositor():
    """Verify ARCompositor alpha blending onto camera frame."""
    compositor = ARCompositor()

    # Black camera frame 200x200
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # Red overlay square 50x50 with 100% alpha
    overlay = np.zeros((50, 50, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255  # Red channel in BGR
    overlay[:, :, 3] = 255  # Alpha 100%

    result = compositor.blend_overlay(frame, overlay, center_x=100, center_y=100)

    # Center pixel of frame should now be RED (BGR: 0, 0, 255)
    assert result[100, 100, 2] == 255
    assert result[100, 100, 0] == 0

    # Corner pixel should remain black (0, 0, 0)
    assert result[0, 0, 0] == 0


def test_alpha_blending_out_of_bounds_clip_safe():
    """Verify blending overlay that extends outside frame boundaries does not crash."""
    compositor = ARCompositor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    overlay = np.full((100, 100, 4), 255, dtype=np.uint8)

    # Center near frame corner (-10, -10)
    result = compositor.blend_overlay(frame, overlay, center_x=-10, center_y=-10)
    assert result.shape == (100, 100, 3)
