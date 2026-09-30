"""
tests/test_ar_renderer.py

Unit tests for Phase 8 ARRenderer rendering pipeline.
Requires NO camera hardware.
"""

import pytest
import numpy as np
from ar.renderer import ARRenderer
from ar.effects import ARState
from vision.face_features import FaceAnchor


def test_ar_renderer_invalid_anchor_returns_frame_unmodified():
    """Verify renderer does not crash and leaves frame untouched when face anchor is invalid."""
    renderer = ARRenderer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_anchor = FaceAnchor(x=0.5, y=0.5, scale=0.5, rotation=0.0, valid=False)

    out = renderer.render(frame, invalid_anchor, ARState(enabled=True))
    assert np.array_equal(out, frame)


def test_ar_renderer_disabled_state_returns_frame_unmodified():
    """Verify renderer leaves frame untouched when ARState.enabled is False."""
    renderer = ARRenderer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    valid_anchor = FaceAnchor(x=0.5, y=0.5, scale=0.5, rotation=0.0, valid=True)

    out = renderer.render(frame, valid_anchor, ARState(enabled=False))
    assert np.array_equal(out, frame)


def test_ar_renderer_valid_anchor_composites_overlay():
    """Verify valid face anchor produces modified frame with emoji overlay."""
    renderer = ARRenderer(smoothing_alpha=1.0)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    valid_anchor = FaceAnchor(x=0.5, y=0.5, scale=0.4, rotation=0.0, valid=True)

    out = renderer.render(frame, valid_anchor, ARState(enabled=True, current_effect="happy"))

    # Frame should no longer be pure zeros since happy emoji was rendered near center
    assert not np.array_equal(out, np.zeros((300, 300, 3), dtype=np.uint8))
