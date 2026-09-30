"""
ar/renderer.py

Phase 8: Real-time AR Emoji & Face Effect Renderer.

Responsibilities:
    - Consume camera frame, FaceAnchor, and ARState
    - Smooth anchor tracking for fluid motion
    - Retrieve cached RGBA emoji assets
    - Apply scaling and head-roll rotation
    - Composite alpha-blended emoji overlay onto camera frame
    - Graceful face-loss handling (hides effect when face.valid is False)
"""

import cv2
import numpy as np
from typing import Optional

from vision.face_features import FaceAnchor
from vision.smoothing import PointSmoother, ExponentialSmoother, AngleSmoother
from ar.assets import AssetManager
from ar.effects import ARState
from ar.compositor import ARCompositor, rotate_overlay, resize_overlay


class ARRenderer:
    """
    Real-time AR Renderer for placing dynamic emoji overlays onto tracked faces.
    """

    def __init__(
        self,
        asset_manager: Optional[AssetManager] = None,
        offset_x: float = 0.0,
        offset_y: float = -0.05,
        scale_multiplier: float = 1.5,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
        smoothing_alpha: float = 0.4,
    ):
        self.asset_manager: AssetManager = asset_manager or AssetManager()
        self.compositor = ARCompositor()

        self.offset_x: float = offset_x
        self.offset_y: float = offset_y
        self.scale_multiplier: float = scale_multiplier
        self.min_scale: float = min_scale
        self.max_scale: float = max_scale

        # AR Motion Smoothers
        self._pos_smoother   = PointSmoother(alpha=smoothing_alpha)
        self._scale_smoother = ExponentialSmoother(alpha=smoothing_alpha)
        self._rot_smoother   = AngleSmoother(alpha=smoothing_alpha)

    def render(
        self,
        frame: np.ndarray,
        face_anchor: Optional[FaceAnchor],
        ar_state: ARState,
    ) -> np.ndarray:
        """
        Renders the active AR effect onto the camera frame.
        If face_anchor is invalid or AR is disabled, returns frame unmodified.
        """
        if frame is None:
            return frame

        # Section 18: Face Loss handling
        if not ar_state.enabled or not ar_state.visible or face_anchor is None or not face_anchor.valid:
            self._pos_smoother.reset()
            self._scale_smoother.reset()
            self._rot_smoother.reset()
            return frame

        # Fetch cached asset
        overlay_asset = self.asset_manager.get(ar_state.current_effect)
        if overlay_asset is None:
            return frame

        fh, fw = frame.shape[:2]

        # 1. Smooth anchor parameters
        sx, sy = self._pos_smoother.update(face_anchor.x, face_anchor.y)
        sscale = self._scale_smoother.update(face_anchor.scale)
        srot   = self._rot_smoother.update(face_anchor.rotation)

        # 2. Section 9: Position in pixel coordinates
        target_x_px = int((sx + self.offset_x) * fw)
        target_y_px = int((sy + self.offset_y) * fh)

        # 3. Section 10: Scale calculation
        raw_target_size = sscale * self.scale_multiplier * fw
        target_size_px = int(np.clip(raw_target_size, self.min_scale * fw, self.max_scale * fw))

        if target_size_px <= 0:
            return frame

        # 4. Section 16: Resize cached overlay
        resized = resize_overlay(overlay_asset, target_size_px, target_size_px)

        # 5. Section 11 & 15: Rotate overlay preserving alpha
        rotated = rotate_overlay(resized, srot)

        # 6. Section 13: Alpha blend compositing
        composited = self.compositor.blend_overlay(
            frame, rotated, target_x_px, target_y_px
        )

        return composited
