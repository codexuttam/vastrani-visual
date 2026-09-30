"""
ar/compositor.py

Phase 8: AR Image Compositor and Alpha Blender.

Responsibilities:
    - Rotate BGRA overlay images preserving transparency and bounds
    - Resize BGRA overlay images
    - Alpha blend BGRA overlay onto BGR camera frame without clipping crashes
"""

import cv2
import numpy as np


def rotate_overlay(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates a BGRA image by angle_deg around its center.
    Expands bounding box so rotated corners are not cropped.
    """
    if abs(angle_deg) < 0.1:
        return image

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute new bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust matrix for translation to new center
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Perform rotation with transparent border
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def resize_overlay(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resizes BGRA image to target width and height."""
    w = max(1, width)
    h = max(1, height)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


class ARCompositor:
    """
    Blends BGRA overlay onto BGR camera frame using alpha channel blending.
    Handles out-of-bounds boundary clipping safely.
    """

    def blend_overlay(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """
        Blends 4-channel overlay onto 3-channel frame centered at (center_x, center_y).
        """
        if frame is None or overlay is None:
            return frame

        fh, fw = frame.shape[:2]
        oh, ow = overlay.shape[:2]

        if oh == 0 or ow == 0 or fh == 0 or fw == 0:
            return frame

        # Bounding box on frame
        x1 = center_x - ow // 2
        y1 = center_y - oh // 2
        x2 = x1 + ow
        y2 = y1 + oh

        # Bounding box clipping
        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(fw, x2)
        clip_y2 = min(fh, y2)

        # Check if entirely outside frame
        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return frame

        # Corresponding region inside overlay
        ox1 = clip_x1 - x1
        oy1 = clip_y1 - y1
        ox2 = ox1 + (clip_x2 - clip_x1)
        oy2 = oy1 + (clip_y2 - clip_y1)

        # Extract regions
        frame_crop   = frame[clip_y1:clip_y2, clip_x1:clip_x2]
        overlay_crop = overlay[oy1:oy2, ox1:ox2]

        # Extract color and alpha channels
        overlay_bgr = overlay_crop[:, :, :3]
        alpha = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis]

        # Perform alpha blending
        blended = (frame_crop * (1.0 - alpha) + overlay_bgr * alpha).astype(np.uint8)

        # Apply blended patch back to frame
        frame[clip_y1:clip_y2, clip_x1:clip_x2] = blended
        return frame
