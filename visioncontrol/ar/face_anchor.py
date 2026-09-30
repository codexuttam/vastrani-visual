"""
ar/face_anchor.py

Phase 8: Face Anchor representation for AR rendering.
Re-exports FaceAnchor from vision.face_features for clean module separation.
"""

from vision.face_features import FaceAnchor

__all__ = ["FaceAnchor"]
