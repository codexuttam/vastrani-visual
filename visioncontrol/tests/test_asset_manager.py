"""
tests/test_asset_manager.py

Unit tests for Phase 8 AssetManager and EMOJI_REGISTRY.
Requires NO camera hardware.
"""

import pytest
import numpy as np
from ar.assets import AssetManager, EMOJI_REGISTRY


def test_asset_manager_registered_effects():
    """Verify all 6 configured emoji effects are registered and loaded."""
    manager = AssetManager()
    effects = manager.list_effects()

    assert "happy" in effects
    assert "laughing" in effects
    assert "cool" in effects
    assert "angry" in effects
    assert "love" in effects
    assert "thinking" in effects
    assert len(effects) == 6


def test_asset_manager_get_cached_image():
    """Verify get returns valid BGRA numpy array for cached images."""
    manager = AssetManager()
    happy_img = manager.get("happy")

    assert happy_img is not None
    assert isinstance(happy_img, np.ndarray)
    assert happy_img.ndim == 3
    assert happy_img.shape[2] == 4  # 4 channels (BGRA)


def test_asset_manager_missing_asset_handling():
    """Verify retrieving missing asset ID returns None safely without exception."""
    manager = AssetManager()
    missing_img = manager.get("non_existent_emoji")

    assert missing_img is None
    assert manager.has_effect("non_existent_emoji") is False
