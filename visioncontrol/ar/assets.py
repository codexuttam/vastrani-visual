"""
ar/assets.py

Phase 8: Emoji Asset Manager and Registry.

Responsibilities:
    - Register local emoji asset paths
    - Pre-load and cache BGRA images once
    - Safe retrieval by asset ID
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional

EMOJI_REGISTRY: Dict[str, str] = {
    "happy":    "assets/emojis/happy.png",
    "laughing": "assets/emojis/laughing.png",
    "cool":     "assets/emojis/cool.png",
    "angry":    "assets/emojis/angry.png",
    "love":     "assets/emojis/love.png",
    "thinking": "assets/emojis/thinking.png",
}

# Text icons for UI rendering fallback
EMOJI_TEXT_ICONS: Dict[str, str] = {
    "happy":    "HAPPY",
    "laughing": "LAUGH",
    "cool":     "COOL",
    "angry":    "ANGRY",
    "love":     "LOVE",
    "thinking": "THINK",
}


class AssetManager:
    """
    Caches and manages AR emoji assets in memory.
    Prevents reading images from disk every frame.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Resolve relative to project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir: str = base_dir
        self._cache: Dict[str, np.ndarray] = {}
        self.load_all()

    def load_all(self):
        """Pre-load all registered emoji images into memory cache."""
        for effect_id, rel_path in EMOJI_REGISTRY.items():
            full_path = os.path.join(self.base_dir, rel_path)
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Ensure 4-channel BGRA image format
                    if img.shape[2] == 3:
                        b, g, r = cv2.split(img)
                        alpha = np.full(b.shape, 255, dtype=np.uint8)
                        img = cv2.merge([b, g, r, alpha])
                    self._cache[effect_id] = img
                else:
                    print(f"[AssetManager] Warning: Failed to read asset '{full_path}'")
            else:
                # Create a simple fallback color square if file missing
                print(f"[AssetManager] Warning: Asset file not found '{full_path}'")

    def get(self, effect_id: str) -> Optional[np.ndarray]:
        """Retrieve cached BGRA image array for specified effect ID."""
        return self._cache.get(effect_id)

    def list_effects(self) -> List[str]:
        """Return list of available effect IDs."""
        return list(EMOJI_REGISTRY.keys())

    def has_effect(self, effect_id: str) -> bool:
        return effect_id in self._cache
