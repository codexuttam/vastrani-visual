"""
ar/effects.py

Phase 8: AR Effect Controller and State Management.

Responsibilities:
    - Maintain ARState (enabled, current_effect, current_index, visible)
    - Consume abstract GestureActions to control AR effects:
        - SWIPE_RIGHT  -> Next effect
        - SWIPE_LEFT   -> Previous effect
        - TWO_FINGERS  -> Select / Activate effect
        - PINCH        -> Toggle AR ON / OFF
"""

from dataclasses import dataclass
from typing import List, Optional
from gestures.types import GestureAction
from ar.assets import EMOJI_REGISTRY


@dataclass
class ARState:
    """Independent AR state representation."""
    enabled: bool = True
    current_effect: str = "happy"
    current_index: int = 0
    visible: bool = True

    def summary(self) -> str:
        status = "ACTIVE" if (self.enabled and self.visible) else "HIDDEN"
        return f"ARState[{status}] effect='{self.current_effect}' idx={self.current_index}"


class ARController:
    """
    Manages AR effect selection and toggling via abstract GestureActions.
    """

    def __init__(self, effects: Optional[List[str]] = None):
        self.effects: List[str] = effects or list(EMOJI_REGISTRY.keys())
        if not self.effects:
            self.effects = ["happy"]

        self.state = ARState(
            enabled=True,
            current_effect=self.effects[0],
            current_index=0,
            visible=True,
        )

    def next_effect(self) -> str:
        """Select next emoji effect with circular wrap-around."""
        n = len(self.effects)
        if n == 0:
            return self.state.current_effect
        self.state.current_index = (self.state.current_index + 1) % n
        self.state.current_effect = self.effects[self.state.current_index]
        self.state.visible = True
        return self.state.current_effect

    def previous_effect(self) -> str:
        """Select previous emoji effect with circular wrap-around."""
        n = len(self.effects)
        if n == 0:
            return self.state.current_effect
        self.state.current_index = (self.state.current_index - 1 + n) % n
        self.state.current_effect = self.effects[self.state.current_index]
        self.state.visible = True
        return self.state.current_effect

    def select_effect(self, effect_name: Optional[str] = None) -> str:
        """Activate specified or current highlighted effect."""
        if effect_name and effect_name in self.effects:
            self.state.current_index = self.effects.index(effect_name)
            self.state.current_effect = effect_name

        self.state.enabled = True
        self.state.visible = True
        return self.state.current_effect

    def toggle(self) -> bool:
        """Toggle AR effects ON / OFF."""
        self.state.enabled = not self.state.enabled
        return self.state.enabled

    def set_visible(self, visible: bool):
        self.state.visible = visible

    def handle_action(self, action: GestureAction):
        """
        Connect abstract gesture actions to AR control.
        """
        if action == GestureAction.NEXT or action == GestureAction.PREVIOUS:
            # We check the raw gesture or direction if needed;
            # NEXT -> next_effect(), PREVIOUS -> previous_effect()
            if action == GestureAction.NEXT:
                self.next_effect()
            else:
                self.previous_effect()
        elif action == GestureAction.SELECT:
            self.select_effect()
        elif action == GestureAction.CONFIRM:
            self.toggle()
        elif action == GestureAction.EMERGENCY_STOP:
            # Hide AR effect on emergency stop
            self.state.visible = False
