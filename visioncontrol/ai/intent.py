"""
ai/intent.py

Phase 9: AI Intent Mapping helpers.

Converts validated AIIntent objects into device operations or gesture actions.
"""

from typing import Optional
from ai.schemas import AIIntent
from gestures.types import GestureAction


def map_ai_intent_to_gesture_action(intent: AIIntent) -> GestureAction:
    """
    Maps an AI intent to a corresponding local GestureAction enum.
    """
    it = intent.intent.upper()
    if it == "NEXT":
        return GestureAction.NEXT
    elif it == "PREVIOUS":
        return GestureAction.PREVIOUS
    elif it == "SELECT":
        return GestureAction.SELECT
    elif it == "CONFIRM":
        return GestureAction.CONFIRM
    elif it in ("STOP", "EMERGENCY_STOP"):
        return GestureAction.EMERGENCY_STOP
    elif it == "TURN_ON" or it == "POWER_ON":
        return GestureAction.ENTER_CONTROL
    return GestureAction.NONE
