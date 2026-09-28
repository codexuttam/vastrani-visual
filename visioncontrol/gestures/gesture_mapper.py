"""
gestures/gesture_mapper.py

Phase 4: Maps recognized gestures to abstract GestureActions.
Also manages the application ControlMode state.

Does NOT:
    - control Arduino
    - call OpenAI
    - control devices
    - contain UI code
"""

from gestures.types import GestureType, GestureAction, GestureEvent, ControlMode


# ─── Gesture → Action mapping table ───────────────────────────────────────────

GESTURE_ACTION_MAP: dict[GestureType, GestureAction] = {
    GestureType.OPEN_PALM:   GestureAction.ENTER_CONTROL,
    GestureType.FIST:        GestureAction.EMERGENCY_STOP,
    GestureType.TWO_FINGERS: GestureAction.SELECT,
    GestureType.PINCH:       GestureAction.CONFIRM,
    GestureType.SWIPE_LEFT:  GestureAction.PREVIOUS,
    GestureType.SWIPE_RIGHT: GestureAction.NEXT,
    GestureType.NONE:        GestureAction.NONE,
}


class GestureMapper:
    """
    Translates GestureEvents into GestureActions and manages ControlMode.

    Usage (called once per frame, only when a new event fires):
        action = mapper.map_event(event)
    """

    def __init__(self):
        self.mode = ControlMode.IDLE

    def map_event(self, event: GestureEvent) -> GestureAction:
        """
        Resolve the abstract action for a GestureEvent and update ControlMode.
        Returns GestureAction.NONE if the event gesture is not mapped.
        """
        action = GESTURE_ACTION_MAP.get(event.gesture, GestureAction.NONE)

        # Update ControlMode based on action
        if action == GestureAction.ENTER_CONTROL:
            self.mode = ControlMode.CONTROL
        elif action == GestureAction.EMERGENCY_STOP:
            self.mode = ControlMode.IDLE

        # Attach action back to event for consumers
        event.action = action
        return action

    def map_gesture(self, gesture: GestureType) -> GestureAction:
        """Direct gesture → action lookup (no side effects)."""
        return GESTURE_ACTION_MAP.get(gesture, GestureAction.NONE)
