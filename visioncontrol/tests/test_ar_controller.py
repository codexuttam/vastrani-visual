"""
tests/test_ar_controller.py

Unit tests for Phase 8 ARController and gesture actions.
Requires NO camera hardware.
"""

import pytest
from ar.effects import ARController, ARState
from gestures.types import GestureAction


def test_ar_controller_init():
    """Verify default initial AR controller state."""
    controller = ARController()
    assert controller.state.enabled is True
    assert controller.state.current_effect == "happy"
    assert controller.state.current_index == 0


def test_ar_controller_next_and_previous_wrap_around():
    """Verify next and previous effect navigation with circular wrap-around."""
    controller = ARController(effects=["happy", "laughing", "cool"])

    # happy -> laughing
    eff1 = controller.next_effect()
    assert eff1 == "laughing"
    assert controller.state.current_index == 1

    # laughing -> cool
    eff2 = controller.next_effect()
    assert eff2 == "cool"
    assert controller.state.current_index == 2

    # cool -> happy (wrap-around)
    eff3 = controller.next_effect()
    assert eff3 == "happy"
    assert controller.state.current_index == 0

    # happy -> cool (previous wrap-around)
    eff_prev = controller.previous_effect()
    assert eff_prev == "cool"
    assert controller.state.current_index == 2


def test_ar_controller_toggle():
    """Verify toggle ON/OFF."""
    controller = ARController()
    assert controller.state.enabled is True

    enabled1 = controller.toggle()
    assert enabled1 is False
    assert controller.state.enabled is False

    enabled2 = controller.toggle()
    assert enabled2 is True
    assert controller.state.enabled is True


def test_ar_controller_gesture_action_integration():
    """Verify mapping abstract gesture actions to AR operations."""
    controller = ARController(effects=["happy", "laughing", "cool"])

    # SWIPE_RIGHT -> NEXT
    controller.handle_action(GestureAction.NEXT)
    assert controller.state.current_effect == "laughing"

    # SWIPE_LEFT -> PREVIOUS
    controller.handle_action(GestureAction.PREVIOUS)
    assert controller.state.current_effect == "happy"

    # TWO_FINGERS -> SELECT
    controller.handle_action(GestureAction.SELECT)
    assert controller.state.visible is True

    # PINCH -> CONFIRM -> TOGGLE
    controller.handle_action(GestureAction.CONFIRM)
    assert controller.state.enabled is False

    # EMERGENCY_STOP -> Hide AR
    controller.handle_action(GestureAction.EMERGENCY_STOP)
    assert controller.state.visible is False
