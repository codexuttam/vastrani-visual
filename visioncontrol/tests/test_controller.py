"""
tests/test_controller.py

Phase 5: Unit and integration tests for DeviceController.
"""

import pytest

from gestures.types import GestureAction, ControlMode
from devices.controller import DeviceController
from devices.registry import create_default_registry


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_controller() -> DeviceController:
    return DeviceController(registry=create_default_registry())


# ─── Control mode ─────────────────────────────────────────────────────────────

class TestControlMode:
    def test_starts_idle(self):
        c = make_controller()
        assert c.mode == ControlMode.IDLE

    def test_enter_control(self):
        c = make_controller()
        c.handle_action(GestureAction.ENTER_CONTROL)
        assert c.mode == ControlMode.CONTROL

    def test_emergency_stop_returns_idle(self):
        c = make_controller()
        c.handle_action(GestureAction.ENTER_CONTROL)
        c.handle_action(GestureAction.EMERGENCY_STOP)
        assert c.mode == ControlMode.IDLE


# ─── Navigation ───────────────────────────────────────────────────────────────

class TestNavigation:
    def test_next_changes_device(self):
        c = make_controller()
        assert c.registry.current().device_id == "LIGHT_01"
        c.handle_action(GestureAction.NEXT)
        assert c.registry.current().device_id == "FAN_01"

    def test_previous_changes_device(self):
        c = make_controller()
        c.handle_action(GestureAction.NEXT)   # → FAN
        c.handle_action(GestureAction.PREVIOUS)  # → LIGHT
        assert c.registry.current().device_id == "LIGHT_01"

    def test_next_wraps(self):
        c = make_controller()
        for _ in range(4):
            c.handle_action(GestureAction.NEXT)
        assert c.registry.current().device_id == "LIGHT_01"

    def test_previous_wraps(self):
        c = make_controller()
        c.handle_action(GestureAction.PREVIOUS)
        assert c.registry.current().device_id == "SERVO_01"

    def test_next_clears_pending_action(self):
        c = make_controller()
        c.handle_action(GestureAction.SELECT)
        assert c.pending_action is not None
        c.handle_action(GestureAction.NEXT)
        assert c.pending_action is None


# ─── Select + Confirm ─────────────────────────────────────────────────────────

class TestSelectConfirm:
    def test_select_sets_pending(self):
        c = make_controller()
        c.handle_action(GestureAction.SELECT)
        assert c.pending_action == "TOGGLE"

    def test_confirm_executes_toggle(self):
        c = make_controller()
        c.handle_action(GestureAction.SELECT)
        device_before = c.registry.current().state.power
        c.handle_action(GestureAction.CONFIRM)
        device_after = c.registry.current().state.power
        assert device_after != device_before

    def test_confirm_without_select_is_safe(self):
        c = make_controller()
        result = c.handle_action(GestureAction.CONFIRM)
        assert result.success   # should not crash

    def test_confirm_clears_pending(self):
        c = make_controller()
        c.handle_action(GestureAction.SELECT)
        c.handle_action(GestureAction.CONFIRM)
        assert c.pending_action is None


# ─── Direct device operations ─────────────────────────────────────────────────

class TestDirectOps:
    def test_power_on(self):
        c = make_controller()
        result = c.power_on("LIGHT_01")
        assert result.success
        assert c.registry.get("LIGHT_01").state.power is True

    def test_power_off(self):
        c = make_controller()
        c.power_on("LIGHT_01")
        result = c.power_off("LIGHT_01")
        assert result.success
        assert c.registry.get("LIGHT_01").state.power is False

    def test_toggle(self):
        c = make_controller()
        c.power_on("LIGHT_01")
        c.toggle("LIGHT_01")
        assert c.registry.get("LIGHT_01").state.power is False

    def test_set_level_valid(self):
        c = make_controller()
        result = c.set_level("LIGHT_01", 75)
        assert result.success

    def test_set_level_out_of_range(self):
        c = make_controller()
        result = c.set_level("LIGHT_01", 200)
        assert not result.success

    def test_unknown_device_rejected(self):
        c = make_controller()
        result = c.power_on("GHOST_DEVICE")
        assert not result.success


# ─── Emergency stop ───────────────────────────────────────────────────────────

class TestEmergencyStop:
    def test_all_devices_off(self):
        c = make_controller()
        for did in ["LIGHT_01", "FAN_01", "MUSIC_01", "SERVO_01"]:
            c.power_on(did)
        c.handle_action(GestureAction.EMERGENCY_STOP)
        for device in c.registry.list_devices():
            assert device.state.power is False

    def test_all_levels_reset(self):
        c = make_controller()
        c.set_level("SERVO_01", 90)
        c.handle_action(GestureAction.EMERGENCY_STOP)
        assert c.registry.get("SERVO_01").state.level == 0

    def test_pending_cleared(self):
        c = make_controller()
        c.handle_action(GestureAction.SELECT)
        c.handle_action(GestureAction.EMERGENCY_STOP)
        assert c.pending_action is None


# ─── Command history ──────────────────────────────────────────────────────────

class TestCommandHistory:
    def test_history_recorded(self):
        c = make_controller()
        c.power_on("LIGHT_01")
        history = c.command_history()
        assert len(history) >= 1
        assert history[-1].device_id == "LIGHT_01"
        assert history[-1].action == "POWER_ON"

    def test_history_limited_to_max_size(self):
        from devices.controller import COMMAND_HISTORY_SIZE
        c = make_controller()
        # Execute more commands than the history limit
        for _ in range(COMMAND_HISTORY_SIZE + 10):
            c.toggle("LIGHT_01")
        assert len(c.command_history()) <= COMMAND_HISTORY_SIZE

    def test_failed_commands_recorded(self):
        c = make_controller()
        c.set_level("LIGHT_01", 999)  # invalid
        history = c.command_history()
        assert any(not r.success for r in history)


# ─── Integration test: full gesture flow ─────────────────────────────────────

class TestGestureIntegration:
    def test_full_flow(self):
        """
        IDLE → ENTER_CONTROL → NEXT → SELECT → CONFIRM → EMERGENCY_STOP
        """
        c = make_controller()

        # Start IDLE
        assert c.mode == ControlMode.IDLE

        # OPEN_PALM → ENTER_CONTROL
        c.handle_action(GestureAction.ENTER_CONTROL)
        assert c.mode == ControlMode.CONTROL

        # SWIPE_RIGHT → NEXT → FAN_01
        c.handle_action(GestureAction.NEXT)
        assert c.registry.current().device_id == "FAN_01"

        # TWO_FINGERS → SELECT → pending TOGGLE
        c.handle_action(GestureAction.SELECT)
        assert c.pending_action is not None

        # PINCH → CONFIRM → toggle FAN
        initial_power = c.registry.current().state.power
        c.handle_action(GestureAction.CONFIRM)
        assert c.registry.current().state.power != initial_power

        # FIST → EMERGENCY_STOP → all off
        c.handle_action(GestureAction.EMERGENCY_STOP)
        for device in c.registry.list_devices():
            assert device.state.power is False
        assert c.mode == ControlMode.IDLE
