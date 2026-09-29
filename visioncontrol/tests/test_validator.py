"""
tests/test_validator.py

Phase 5: Unit tests for DeviceValidator.
"""

import pytest

from devices.models import DeviceState, DeviceType
from devices.validator import DeviceValidator, SUPPORTED_ACTIONS


REGISTERED = {"LIGHT_01", "FAN_01", "MUSIC_01", "SERVO_01"}


def light_state() -> DeviceState:
    return DeviceState("LIGHT_01", "Living Room Light", DeviceType.LIGHT)

def fan_state() -> DeviceState:
    return DeviceState("FAN_01", "Ceiling Fan", DeviceType.FAN)

def servo_state() -> DeviceState:
    return DeviceState("SERVO_01", "Servo Motor", DeviceType.SERVO)


class TestDeviceExists:
    def test_known_device_passes(self):
        v = DeviceValidator()
        result = v.validate_device_exists("LIGHT_01", REGISTERED)
        assert result.success

    def test_unknown_device_fails(self):
        v = DeviceValidator()
        result = v.validate_device_exists("GHOST", REGISTERED)
        assert not result.success
        assert "GHOST" in result.message

    def test_empty_registry_fails(self):
        v = DeviceValidator()
        result = v.validate_device_exists("LIGHT_01", set())
        assert not result.success


class TestActionValidation:
    def test_all_supported_actions_pass(self):
        v = DeviceValidator()
        for action in SUPPORTED_ACTIONS:
            assert v.validate_action(action).success

    def test_unknown_action_fails(self):
        v = DeviceValidator()
        result = v.validate_action("EXPLODE")
        assert not result.success

    def test_lowercase_action_fails(self):
        v = DeviceValidator()
        result = v.validate_action("power_on")
        assert not result.success


class TestLevelValidation:
    def test_light_valid_range(self):
        v = DeviceValidator()
        s = light_state()
        assert v.validate_level("LIGHT_01", 0, s).success
        assert v.validate_level("LIGHT_01", 50, s).success
        assert v.validate_level("LIGHT_01", 100, s).success

    def test_light_101_fails(self):
        v = DeviceValidator()
        s = light_state()
        result = v.validate_level("LIGHT_01", 101, s)
        assert not result.success

    def test_fan_valid_range(self):
        v = DeviceValidator()
        s = fan_state()
        for speed in range(4):
            assert v.validate_level("FAN_01", speed, s).success

    def test_fan_4_fails(self):
        v = DeviceValidator()
        s = fan_state()
        assert not v.validate_level("FAN_01", 4, s).success

    def test_servo_valid_range(self):
        v = DeviceValidator()
        s = servo_state()
        assert v.validate_level("SERVO_01", 0, s).success
        assert v.validate_level("SERVO_01", 90, s).success
        assert v.validate_level("SERVO_01", 180, s).success

    def test_servo_181_fails(self):
        v = DeviceValidator()
        s = servo_state()
        assert not v.validate_level("SERVO_01", 181, s).success

    def test_non_int_fails(self):
        v = DeviceValidator()
        s = light_state()
        assert not v.validate_level("LIGHT_01", 50.5, s).success   # type: ignore


class TestFullCommandValidation:
    def test_power_on_valid(self):
        v = DeviceValidator()
        result = v.validate_command("LIGHT_01", "POWER_ON", REGISTERED)
        assert result.success

    def test_unknown_device_fails(self):
        v = DeviceValidator()
        result = v.validate_command("GHOST", "POWER_ON", REGISTERED)
        assert not result.success

    def test_unknown_action_fails(self):
        v = DeviceValidator()
        result = v.validate_command("LIGHT_01", "EXPLODE", REGISTERED)
        assert not result.success

    def test_set_level_without_value_fails(self):
        v = DeviceValidator()
        result = v.validate_command("LIGHT_01", "SET_LEVEL", REGISTERED, state=light_state(), level=None)
        assert not result.success

    def test_set_level_valid(self):
        v = DeviceValidator()
        result = v.validate_command("LIGHT_01", "SET_LEVEL", REGISTERED, state=light_state(), level=80)
        assert result.success

    def test_set_level_out_of_range_fails(self):
        v = DeviceValidator()
        result = v.validate_command("LIGHT_01", "SET_LEVEL", REGISTERED, state=light_state(), level=999)
        assert not result.success

    def test_fan_speed_out_of_range_fails(self):
        v = DeviceValidator()
        result = v.validate_command("FAN_01", "SET_LEVEL", REGISTERED, state=fan_state(), level=5)
        assert not result.success

    def test_servo_angle_out_of_range_fails(self):
        v = DeviceValidator()
        result = v.validate_command("SERVO_01", "SET_LEVEL", REGISTERED, state=servo_state(), level=200)
        assert not result.success
