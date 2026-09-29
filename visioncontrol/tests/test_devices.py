"""
tests/test_devices.py

Phase 5: Unit tests for DeviceState, VirtualDevice, and DeviceRegistry.
"""

import pytest

from devices.models import DeviceState, DeviceType
from devices.virtual_device import VirtualDevice
from devices.registry import DeviceRegistry, create_default_registry


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_light() -> VirtualDevice:
    return VirtualDevice(DeviceState("LIGHT_01", "Living Room Light", DeviceType.LIGHT))

def make_fan() -> VirtualDevice:
    return VirtualDevice(DeviceState("FAN_01", "Ceiling Fan", DeviceType.FAN))

def make_music() -> VirtualDevice:
    return VirtualDevice(DeviceState("MUSIC_01", "Music Player", DeviceType.MUSIC))

def make_servo() -> VirtualDevice:
    return VirtualDevice(DeviceState("SERVO_01", "Servo Motor", DeviceType.SERVO))


# ─── DeviceState ──────────────────────────────────────────────────────────────

class TestDeviceState:
    def test_light_max_level(self):
        s = DeviceState("L", "L", DeviceType.LIGHT)
        assert s.max_level == 100

    def test_fan_max_level(self):
        s = DeviceState("F", "F", DeviceType.FAN)
        assert s.max_level == 3

    def test_music_max_level(self):
        s = DeviceState("M", "M", DeviceType.MUSIC)
        assert s.max_level == 100

    def test_servo_max_level(self):
        s = DeviceState("S", "S", DeviceType.SERVO)
        assert s.max_level == 180

    def test_min_level_zero(self):
        for dt in DeviceType:
            s = DeviceState("X", "X", dt)
            assert s.min_level == 0

    def test_summary_contains_id(self):
        s = DeviceState("LIGHT_01", "LR Light", DeviceType.LIGHT)
        assert "LIGHT_01" in s.summary()


# ─── VirtualDevice — power ────────────────────────────────────────────────────

class TestVirtualDevicePower:
    def test_power_on(self):
        d = make_light()
        result = d.power_on()
        assert result.success
        assert d.state.power is True

    def test_power_off(self):
        d = make_light()
        d.power_on()
        result = d.power_off()
        assert result.success
        assert d.state.power is False

    def test_power_on_already_on(self):
        d = make_light()
        d.power_on()
        result = d.power_on()
        assert result.success          # idempotent

    def test_power_off_already_off(self):
        d = make_light()
        result = d.power_off()
        assert result.success          # idempotent

    def test_toggle_off_to_on(self):
        d = make_light()
        result = d.toggle()
        assert result.success
        assert d.state.power is True

    def test_toggle_on_to_off(self):
        d = make_light()
        d.power_on()
        result = d.toggle()
        assert result.success
        assert d.state.power is False

    def test_disabled_device_rejected(self):
        d = make_light()
        d.state.enabled = False
        assert not d.power_on().success
        assert not d.power_off().success
        assert not d.toggle().success


# ─── VirtualDevice — levels ──────────────────────────────────────────────────

class TestVirtualDeviceLevels:
    def test_light_valid_level(self):
        d = make_light()
        result = d.set_level(80)
        assert result.success
        assert d.state.level == 80

    def test_light_min_level(self):
        d = make_light()
        assert d.set_level(0).success

    def test_light_max_level(self):
        d = make_light()
        assert d.set_level(100).success

    def test_light_level_101_rejected(self):
        d = make_light()
        result = d.set_level(101)
        assert not result.success

    def test_light_level_negative_rejected(self):
        d = make_light()
        result = d.set_level(-1)
        assert not result.success

    def test_fan_valid_speed(self):
        d = make_fan()
        for v in range(4):   # 0, 1, 2, 3
            assert d.set_level(v).success

    def test_fan_speed_4_rejected(self):
        d = make_fan()
        result = d.set_level(4)
        assert not result.success

    def test_servo_valid_angle(self):
        d = make_servo()
        assert d.set_level(0).success
        assert d.set_level(90).success
        assert d.set_level(180).success

    def test_servo_angle_181_rejected(self):
        d = make_servo()
        result = d.set_level(181)
        assert not result.success

    def test_music_valid_volume(self):
        d = make_music()
        assert d.set_level(50).success
        assert d.set_level(100).success

    def test_non_int_level_rejected(self):
        d = make_light()
        result = d.set_level(50.5)   # type: ignore
        assert not result.success


# ─── VirtualDevice — emergency stop ──────────────────────────────────────────

class TestEmergencyStop:
    def test_emergency_stop_turns_off(self):
        d = make_light()
        d.power_on()
        d.set_level(80)
        result = d.emergency_stop()
        assert result.success
        assert d.state.power is False

    def test_emergency_stop_resets_level(self):
        d = make_fan()
        d.power_on()
        d.set_level(3)
        d.emergency_stop()
        assert d.state.level == 0


# ─── DeviceRegistry ──────────────────────────────────────────────────────────

class TestDeviceRegistry:
    def test_register_and_get(self):
        reg = DeviceRegistry()
        reg.register(DeviceState("L", "Light", DeviceType.LIGHT))
        assert reg.get("L") is not None

    def test_list_devices_order(self):
        reg = create_default_registry()
        ids = [d.device_id for d in reg.list_devices()]
        assert ids == ["LIGHT_01", "FAN_01", "MUSIC_01", "SERVO_01"]

    def test_duplicate_register_raises(self):
        reg = DeviceRegistry()
        reg.register(DeviceState("L", "Light", DeviceType.LIGHT))
        with pytest.raises(ValueError):
            reg.register(DeviceState("L", "Light2", DeviceType.LIGHT))

    def test_get_unknown_returns_none(self):
        reg = DeviceRegistry()
        assert reg.get("UNKNOWN") is None

    def test_default_registry_count(self):
        reg = create_default_registry()
        assert reg.count == 4


# ─── Selection — next / previous / wrap ──────────────────────────────────────

class TestRegistrySelection:
    def test_first_selected_by_default(self):
        reg = create_default_registry()
        assert reg.current().device_id == "LIGHT_01"

    def test_next_advances(self):
        reg = create_default_registry()
        reg.next()
        assert reg.current().device_id == "FAN_01"

    def test_next_wraps_around(self):
        reg = create_default_registry()
        reg.next()   # FAN
        reg.next()   # MUSIC
        reg.next()   # SERVO
        reg.next()   # wraps → LIGHT
        assert reg.current().device_id == "LIGHT_01"

    def test_previous_wraps_around(self):
        reg = create_default_registry()
        # At LIGHT, going previous should wrap to SERVO
        reg.previous()
        assert reg.current().device_id == "SERVO_01"

    def test_select_by_id(self):
        reg = create_default_registry()
        device = reg.select("MUSIC_01")
        assert device is not None
        assert reg.current().device_id == "MUSIC_01"

    def test_select_unknown_returns_none(self):
        reg = create_default_registry()
        result = reg.select("UNKNOWN")
        assert result is None
