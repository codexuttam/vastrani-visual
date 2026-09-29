"""
tests/test_hardware_integration.py

Unit tests for Phase 6 end-to-end hardware integration flow with mocked hardware.
Requires NO physical hardware.
"""

from unittest.mock import MagicMock
import pytest
from devices import DeviceController, CommandBus, SerialTransport, SerialResponse
from gestures.types import GestureAction, ControlMode


def test_validation_prevents_invalid_command_reaching_transport():
    """
    Requirement 12: Never allow an invalid command to reach Arduino.
    Validate bounds & unknown devices before transmission.
    """
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = True
    bus = CommandBus(mode="HARDWARE", transport=mock_transport)
    controller = DeviceController(command_bus=bus, mode="HARDWARE")

    # 1. LIGHT_01 level 150 -> REJECT
    res1 = controller.set_level("LIGHT_01", 150)
    assert res1.success is False
    mock_transport.send_command_and_get_response.assert_not_called()

    # 2. FAN_01 level 7 -> REJECT
    res2 = controller.set_level("FAN_01", 7)
    assert res2.success is False
    mock_transport.send_command_and_get_response.assert_not_called()

    # 3. SERVO_01 level 300 -> REJECT
    res3 = controller.set_level("SERVO_01", 300)
    assert res3.success is False
    mock_transport.send_command_and_get_response.assert_not_called()

    # 4. UNKNOWN_01 -> REJECT
    res4 = controller.power_on("UNKNOWN_01")
    assert res4.success is False
    mock_transport.send_command_and_get_response.assert_not_called()


def test_hardware_mode_valid_command_transmission():
    """Verify valid gesture actions translate to validated hardware commands."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = True
    mock_transport.send_command_and_get_response.return_value = SerialResponse(
        success=True, device_id="LIGHT_01", value="ON", raw_response="OK|LIGHT_01|ON"
    )

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)
    controller = DeviceController(command_bus=bus, mode="HARDWARE")

    res = controller.power_on("LIGHT_01")
    assert res.success is True

    history = controller.command_history()
    assert len(history) == 1
    rec = history[0]
    assert rec.device_id == "LIGHT_01"
    assert rec.action == "POWER_ON"
    assert rec.virtual_success is True
    assert rec.hardware_sent is True
    assert rec.hardware_ack is True


def test_emergency_stop_hardware_flow():
    """Verify emergency stop sends ALL_OFF|SYSTEM to transport and turns off all devices."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = True
    mock_transport.send_command_and_get_response.return_value = SerialResponse(
        success=True, device_id="SYSTEM", value="ALL_OFF", raw_response="OK|SYSTEM|ALL_OFF"
    )

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)
    controller = DeviceController(command_bus=bus, mode="HARDWARE")

    # Turn light on first
    controller.registry.get("LIGHT_01").power_on()

    res = controller.handle_action(GestureAction.EMERGENCY_STOP)
    assert res.success is True

    # Check hardware ALL_OFF was sent
    mock_transport.send_command_and_get_response.assert_called_with("ALL_OFF|SYSTEM", timeout=0.5)

    # Check all virtual devices are OFF
    for dev in controller.registry.list_devices():
        assert dev.state.power is False


def test_serial_disconnect_recovery_does_not_crash():
    """Verify application handles hardware disconnect during operation without crashing."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = False
    mock_transport.connect.return_value = False  # fails reconnect

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)
    controller = DeviceController(command_bus=bus, mode="HARDWARE")

    # Operation should update virtual state and return without throwing crash exception
    res = controller.power_on("LIGHT_01")
    assert res.success is True  # Virtual state succeeded
    history = controller.command_history()
    assert len(history) == 1
    rec = history[0]
    assert rec.virtual_success is True
    assert rec.hardware_sent is False
    assert rec.error == "Arduino: DISCONNECTED"
