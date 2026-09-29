"""
tests/test_command_bus.py

Unit tests for Phase 6 CommandBus in VIRTUAL and HARDWARE modes using mocks.
Requires NO physical hardware.
"""

from unittest.mock import MagicMock
import pytest
from devices.command_bus import CommandBus
from devices.serial_transport import SerialResponse


def test_command_bus_virtual_mode():
    """Verify CommandBus in VIRTUAL mode does not transmit over serial."""
    mock_transport = MagicMock()
    bus = CommandBus(mode="VIRTUAL", transport=mock_transport)

    hw_sent, hw_ack, err = bus.send("LIGHT_01", "POWER_ON")
    assert hw_sent is False
    assert hw_ack is False
    assert err is None
    mock_transport.send_command_and_get_response.assert_not_called()


def test_command_bus_hardware_mode_connected():
    """Verify CommandBus in HARDWARE mode transmits commands and parses ACK."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = True
    mock_transport.send_command_and_get_response.return_value = SerialResponse(
        success=True, device_id="LIGHT_01", value="ON", raw_response="OK|LIGHT_01|ON"
    )

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)

    hw_sent, hw_ack, err = bus.send("LIGHT_01", "POWER_ON")
    assert hw_sent is True
    assert hw_ack is True
    assert err is None
    mock_transport.send_command_and_get_response.assert_called_once_with("ON|LIGHT_01", timeout=0.5)


def test_command_bus_hardware_mode_ack_error():
    """Verify CommandBus handling of hardware error ACK."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = True
    mock_transport.send_command_and_get_response.return_value = SerialResponse(
        success=False, error="INVALID_VALUE", raw_response="ERR|INVALID_VALUE"
    )

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)

    hw_sent, hw_ack, err = bus.send("LIGHT_01", "SET_LEVEL", 150)
    assert hw_sent is True
    assert hw_ack is False
    assert err == "INVALID_VALUE"


def test_command_bus_hardware_mode_disconnected():
    """Verify CommandBus in HARDWARE mode when transport is disconnected does not crash."""
    mock_transport = MagicMock()
    mock_transport.is_connected.return_value = False

    bus = CommandBus(mode="HARDWARE", transport=mock_transport)

    hw_sent, hw_ack, err = bus.send("LIGHT_01", "POWER_ON")
    assert hw_sent is False
    assert hw_ack is False
    assert err == "Arduino: DISCONNECTED"
