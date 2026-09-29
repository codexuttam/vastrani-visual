"""
tests/test_serial_transport.py

Unit tests for Phase 6 SerialTransport connection lifecycle and non-blocking I/O.
Requires NO physical hardware.
"""

from unittest.mock import MagicMock, patch
import pytest
from devices.serial_transport import SerialTransport, SerialResponse


def test_serial_transport_init():
    """Verify SerialTransport default state."""
    transport = SerialTransport(port="COM3", baud_rate=115200, auto_detect=False)
    assert transport.port == "COM3"
    assert transport.baud_rate == 115200
    assert transport.is_connected() is False
    assert transport.is_unresponsive is False


@patch("devices.serial_transport.serial")
def test_serial_transport_connect_success(mock_serial_mod):
    """Verify successful connection setup."""
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_serial_mod.Serial.return_value = mock_ser

    transport = SerialTransport(port="/dev/ttyACM0", auto_detect=False, debug_serial=False)
    success = transport.connect()

    assert success is True
    assert transport.is_connected() is True
    mock_serial_mod.Serial.assert_called_once_with(
        port="/dev/ttyACM0", baudrate=115200, timeout=1.0, write_timeout=1.0
    )


@patch("devices.serial_transport.serial")
def test_serial_transport_send_and_read(mock_serial_mod):
    """Verify sending commands and reading line responses."""
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.readline.return_value = b"OK|LIGHT_01|80\n"
    mock_serial_mod.Serial.return_value = mock_ser

    transport = SerialTransport(port="/dev/ttyACM0", auto_detect=False, debug_serial=False)
    transport.connect()

    sent = transport.send("SET|LIGHT_01|80")
    assert sent is True
    mock_ser.write.assert_called_once_with(b"SET|LIGHT_01|80\n")

    resp = transport.read_response()
    assert resp.success is True
    assert resp.device_id == "LIGHT_01"
    assert resp.value == "80"


@patch("devices.serial_transport.serial")
def test_serial_transport_ping_handshake(mock_serial_mod):
    """Verify PING|SYSTEM -> PONG|SYSTEM handshake."""
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.readline.return_value = b"PONG|SYSTEM\n"
    mock_serial_mod.Serial.return_value = mock_ser

    transport = SerialTransport(port="/dev/ttyACM0", auto_detect=False, debug_serial=False)
    transport.connect()

    ping_ok = transport.ping()
    assert ping_ok is True
    assert transport.is_unresponsive is False


def test_serial_transport_send_when_disconnected():
    """Verify send returns False when transport is disconnected without throwing exception."""
    transport = SerialTransport(port="/dev/ttyACM0", auto_detect=False)
    assert transport.send("PING|SYSTEM") is False
    assert transport.read_response().success is False
