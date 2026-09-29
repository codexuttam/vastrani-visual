"""
tests/test_serial_protocol.py

Unit tests for Phase 6 Serial Protocol formatting and parsing.
Requires NO physical hardware.
"""

import pytest
from devices.command_bus import device_command_to_serial
from devices.serial_transport import parse_serial_response, SerialResponse


def test_device_command_to_serial_formatting():
    """Verify deterministic command string formatting."""
    assert device_command_to_serial("LIGHT_01", "POWER_ON") == "ON|LIGHT_01"
    assert device_command_to_serial("LIGHT_01", "POWER_OFF") == "OFF|LIGHT_01"
    assert device_command_to_serial("LIGHT_01", "TOGGLE") == "TOGGLE|LIGHT_01"
    assert device_command_to_serial("LIGHT_01", "SET_LEVEL", 80) == "SET|LIGHT_01|80"
    assert device_command_to_serial("FAN_01", "SET_LEVEL", 2) == "SET|FAN_01|2"
    assert device_command_to_serial("SERVO_01", "SET_LEVEL", 90) == "SET|SERVO_01|90"
    assert device_command_to_serial("SYSTEM", "EMERGENCY_STOP") == "ALL_OFF|SYSTEM"
    assert device_command_to_serial("ANY", "EMERGENCY_STOP") == "ALL_OFF|SYSTEM"
    assert device_command_to_serial("SYSTEM", "PING") == "PING|SYSTEM"


def test_parse_serial_response_ok():
    """Verify parsing valid OK responses from Arduino."""
    resp = parse_serial_response("OK|LIGHT_01|80")
    assert resp.success is True
    assert resp.device_id == "LIGHT_01"
    assert resp.value == "80"
    assert resp.error is None

    resp2 = parse_serial_response("OK|LIGHT_01|ON\n")
    assert resp2.success is True
    assert resp2.device_id == "LIGHT_01"
    assert resp2.value == "ON"

    resp3 = parse_serial_response("OK|SYSTEM|ALL_OFF")
    assert resp3.success is True
    assert resp3.device_id == "SYSTEM"
    assert resp3.value == "ALL_OFF"


def test_parse_serial_response_pong():
    """Verify parsing PONG handshake response."""
    resp = parse_serial_response("PONG|SYSTEM")
    assert resp.success is True
    assert resp.device_id == "SYSTEM"
    assert resp.value == "PONG"


def test_parse_serial_response_errors():
    """Verify parsing ERR responses from Arduino."""
    resp1 = parse_serial_response("ERR|INVALID_COMMAND")
    assert resp1.success is False
    assert resp1.error == "INVALID_COMMAND"

    resp2 = parse_serial_response("ERR|UNKNOWN_DEVICE")
    assert resp2.success is False
    assert resp2.error == "UNKNOWN_DEVICE"

    resp3 = parse_serial_response("ERR|INVALID_VALUE")
    assert resp3.success is False
    assert resp3.error == "INVALID_VALUE"


def test_parse_serial_response_malformed_and_empty():
    """Verify robustness against malformed or empty serial input."""
    resp_empty = parse_serial_response("")
    assert resp_empty.success is False
    assert resp_empty.error == "EMPTY_RESPONSE"

    resp_garbage = parse_serial_response("SOME_RANDOM_GARBAGE_LINE")
    assert resp_garbage.success is False
    assert resp_garbage.error == "MALFORMED_RESPONSE"
