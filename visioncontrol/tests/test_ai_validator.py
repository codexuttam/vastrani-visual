"""
tests/test_ai_validator.py

Phase 9 Unit Tests: AISafetyValidator
"""

import pytest
from ai.schemas import AIIntent
from ai.safety import AISafetyValidator
from devices.registry import DeviceRegistry
from devices.models import DeviceState, DeviceType


@pytest.fixture
def sample_registry():
    reg = DeviceRegistry()
    reg.register(DeviceState(
        device_id="LIGHT_01",
        name="Living Room Light",
        device_type=DeviceType.LIGHT,
    ))
    reg.register(DeviceState(
        device_id="FAN_01",
        name="Ceiling Fan",
        device_type=DeviceType.FAN,
    ))
    return reg


def test_validator_valid_turn_on(sample_registry):
    validator = AISafetyValidator(min_confidence=0.75)
    intent = AIIntent(intent="TURN_ON", device_id="LIGHT_01", value=None, confidence=0.95, reason="Valid")
    is_safe, msg = validator.validate(intent, sample_registry)
    assert is_safe is True
    assert "Passed" in msg


def test_validator_unknown_device_rejected(sample_registry):
    validator = AISafetyValidator()
    intent = AIIntent(intent="TURN_ON", device_id="LIGHT_99", value=None, confidence=0.90, reason="Unknown dev")
    is_safe, msg = validator.validate(intent, sample_registry)
    assert is_safe is False
    assert "Unknown device ID" in msg


def test_validator_invalid_level_value_rejected(sample_registry):
    validator = AISafetyValidator()
    # Light level max is 100
    intent = AIIntent(intent="SET_LEVEL", device_id="LIGHT_01", value=150, confidence=0.90, reason="Too high")
    is_safe, msg = validator.validate(intent, sample_registry)
    assert is_safe is False
    assert "out of range" in msg

    # Fan level max is 3
    intent_fan = AIIntent(intent="SET_LEVEL", device_id="FAN_01", value=10, confidence=0.90, reason="Too high fan")
    is_safe_f, msg_f = validator.validate(intent_fan, sample_registry)
    assert is_safe_f is False
    assert "out of range" in msg_f


def test_validator_low_confidence_rejected(sample_registry):
    validator = AISafetyValidator(min_confidence=0.75)
    intent = AIIntent(intent="TURN_ON", device_id="LIGHT_01", value=None, confidence=0.40, reason="Low conf")
    is_safe, msg = validator.validate(intent, sample_registry)
    assert is_safe is False
    assert "Low confidence" in msg
