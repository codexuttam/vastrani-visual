"""
tests/test_ai_integration.py

Phase 9 Integration Tests: End-to-End Natural Language -> AI -> Safety -> Device Pipeline
"""

import pytest
from ai.router import AIRouter
from ai.openai_client import OpenAIClient
from devices.registry import DeviceRegistry
from devices.controller import DeviceController
from devices.models import DeviceState, DeviceType


@pytest.fixture
def integration_environment():
    reg = DeviceRegistry()
    
    light = DeviceState(
        device_id="LIGHT_01",
        name="Living Room Light",
        device_type=DeviceType.LIGHT,
        power=False,
        level=0,
    )
    fan = DeviceState(
        device_id="FAN_01",
        name="Ceiling Fan",
        device_type=DeviceType.FAN,
        power=False,
        level=0,
    )

    reg.register(light)
    reg.register(fan)
    ctrl = DeviceController(registry=reg)

    client = OpenAIClient(api_key="", enabled=True, min_request_interval=0.0)
    router = AIRouter(ai_mode="EVENT", client=client)

    yield reg, ctrl, router

    router.stop()


def test_e2e_turn_on_light(integration_environment):
    reg, ctrl, router = integration_environment
    rec = router.process_text_command_sync("Turn the living room light on", reg, ctrl)

    assert rec.status == "SUCCESS"
    assert rec.intent == "TURN_ON"
    assert rec.device_id == "LIGHT_01"
    assert reg.get("LIGHT_01").state.power is True


def test_e2e_set_light_level(integration_environment):
    reg, ctrl, router = integration_environment
    rec = router.process_text_command_sync("Set the living room light to 70 percent", reg, ctrl)

    assert rec.status == "SUCCESS"
    assert rec.intent == "SET_LEVEL"
    assert rec.device_id == "LIGHT_01"
    assert rec.value == 70
    assert reg.get("LIGHT_01").state.level == 70


def test_e2e_set_fan_speed_valid(integration_environment):
    reg, ctrl, router = integration_environment
    rec = router.process_text_command_sync("Set the fan to speed 2", reg, ctrl)

    assert rec.status == "SUCCESS"
    assert rec.intent == "SET_LEVEL"
    assert rec.device_id == "FAN_01"
    assert rec.value == 2
    assert reg.get("FAN_01").state.level == 2


def test_e2e_unknown_device_rejected(integration_environment):
    reg, ctrl, router = integration_environment
    rec = router.process_text_command_sync("Turn on the bedroom AC", reg, ctrl)

    # Mock fallback returns NONE intent with low confidence for unknown inputs
    # Safety validator rejects as LOW_CONFIDENCE or REJECTED
    assert rec.status in ("REJECTED", "LOW_CONFIDENCE")


def test_e2e_unknown_intent_rejected(integration_environment):
    reg, ctrl, router = integration_environment
    rec = router.process_text_command_sync("Launch rocket into space", reg, ctrl)

    # Mock fallback returns NONE intent with confidence 0.1 for gibberish
    # Safety validator rejects as LOW_CONFIDENCE
    assert rec.status in ("REJECTED", "LOW_CONFIDENCE", "ERROR")
