"""
tests/test_ai_router.py

Phase 9 Unit Tests: AIRouter
"""

import time
import pytest
from ai.router import AIRouter
from ai.openai_client import OpenAIClient
from devices.registry import DeviceRegistry
from devices.controller import DeviceController
from devices.models import DeviceState, DeviceType
from gestures.types import GestureAction


@pytest.fixture
def sample_setup():
    reg = DeviceRegistry()
    dev_state = DeviceState(
        device_id="LIGHT_01",
        name="Living Room Light",
        device_type=DeviceType.LIGHT,
        power=False,
        level=0,
    )
    reg.register(dev_state)
    ctrl = DeviceController(registry=reg)
    return reg, ctrl


def test_ai_router_off_mode(sample_setup):
    reg, ctrl = sample_setup
    router = AIRouter(ai_mode="OFF")
    record = router.process_text_command_sync("Turn on light", reg, ctrl)
    assert record.status == "DISABLED"
    assert record.intent == "NONE"
    router.stop()


def test_ai_router_sync_valid_execution(sample_setup):
    reg, ctrl = sample_setup
    client = OpenAIClient(api_key="", enabled=True)
    router = AIRouter(ai_mode="EVENT", client=client)

    assert reg.get("LIGHT_01").state.power is False

    record = router.process_text_command_sync("Turn the living room light on", reg, ctrl)
    assert record.status == "SUCCESS"
    assert record.intent == "TURN_ON"
    assert record.device_id == "LIGHT_01"
    assert reg.get("LIGHT_01").state.power is True
    router.stop()


def test_ai_router_async_execution(sample_setup):
    reg, ctrl = sample_setup
    client = OpenAIClient(api_key="", enabled=True, min_request_interval=0.0)
    router = AIRouter(ai_mode="EVENT", client=client)

    queued = router.process_text_command_async("Turn the living room light on", reg, ctrl)
    assert queued is True

    # Wait briefly for worker thread
    time.sleep(0.3)
    completed = router.check_completed_results()
    assert len(completed) >= 1
    assert completed[0].status == "SUCCESS"
    assert reg.get("LIGHT_01").state.power is True
    router.stop()


def test_emergency_stop_bypasses_ai(sample_setup):
    reg, ctrl = sample_setup
    # Set light on first
    reg.get("LIGHT_01").state.power = True
    assert reg.get("LIGHT_01").state.power is True

    # Direct FIST Emergency stop executed locally by DeviceController (no AI involved)
    ctrl.handle_action(GestureAction.EMERGENCY_STOP)
    assert reg.get("LIGHT_01").state.power is False
