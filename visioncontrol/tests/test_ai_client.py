"""
tests/test_ai_client.py

Phase 9 Unit Tests: OpenAIClient
"""

import time
import pytest
from ai.openai_client import OpenAIClient
from devices.registry import DeviceRegistry
from devices.models import DeviceState, DeviceType


@pytest.fixture
def sample_registry():
    reg = DeviceRegistry()
    reg.register(DeviceState(
        device_id="LIGHT_01",
        name="Living Room Light",
        device_type=DeviceType.LIGHT
    ))
    return reg


def test_openai_client_disabled():
    client = OpenAIClient(enabled=False)
    intent, status, err = client.interpret("Turn on light", None)
    assert intent is None
    assert status == "DISABLED"


def test_openai_client_mock_fallback(sample_registry):
    # Blank key uses mock fallback
    client = OpenAIClient(api_key="", enabled=True)
    intent, status, err = client.interpret("Turn the living room light on", sample_registry)
    assert intent is not None
    assert intent.intent == "TURN_ON"
    assert intent.device_id == "LIGHT_01"
    assert status == "SUCCESS"


def test_openai_client_rate_limiting(sample_registry):
    client = OpenAIClient(api_key="", enabled=True, min_request_interval=0.5)
    # First call
    client.interpret("Turn on light", sample_registry)
    # Immediate second call should hit rate limiting
    intent2, status2, err2 = client.interpret("Turn off light", sample_registry)
    assert intent2 is None
    assert status2 == "RATE_LIMITED"
