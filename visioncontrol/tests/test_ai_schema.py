"""
tests/test_ai_schema.py

Phase 9 Unit Tests: AIIntent Schema & AIResultRecord Validation
"""

import pytest
from ai.schemas import AIIntent, AIResultRecord, SUPPORTED_INTENTS


def test_ai_intent_creation_valid():
    intent = AIIntent(
        intent="TURN_ON",
        device_id="LIGHT_01",
        value=None,
        confidence=0.95,
        reason="User requested light to turn on."
    )
    assert intent.intent == "TURN_ON"
    assert intent.device_id == "LIGHT_01"
    assert intent.confidence == 0.95


def test_ai_intent_unsupported_rejected():
    with pytest.raises(ValueError, match="Unsupported intent"):
        AIIntent(
            intent="HACK_SYSTEM",
            device_id="LIGHT_01",
            value=None,
            confidence=0.90,
            reason="Malicious intent"
        )


def test_ai_intent_confidence_bounds():
    with pytest.raises(ValueError):
        AIIntent(
            intent="TURN_ON",
            device_id="LIGHT_01",
            value=None,
            confidence=4.5,
            reason="Invalid confidence"
        )



def test_ai_result_record_fields():
    rec = AIResultRecord(
        intent="SET_LEVEL",
        device_id="LIGHT_01",
        value=50,
        confidence=0.88,
        reason="Set level to 50%",
        timestamp=100.0,
        status="SUCCESS",
        input_text="Set light to 50%"
    )
    assert rec.intent == "SET_LEVEL"
    assert rec.value == 50
    assert rec.status == "SUCCESS"
