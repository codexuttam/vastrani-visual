"""
ai/schemas.py

Phase 9: Structured AI Intent Schemas & Models.

Defines Pydantic models for strict OpenAI response parsing and validation.
No direct OpenAI client imports.
"""

from dataclasses import dataclass
from typing import Optional, Union, Set
from pydantic import BaseModel, Field, field_validator


SUPPORTED_INTENTS: Set[str] = frozenset({
    "NONE",
    "TURN_ON",
    "TURN_OFF",
    "TOGGLE",
    "SET_LEVEL",
    "NEXT",
    "PREVIOUS",
    "STOP",
    "SELECT",
    "CONFIRM",
})


class AIIntent(BaseModel):
    """
    Structured intent output requested from OpenAI.
    Enforces strict typing and confidence bound validation.
    """
    intent: str = Field(
        ...,
        description="The semantic intent (MUST be one of: NONE, TURN_ON, TURN_OFF, TOGGLE, SET_LEVEL, NEXT, PREVIOUS, STOP, SELECT, CONFIRM)"
    )

    @field_validator('intent')
    @classmethod
    def validate_intent_name(cls, v: str) -> str:
        u = v.upper()
        if u not in SUPPORTED_INTENTS:
            raise ValueError(f"Unsupported intent '{v}'. Must be one of {sorted(list(SUPPORTED_INTENTS))}")
        return u

    device_id: Optional[str] = Field(
        None,
        description="Target registered device ID (e.g. LIGHT_01, FAN_01, MUSIC_01, SERVO_01)"
    )
    value: Optional[Union[int, float]] = Field(
        None,
        description="Numeric level value if intent is SET_LEVEL"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reason: str = Field(
        "",
        description="Brief explanation of why this intent was selected"
    )


@dataclass
class AIResultRecord:
    """Audit record for AI interpretation results."""
    intent: str
    device_id: Optional[str]
    value: Optional[Union[int, float]]
    confidence: float
    reason: str
    timestamp: float
    status: str  # "DISABLED", "READY", "PROCESSING", "SUCCESS", "LOW_CONFIDENCE", "REJECTED", "ERROR", "TIMEOUT"
    error: Optional[str] = None
    input_text: str = ""

    def formatted_time(self) -> str:
        import time
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))
