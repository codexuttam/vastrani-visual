"""
ai/openai_client.py

Phase 9: Isolated OpenAI API Client wrapper.

Responsibilities:
    - Manage official OpenAI Python SDK client
    - Structured Response parsing via Pydantic AIIntent schema
    - Enforce request timeout and minimum request interval rate limiting
    - Graceful handling of API errors, timeouts, rate limits, and offline mode
"""

import time
import json
import logging
from typing import Optional, Tuple

try:
    import openai
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_SDK_AVAILABLE = False

from ai.schemas import AIIntent
from ai.prompts import build_system_prompt
from devices.registry import DeviceRegistry

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Isolated client for OpenAI API interaction.
    Does NOT leak SDK objects outside the ai module.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        timeout: float = 10.0,
        enabled: bool = True,
        min_request_interval: float = 1.0,
    ):
        self.api_key: str = api_key
        self.model: str = model
        self.timeout: float = timeout
        self.enabled: bool = enabled
        self.min_request_interval: float = min_request_interval

        self._client: Optional[object] = None
        self._last_request_time: float = 0.0

        if OPENAI_SDK_AVAILABLE and self.enabled and self.api_key:
            try:
                self._client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                print(f"[OpenAIClient] Error initializing OpenAI client: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None and self.enabled

    def interpret(
        self,
        text_input: str,
        registry: Optional[DeviceRegistry] = None,
    ) -> Tuple[Optional[AIIntent], str, Optional[str]]:
        """
        Interprets natural language text input into a structured AIIntent.

        Returns:
            (ai_intent: Optional[AIIntent], status: str, error_message: Optional[str])
            Status values: "SUCCESS", "DISABLED", "RATE_LIMITED", "TIMEOUT", "ERROR"
        """
        if not self.enabled:
            return None, "DISABLED", "OpenAI integration is disabled."

        # Rate limiting check — enforced for both real API and mock fallback
        now = time.time()
        if now - self._last_request_time < self.min_request_interval:
            return None, "RATE_LIMITED", "Request ignored due to rate limiting interval."
        self._last_request_time = now

        if not self.is_configured:
            # Fallback mock for developer testing without API key
            return self._mock_fallback_interpretation(text_input, registry)

        system_prompt = build_system_prompt(registry)

        try:
            # Official OpenAI Structured Outputs API
            completion = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                response_format=AIIntent,
                timeout=self.timeout,
            )

            parsed_intent = completion.choices[0].message.parsed
            if parsed_intent:
                return parsed_intent, "SUCCESS", None
            else:
                return None, "ERROR", "Failed to parse structured AIIntent response."

        except getattr(openai, "APITimeoutError", Exception) as e:
            return None, "TIMEOUT", f"OpenAI API request timed out ({self.timeout}s): {e}"
        except Exception as e:
            err_msg = str(e)
            return None, "ERROR", f"OpenAI API Error: {err_msg}"

    def _mock_fallback_interpretation(
        self, text_input: str, registry: Optional[DeviceRegistry]
    ) -> Tuple[Optional[AIIntent], str, Optional[str]]:
        """
        Deterministic offline fallback for testing natural language commands without live API key.
        """
        txt = text_input.lower().strip()

        # Simple semantic rules for offline testing
        if "light" in txt:
            dev_id = "LIGHT_01"
            if "on" in txt:
                return AIIntent(intent="TURN_ON", device_id=dev_id, confidence=0.95, reason="User requested light on."), "SUCCESS", None
            elif "off" in txt:
                return AIIntent(intent="TURN_OFF", device_id=dev_id, confidence=0.95, reason="User requested light off."), "SUCCESS", None
            elif "percent" in txt or "%" in txt or "set" in txt or "level" in txt:
                # Extract number
                import re
                nums = re.findall(r'\d+', txt)
                val = int(nums[0]) if nums else 50
                return AIIntent(intent="SET_LEVEL", device_id=dev_id, value=val, confidence=0.95, reason=f"User set light level to {val}."), "SUCCESS", None
            elif "toggle" in txt:
                return AIIntent(intent="TOGGLE", device_id=dev_id, confidence=0.95, reason="User toggled light."), "SUCCESS", None

        if "fan" in txt:
            dev_id = "FAN_01"
            if "speed" in txt or "set" in txt or "level" in txt:
                import re
                nums = re.findall(r'\d+', txt)
                val = int(nums[0]) if nums else 1
                return AIIntent(intent="SET_LEVEL", device_id=dev_id, value=val, confidence=0.95, reason=f"User set fan speed to {val}."), "SUCCESS", None
            elif "on" in txt:
                return AIIntent(intent="TURN_ON", device_id=dev_id, confidence=0.95, reason="User turned fan on."), "SUCCESS", None
            elif "off" in txt:
                return AIIntent(intent="TURN_OFF", device_id=dev_id, confidence=0.95, reason="User turned fan off."), "SUCCESS", None

        if "servo" in txt or "motor" in txt or "angle" in txt:
            dev_id = "SERVO_01"
            import re
            nums = re.findall(r'\d+', txt)
            val = int(nums[0]) if nums else 90
            return AIIntent(intent="SET_LEVEL", device_id=dev_id, value=val, confidence=0.95, reason=f"User set servo angle to {val}."), "SUCCESS", None

        if "next" in txt:
            return AIIntent(intent="NEXT", confidence=0.95, reason="User requested next."), "SUCCESS", None
        if "previous" in txt or "prev" in txt:
            return AIIntent(intent="PREVIOUS", confidence=0.95, reason="User requested previous."), "SUCCESS", None
        if "stop" in txt:
            return AIIntent(intent="STOP", confidence=0.95, reason="User requested stop."), "SUCCESS", None

        return AIIntent(intent="NONE", confidence=0.1, reason="Could not parse intent."), "SUCCESS", None
