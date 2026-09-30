"""
ai/prompts.py

Phase 9: OpenAI System Prompt Templates.

Constructs concise prompts supplying device context and capabilities.
"""

from typing import List, Optional
from devices.registry import DeviceRegistry


SYSTEM_PROMPT_TEMPLATE = """You are the intent interpretation layer for VisionControl, an AI-powered touchless human-computer interface.

You do not directly control hardware. Your job is to interpret the user's natural language input or event context into a single structured intent.

Allowed intents MUST be one of:
- NONE
- TURN_ON
- TURN_OFF
- TOGGLE
- SET_LEVEL
- NEXT
- PREVIOUS
- STOP
- SELECT
- CONFIRM

Registered Devices & Capabilities:
{device_context}

STRICT RULES:
1. Never invent device IDs. Only use device IDs listed above.
2. Never output arbitrary or unsupported commands.
3. For SET_LEVEL intent, ensure 'value' is within allowed range for that device type.
4. Set confidence between 0.0 and 1.0 based on how clearly the request maps to the intent.
5. Return JSON matching the requested schema.
"""


def build_system_prompt(registry: Optional[DeviceRegistry] = None) -> str:
    """
    Constructs the system prompt injecting registered devices and bounds.
    """
    if registry is not None:
        devices = registry.list_devices()
        lines = []
        for dev in devices:
            s = dev.state
            lines.append(
                f"- {s.device_id}: {s.name} (Type: {s.device_type.value}, Allowed Range: [{s.min_level}, {s.max_level}]{s.level_unit})"
            )
        ctx = "\n".join(lines)
    else:
        ctx = (
            "- LIGHT_01: Living Room Light (Type: LIGHT, Range: [0, 100]%)\n"
            "- FAN_01: Ceiling Fan (Type: FAN, Range: [0, 3]/3)\n"
            "- MUSIC_01: Music Player (Type: MUSIC, Range: [0, 100]%)\n"
            "- SERVO_01: Servo Motor (Type: SERVO, Range: [0, 180]°)"
        )

    return SYSTEM_PROMPT_TEMPLATE.format(device_context=ctx)
