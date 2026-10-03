"""
intent/logging_utils.py

Phase 10: Structured (JSON) logging for the intent pipeline.
Raw input is truncated and long digit runs are masked; logging raw input can
be disabled entirely via INTENT_LOG_RAW_INPUT=false.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger("visioncontrol.intent")

MAX_LOGGED_INPUT = 120


def redact(text: Any) -> str:
    s = str(text or "")
    s = re.sub(r"\d{6,}", lambda m: "#" * len(m.group(0)), s)
    return s if len(s) <= MAX_LOGGED_INPUT else s[:MAX_LOGGED_INPUT] + "…"


def log_event(stage: str, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"stage": stage, **fields}
    try:
        logger.log(level, json.dumps(payload, default=str, ensure_ascii=False))
    except Exception:  # logging must never break the pipeline
        pass
