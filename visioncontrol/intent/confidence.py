"""
intent/confidence.py

Phase 10: Confidence policy — the ONLY place thresholds are interpreted.
Threshold values come from config.Config (INTENT_AUTO_EXECUTE_THRESHOLD /
INTENT_CONFIRM_THRESHOLD) and are injected here.
"""

from dataclasses import dataclass

DEFAULT_AUTO_EXECUTE_THRESHOLD = 0.85
DEFAULT_CONFIRM_THRESHOLD = 0.60


class ConfidenceDecision:
    EXECUTE  = "execute"
    CONFIRM  = "confirm"
    REPHRASE = "rephrase"


@dataclass(frozen=True)
class ConfidencePolicy:
    auto_execute_threshold: float = DEFAULT_AUTO_EXECUTE_THRESHOLD
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD

    def __post_init__(self):
        if not 0.0 <= self.confirm_threshold <= self.auto_execute_threshold <= 1.0:
            raise ValueError("Require 0 <= confirm_threshold <= auto_execute_threshold <= 1.")

    @classmethod
    def from_config(cls, cfg) -> "ConfidencePolicy":
        return cls(
            auto_execute_threshold=getattr(cfg, "INTENT_AUTO_EXECUTE_THRESHOLD", DEFAULT_AUTO_EXECUTE_THRESHOLD),
            confirm_threshold=getattr(cfg, "INTENT_CONFIRM_THRESHOLD", DEFAULT_CONFIRM_THRESHOLD),
        )

    def decide(self, confidence: float) -> str:
        if confidence >= self.auto_execute_threshold:
            return ConfidenceDecision.EXECUTE
        if confidence >= self.confirm_threshold:
            return ConfidenceDecision.CONFIRM
        return ConfidenceDecision.REPHRASE
