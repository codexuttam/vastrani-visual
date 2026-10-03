"""
intent — Phase 10: Natural-Language Intent Engine.

Text / voice transcript → validated StructuredCommand → existing action layer.
"""

from intent.schema import IntentType, Action, StructuredCommand, SchemaError
from intent.confidence import ConfidencePolicy, ConfidenceDecision
from intent.validator import IntentValidator, ValidationResult, ValidationCode
from intent.fallback_parser import FallbackParser
from intent.providers import (
    IntentProvider, OpenAIIntentProvider, ProviderError, ProviderTimeout,
    ProviderUnavailable, create_provider, register_provider,
)
from intent.action_adapter import ActionRouterAdapter, ExecutionResult
from intent.engine import IntentEngine, IntentResult, IntentStatus
from intent.service import IntentService

__all__ = [
    "IntentType", "Action", "StructuredCommand", "SchemaError",
    "ConfidencePolicy", "ConfidenceDecision",
    "IntentValidator", "ValidationResult", "ValidationCode",
    "FallbackParser",
    "IntentProvider", "OpenAIIntentProvider", "ProviderError", "ProviderTimeout",
    "ProviderUnavailable", "create_provider", "register_provider",
    "ActionRouterAdapter", "ExecutionResult",
    "IntentEngine", "IntentResult", "IntentStatus", "IntentService",
]
