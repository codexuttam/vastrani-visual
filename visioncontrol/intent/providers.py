"""
intent/providers.py

Phase 10: LLM provider abstraction.

    IntentProvider.parse_intent(text, context) -> dict  (raw candidate command)

Providers only *propose* a command. Their output is always passed through
StructuredCommand.from_dict + IntentValidator before anything happens.
New providers (Gemini, local model, ...) subclass IntentProvider and register
a factory with `register_provider(name, factory)`.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from intent.schema import schema_summary


class ProviderError(Exception):
    """Generic provider failure (network, auth, bad response)."""


class ProviderTimeout(ProviderError):
    """Provider did not answer within the configured timeout."""


class ProviderUnavailable(ProviderError):
    """Provider is not configured (no SDK / no API key / disabled)."""


class IntentProvider(ABC):
    name: str = "base"

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def parse_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a candidate command dict. May raise ProviderError subclasses."""


def build_intent_system_prompt() -> str:
    """System prompt derived from the single-source schema."""
    return (
        "You are the intent parser for Vatsrani Vision, a touchless home-control system.\n"
        "Convert the user's command into ONE JSON object. Never invent intents, actions, "
        "devices or entities outside this schema:\n"
        f"{json.dumps(schema_summary(), indent=1)}\n\n"
        "Single command format:\n"
        '{"intent": str, "action": str, "entities": {..}, "parameters": {}, '
        '"confidence": 0..1, "requires_confirmation": bool}\n'
        "Multiple commands: "
        '{"intent": "multi_action", "actions": [<single command>...], "confidence": 0..1}\n'
        "Rules: use intent 'unknown' with low confidence when unsure; if the user refers to "
        "'it'/'that' without context, set the entity to that pronoun; numeric entities must be "
        "numbers; requests to run programs, delete files or shut down the computer are "
        "intent 'system_command' with action 'run_shell', 'delete_files' or 'shutdown_host'. "
        "Respond with JSON only."
    )


class OpenAIIntentProvider(IntentProvider):
    """OpenAI chat-completions provider using JSON mode (reuses Phase 9 config)."""

    name = "openai"

    def __init__(self, api_key: str = "", model: str = "gpt-4o", timeout: float = 5.0):
        self.model = model
        self.timeout = timeout
        self._client = None
        if api_key:
            try:
                import openai  # optional dependency
                self._client = openai.OpenAI(api_key=api_key, timeout=timeout)
                self._timeout_exc = getattr(openai, "APITimeoutError", TimeoutError)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def parse_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.available:
            raise ProviderUnavailable("OpenAI provider is not configured.")
        messages = [{"role": "system", "content": build_intent_system_prompt()}]
        if context:
            messages.append({"role": "system", "content": f"Context: {json.dumps(context)}"})
        messages.append({"role": "user", "content": text})
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                response_format={"type": "json_object"}, temperature=0, timeout=self.timeout,
            )
            content = resp.choices[0].message.content or ""
        except self._timeout_exc as e:
            raise ProviderTimeout(str(e)) from e
        except Exception as e:
            raise ProviderError(type(e).__name__) from e
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderError("Malformed JSON from provider.") from e


# ─── Provider registry ────────────────────────────────────────────────────────

ProviderFactory = Callable[[Any], Optional[IntentProvider]]
_PROVIDER_FACTORIES: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDER_FACTORIES[name.lower()] = factory


def create_provider(name: str, cfg: Any) -> Optional[IntentProvider]:
    """
    Build the configured provider. 'fallback'/'none' → None (rule-based only).
    'auto' → OpenAI if an API key is configured, else None.
    """
    name = (name or "auto").lower()
    if name in ("fallback", "none", "off", "rules"):
        return None
    if name == "auto":
        provider = _PROVIDER_FACTORIES["openai"](cfg)
        return provider if provider is not None and provider.available else None
    factory = _PROVIDER_FACTORIES.get(name)
    return factory(cfg) if factory else None


register_provider("openai", lambda cfg: OpenAIIntentProvider(
    api_key=getattr(cfg, "OPENAI_API_KEY", "") if getattr(cfg, "OPENAI_ENABLED", True) else "",
    model=getattr(cfg, "OPENAI_MODEL", "gpt-4o"),
    timeout=getattr(cfg, "INTENT_PROVIDER_TIMEOUT", 5.0),
))
