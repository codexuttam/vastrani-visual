"""
intent/engine.py

Phase 10: Natural-Language Intent Engine (orchestrator).

    raw text / transcript
        → normalize            (intent.normalizer)
        → parse                (LLM provider → fallback parser on failure)
        → entities             (inside parser; intent.entities)
        → context resolution   ("turn it on" after "turn off the fan")
        → validate + safety    (intent.validator)
        → confidence policy    (intent.confidence)
        → confirmation flow    (pending command, yes / no)
        → ActionRouterAdapter  (existing DeviceController / ARController)

The engine itself never touches devices: execution is delegated to the
injected adapter, and only for validated commands.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from intent import messages as M
from intent.action_adapter import ActionRouterAdapter, ExecutionResult
from intent.confidence import ConfidenceDecision, ConfidencePolicy
from intent.fallback_parser import FallbackParser
from intent.logging_utils import log_event, redact
from intent.normalizer import normalize
from intent.providers import (
    IntentProvider, ProviderError, ProviderTimeout, ProviderUnavailable, create_provider,
)
from intent.schema import (
    MAX_RAW_INPUT_LENGTH, PRONOUNS, IntentType, SchemaError, StructuredCommand,
)
from intent.validator import IntentValidator, ValidationResult

CONTEXT_RESOLVED_CONFIDENCE = 0.80   # pronoun resolved from context → confirm first

_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|confirm|continue|go ahead|do it|proceed|affirmative)\b")
_NO = re.compile(r"^(no|nope|cancel|stop|never ?mind|abort|don't|do not|negative)\b")


class IntentStatus:
    READY               = "READY"                # parsed + validated, not executed (parse-only)
    EXECUTED            = "EXECUTED"
    NEEDS_CONFIRMATION  = "NEEDS_CONFIRMATION"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REJECTED            = "REJECTED"
    CANCELLED           = "CANCELLED"
    EXECUTION_FAILED    = "EXECUTION_FAILED"
    ERROR               = "ERROR"

    SUCCESSFUL = frozenset({READY, EXECUTED, NEEDS_CONFIRMATION, CANCELLED})


@dataclass
class IntentResult:
    status: str
    message: str
    raw_input: str = ""
    normalized_input: str = ""
    command: Optional[StructuredCommand] = None
    source: str = "none"                      # "llm" | "fallback" | "none"
    validation_code: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    execution: Optional[ExecutionResult] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.status in IntentStatus.SUCCESSFUL

    @property
    def requires_confirmation(self) -> bool:
        return self.status == IntentStatus.NEEDS_CONFIRMATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "intent": self.command.to_dict(include_raw=False) if self.command else None,
            "source": self.source,
            "validation_code": self.validation_code,
            "errors": list(self.errors),
            "execution": self.execution.to_dict() if self.execution else None,
        }


@dataclass
class ConversationContext:
    """Short-term memory used for reference resolution (extensible in Phase 11)."""
    last_device: Optional[str] = None
    pending: Optional[StructuredCommand] = None
    pending_since: float = 0.0
    history: List[str] = field(default_factory=list)

    def as_prompt_context(self) -> Dict[str, Any]:
        return {"last_device": self.last_device}


class IntentEngine:
    def __init__(
        self,
        provider: Optional[IntentProvider] = None,
        fallback: Optional[FallbackParser] = None,
        validator: Optional[IntentValidator] = None,
        policy: Optional[ConfidencePolicy] = None,
        executor: Optional[ActionRouterAdapter] = None,
        provider_timeout: float = 5.0,
        confirmation_timeout: float = 30.0,
        log_raw_input: bool = True,
    ):
        self.provider = provider
        self.fallback = fallback or FallbackParser()
        self.validator = validator or IntentValidator()
        self.policy = policy or ConfidencePolicy()
        self.executor = executor
        self.provider_timeout = provider_timeout
        self.confirmation_timeout = confirmation_timeout
        self.log_raw_input = log_raw_input
        self.context = ConversationContext()
        self.last_result: Optional[IntentResult] = None
        self._lock = threading.RLock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="intent-llm")

    @classmethod
    def from_config(cls, cfg, device_controller=None, ar_controller=None) -> "IntentEngine":
        return cls(
            provider=create_provider(getattr(cfg, "INTENT_PROVIDER", "auto"), cfg),
            policy=ConfidencePolicy.from_config(cfg),
            executor=ActionRouterAdapter(device_controller, ar_controller),
            provider_timeout=getattr(cfg, "INTENT_PROVIDER_TIMEOUT", 5.0),
            confirmation_timeout=getattr(cfg, "INTENT_CONFIRMATION_TIMEOUT", 30.0),
            log_raw_input=getattr(cfg, "INTENT_LOG_RAW_INPUT", True),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def registry(self):
        ctrl = getattr(self.executor, "device_controller", None)
        return getattr(ctrl, "registry", None)

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider else "fallback"

    @property
    def has_pending(self) -> bool:
        return self.context.pending is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, text: Any) -> IntentResult:
        """Parse + validate + decide. Never executes and never changes pending state."""
        try:
            return self._finish(self._parse(text))
        except Exception as e:  # last-resort guard: no stack traces to users
            log_event("error", error=type(e).__name__)
            return self._finish(IntentResult(IntentStatus.ERROR, M.INTERNAL_ERROR_MESSAGE,
                                             raw_input=str(text or "")[:MAX_RAW_INPUT_LENGTH]))

    def handle(self, text: Any) -> IntentResult:
        """Full pipeline: confirmation replies, parsing, and execution of safe commands."""
        with self._lock:
            try:
                if self.has_pending:
                    reply = normalize(text if isinstance(text, str) else "").text
                    if _YES.match(reply):
                        return self.confirm()
                    if _NO.match(reply):
                        return self.cancel()
                    # Any other input abandons the pending command.
                    self._clear_pending()

                result = self._parse(text)
                if result.status == IntentStatus.READY:
                    result = self._execute(result)
                elif result.status == IntentStatus.NEEDS_CONFIRMATION:
                    self.context.pending = result.command
                    self.context.pending_since = time.time()
                return self._finish(result)
            except Exception as e:
                log_event("error", error=type(e).__name__)
                return self._finish(IntentResult(IntentStatus.ERROR, M.INTERNAL_ERROR_MESSAGE))

    def confirm(self) -> IntentResult:
        with self._lock:
            cmd = self.context.pending
            if cmd is None:
                return self._finish(IntentResult(IntentStatus.REJECTED, M.NOTHING_PENDING_MESSAGE))
            expired = time.time() - self.context.pending_since > self.confirmation_timeout
            self._clear_pending()
            if expired:
                return self._finish(IntentResult(IntentStatus.CANCELLED, M.EXPIRED_MESSAGE, command=cmd))
            # Re-validate: device registry may have changed since the prompt.
            v = self.validator.validate(cmd, self.registry)
            if not v.valid:
                return self._finish(IntentResult(IntentStatus.REJECTED, M.rejection_message(v), command=cmd,
                                                 validation_code=v.code, errors=v.errors))
            log_event("confirmation", decision="confirmed", intent=cmd.intent)
            res = IntentResult(IntentStatus.READY, "", raw_input=cmd.raw_input, command=cmd, source="confirmation")
            return self._finish(self._execute(res))

    def cancel(self) -> IntentResult:
        with self._lock:
            cmd = self.context.pending
            self._clear_pending()
            if cmd is None:
                return self._finish(IntentResult(IntentStatus.REJECTED, M.NOTHING_PENDING_MESSAGE))
            log_event("confirmation", decision="cancelled", intent=cmd.intent)
            return self._finish(IntentResult(IntentStatus.CANCELLED, M.CANCELLED_MESSAGE, command=cmd))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    # ── Pipeline internals ────────────────────────────────────────────────────

    def _parse(self, text: Any) -> IntentResult:
        raw = text if isinstance(text, str) else ""
        raw = raw[:MAX_RAW_INPUT_LENGTH]
        norm = normalize(raw)
        log_event("input", raw_input=redact(raw) if self.log_raw_input else "<hidden>",
                  normalized=redact(norm.text) if self.log_raw_input else "<hidden>")

        if norm.is_empty:
            return IntentResult(IntentStatus.NEEDS_CLARIFICATION, M.EMPTY_INPUT_MESSAGE,
                                raw_input=raw, validation_code="EMPTY_INPUT")

        cmd, source, provider_error = self._run_parsers(raw, norm)
        cmd.raw_input = raw
        self._resolve_references(cmd)

        log_event("parsed", source=source, intent=cmd.intent,
                  actions=[c.action for c in cmd.children()],
                  entities=[c.entities for c in cmd.children()],
                  confidence=round(cmd.confidence, 3), provider_error=provider_error)

        base = dict(raw_input=raw, normalized_input=norm.text, command=cmd, source=source)
        v = self.validator.validate(cmd, self.registry)
        log_event("validation", valid=v.valid, code=v.code, errors=v.errors)

        if not v.valid:
            if v.needs_clarification:
                return IntentResult(IntentStatus.NEEDS_CLARIFICATION, M.clarification_message(cmd, v),
                                    validation_code=v.code, errors=v.errors, **base)
            return IntentResult(IntentStatus.REJECTED, M.rejection_message(v),
                                validation_code=v.code, errors=v.errors, **base)

        decision = self.policy.decide(cmd.confidence)
        needs_confirm = v.requires_confirmation or decision == ConfidenceDecision.CONFIRM
        log_event("confidence", confidence=round(cmd.confidence, 3), decision=decision,
                  requires_confirmation=needs_confirm)

        if decision == ConfidenceDecision.REPHRASE:
            return IntentResult(IntentStatus.NEEDS_CLARIFICATION, M.LOW_CONFIDENCE_MESSAGE,
                                validation_code="LOW_CONFIDENCE", **base)
        if needs_confirm:
            cmd.requires_confirmation = True
            for c in cmd.children():
                c.requires_confirmation = True
            return IntentResult(IntentStatus.NEEDS_CONFIRMATION, M.confirmation_prompt(cmd),
                                validation_code=v.code, **base)
        cmd.requires_confirmation = False
        return IntentResult(IntentStatus.READY, f"Ready to {M.describe(cmd)}.", validation_code=v.code, **base)

    def _run_parsers(self, raw: str, norm):
        """LLM first (if configured), deterministic fallback on any failure."""
        provider_error = None
        if self.provider is not None and self.provider.available:
            try:
                fut = self._pool.submit(self.provider.parse_intent, raw, self.context.as_prompt_context())
                data = fut.result(timeout=self.provider_timeout)
                return StructuredCommand.from_dict(data, raw_input=raw), "llm", None
            except concurrent.futures.TimeoutError:
                provider_error = "timeout"
            except ProviderTimeout:
                provider_error = "timeout"
            except ProviderUnavailable:
                provider_error = "unavailable"
            except SchemaError as e:
                provider_error = f"malformed_response: {e}"
            except ProviderError as e:
                provider_error = f"provider_error: {e}"
            except Exception as e:
                provider_error = f"provider_exception: {type(e).__name__}"
            log_event("provider_fallback", provider=self.provider_name, reason=provider_error)
        return self.fallback.parse(raw, norm), "fallback", provider_error

    def _resolve_references(self, cmd: StructuredCommand) -> None:
        last = self.context.last_device
        if not last:
            return
        changed = False
        for c in cmd.children():
            if c.intent == IntentType.DEVICE_CONTROL.value and c.entities.get("device") in PRONOUNS:
                c.entities["device"] = last
                c.confidence = min(max(c.confidence, CONTEXT_RESOLVED_CONFIDENCE), CONTEXT_RESOLVED_CONFIDENCE)
                changed = True
        if changed and cmd.is_multi:
            cmd.confidence = min(c.confidence for c in cmd.actions)
        elif changed:
            cmd.confidence = min(cmd.confidence, CONTEXT_RESOLVED_CONFIDENCE)

    def _execute(self, result: IntentResult) -> IntentResult:
        cmd = result.command
        if self.executor is None:
            return result  # parse-only engine
        exec_result = self.executor.execute(cmd)
        log_event("execution", success=exec_result.success, message=exec_result.message,
                  devices=exec_result.device_ids)
        if exec_result.success:
            for c in cmd.children():
                dev = c.entities.get("device")
                if dev and dev not in PRONOUNS and dev != "all":
                    self.context.last_device = dev
            result.status = IntentStatus.EXECUTED
            result.message = exec_result.message
        else:
            result.status = IntentStatus.EXECUTION_FAILED
            result.message = f"I couldn't complete that: {exec_result.message}"
        result.execution = exec_result
        return result

    def _clear_pending(self) -> None:
        self.context.pending = None
        self.context.pending_since = 0.0

    def _finish(self, result: IntentResult) -> IntentResult:
        self.last_result = result
        log_event("result", status=result.status, success=result.success,
                  requires_confirmation=result.requires_confirmation)
        return result
