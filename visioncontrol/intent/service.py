"""
intent/service.py

Phase 10: Service boundary for the intent engine.

VisionControl has no web backend, so the canonical boundary is this Python
service (used by app.py / the terminal console / Phase 11 voice input).
`intent/api.py` exposes the same service over a minimal, optional,
dependency-free local HTTP server:

    POST /api/intent/parse     {"input": "..."}   → parse only, never executes
    POST /api/intent/execute   {"input": "..."}   → full pipeline (may ask to confirm)
    POST /api/intent/confirm   {}                 → execute pending command
    POST /api/intent/cancel    {}                 → drop pending command
    GET  /api/intent/schema                       → supported intents/actions
"""

from typing import Any, Dict

from intent.engine import IntentEngine
from intent.schema import schema_summary


class IntentService:
    def __init__(self, engine: IntentEngine):
        self.engine = engine

    @staticmethod
    def _input(payload: Any):
        if not isinstance(payload, dict):
            return None, {"success": False, "status": "BAD_REQUEST", "message": "Request body must be a JSON object."}
        value = payload.get("input")
        if value is not None and not isinstance(value, str):
            return None, {"success": False, "status": "BAD_REQUEST", "message": "'input' must be a string."}
        return value or "", None

    def parse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text, err = self._input(payload)
        return err or self.engine.parse(text).to_dict()

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text, err = self._input(payload)
        return err or self.engine.handle(text).to_dict()

    def confirm(self, _payload: Any = None) -> Dict[str, Any]:
        return self.engine.confirm().to_dict()

    def cancel(self, _payload: Any = None) -> Dict[str, Any]:
        return self.engine.cancel().to_dict()

    @staticmethod
    def schema() -> Dict[str, Any]:
        return {"success": True, "schema": schema_summary()}
