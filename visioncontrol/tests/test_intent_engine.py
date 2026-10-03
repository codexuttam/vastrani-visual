"""
tests/test_intent_engine.py — Phase 10: Natural-Language Intent Engine.
"""

import json
import threading
import time
import urllib.request

import pytest

from devices.controller import DeviceController
from devices.registry import create_default_registry
from intent import (
    ActionRouterAdapter, ConfidencePolicy, FallbackParser, IntentEngine,
    IntentProvider, IntentService, IntentStatus, IntentValidator, ProviderError,
    ProviderTimeout, SchemaError, StructuredCommand, ValidationCode,
)
from intent.api import IntentAPIServer
from intent.normalizer import normalize
from intent.providers import create_provider
from intent.schema import INTENT_ACTIONS, SUPPORTED_INTENTS


# ─── Fixtures / fakes ─────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    return FallbackParser()


@pytest.fixture
def engine():
    """Parse-only engine (no execution)."""
    e = IntentEngine()
    yield e
    e.shutdown()


@pytest.fixture
def live():
    """Engine wired to a real virtual DeviceController (existing action layer)."""
    ctrl = DeviceController(registry=create_default_registry())
    e = IntentEngine(executor=ActionRouterAdapter(ctrl))
    yield e, ctrl
    e.shutdown()


def dev(ctrl, device_id):
    return ctrl.registry.get(device_id).state


class StaticProvider(IntentProvider):
    name = "static"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def parse_intent(self, text, context=None):
        self.calls += 1
        return self.response


class SlowProvider(IntentProvider):
    name = "slow"

    def parse_intent(self, text, context=None):
        time.sleep(1.0)
        return {"intent": "device_control", "action": "turn_off", "entities": {"device": "fan"}, "confidence": 0.99}


class TimeoutProvider(IntentProvider):
    name = "timeout"

    def parse_intent(self, text, context=None):
        raise ProviderTimeout("simulated")


class BrokenProvider(IntentProvider):
    name = "broken"

    def parse_intent(self, text, context=None):
        raise ProviderError("simulated outage")


# ─── Schema ───────────────────────────────────────────────────────────────────

def test_schema_has_all_required_intents():
    for name in ("device_control", "display_control", "media_control", "navigation",
                 "gesture_command", "face_command", "system_command", "multi_action", "unknown"):
        assert name in SUPPORTED_INTENTS
        assert name in INTENT_ACTIONS


def test_structured_command_round_trip():
    d = {"intent": "device_control", "action": "turn_on", "entities": {"device": "fan", "location": "bedroom"},
         "parameters": {}, "confidence": 0.95, "requires_confirmation": False}
    cmd = StructuredCommand.from_dict(d, raw_input="turn on the bedroom fan")
    out = cmd.to_dict()
    assert out["raw_input"] == "turn on the bedroom fan"
    assert {k: out[k] for k in d} == d


@pytest.mark.parametrize("bad", [
    "not a dict", [], {}, {"intent": ""}, {"intent": "device_control", "confidence": "high"},
    {"intent": "device_control", "confidence": 1.5}, {"intent": "device_control", "entities": [], "confidence": 0.9},
    {"intent": "multi_action", "actions": [], "confidence": 0.9},
    {"intent": "multi_action", "confidence": 0.9, "actions": [{"intent": "multi_action", "actions": [{}]}]},
])
def test_malformed_commands_rejected(bad):
    with pytest.raises(SchemaError):
        StructuredCommand.from_dict(bad)


# ─── Normalizer ───────────────────────────────────────────────────────────────

def test_normalizer_strips_fillers_and_articles():
    n = normalize("Could you PLEASE switch the fan on?")
    assert n.text == "switch fan on"


def test_normalizer_number_words():
    assert "55%" in normalize("set the light to fifty five percent").text


# ─── Basic commands ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,intent,action,entities", [
    ("turn on fan", "device_control", "turn_on", {"device": "fan"}),
    ("turn off fan", "device_control", "turn_off", {"device": "fan"}),
    ("increase brightness", "display_control", "increase_brightness", {}),
    ("decrease brightness", "display_control", "decrease_brightness", {}),
    ("play music", "media_control", "play", {"media": "music"}),
    ("pause music", "media_control", "pause", {"media": "music"}),
])
def test_basic_commands(engine, text, intent, action, entities):
    r = engine.parse(text)
    assert r.status == IntentStatus.READY
    assert r.command.intent == intent and r.command.action == action
    assert r.command.entities == entities
    assert r.command.confidence >= 0.85
    assert r.command.requires_confirmation is False


# ─── Normalization of variations ─────────────────────────────────────────────

@pytest.mark.parametrize("text", ["turn on the fan", "switch on the fan", "start the fan", "fan on"])
def test_turn_on_variations(engine, text):
    c = engine.parse(text).command
    assert (c.intent, c.action, c.entities.get("device")) == ("device_control", "turn_on", "fan")


@pytest.mark.parametrize("text", ["turn the lights off", "switch off the lights", "lights off"])
def test_turn_off_variations(engine, text):
    c = engine.parse(text).command
    assert (c.intent, c.action, c.entities.get("device")) == ("device_control", "turn_off", "light")


def test_natural_variations(engine):
    c = engine.parse("could you please switch the fan on").command
    assert (c.action, c.entities["device"]) == ("turn_on", "fan")
    c = engine.parse("can you turn the lights off").command
    assert (c.action, c.entities["device"]) == ("turn_off", "light")
    r = engine.parse("make the screen brighter")
    assert r.status == IntentStatus.READY
    assert (r.command.intent, r.command.action) == ("display_control", "increase_brightness")
    assert r.command.entities["target"] == "screen"


# ─── Entity extraction ────────────────────────────────────────────────────────

def test_entity_extraction_location_value_amount(parser):
    c = parser.parse("turn on the bedroom fan")
    assert c.entities == {"device": "fan", "location": "bedroom"}
    c = parser.parse("set the fan to speed 2")
    assert (c.action, c.entities["value"]) == ("set_level", 2)
    c = parser.parse("volume up by 20")
    assert (c.action, c.entities["amount"]) == ("volume_up", 20)
    c = parser.parse("turn off the light for 10 minutes")
    assert c.entities["duration"] == 600


def test_entity_extractor_is_extensible():
    p = FallbackParser()
    p.extractor.register("person", lambda t: "alice" if "alice" in t else None)
    assert p.extractor.extract("turn on alice fan")["person"] == "alice"


# ─── Ambiguous commands ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["turn it on", "make it brighter", "do that"])
def test_ambiguous_commands_request_clarification(engine, text):
    r = engine.parse(text)
    assert r.status == IntentStatus.NEEDS_CLARIFICATION
    assert r.validation_code in (ValidationCode.AMBIGUOUS, ValidationCode.MISSING_ENTITY)
    assert "?" in r.message


def test_missing_device_requests_clarification(engine):
    r = engine.parse("switch on")
    assert r.status == IntentStatus.NEEDS_CLARIFICATION
    assert r.validation_code == ValidationCode.MISSING_ENTITY
    assert "Which device" in r.message


def test_pronoun_resolved_from_context_requires_confirmation(live):
    e, ctrl = live
    assert e.handle("turn on the fan").status == IntentStatus.EXECUTED
    r = e.handle("turn it off")
    assert r.status == IntentStatus.NEEDS_CONFIRMATION
    assert r.command.entities["device"] == "fan"
    assert dev(ctrl, "FAN_01").power is True
    assert e.handle("yes").status == IntentStatus.EXECUTED
    assert dev(ctrl, "FAN_01").power is False


# ─── Multi-action ─────────────────────────────────────────────────────────────

def test_multi_action_parse(engine):
    r = engine.parse("turn on the fan and lower brightness")
    c = r.command
    assert r.status == IntentStatus.READY
    assert c.intent == "multi_action" and len(c.actions) == 2
    assert (c.actions[0].intent, c.actions[0].action, c.actions[0].entities) == \
        ("device_control", "turn_on", {"device": "fan"})
    assert (c.actions[1].intent, c.actions[1].action) == ("display_control", "decrease_brightness")
    assert c.confidence >= 0.85


def test_multi_action_inherits_verb(parser):
    c = parser.parse("turn on the fan and the light")
    assert [a.entities["device"] for a in c.actions] == ["fan", "light"]
    assert all(a.action == "turn_on" for a in c.actions)


def test_multi_action_children_validated_independently(engine):
    r = engine.parse("turn on the fan and open spotify")
    assert r.status == IntentStatus.REJECTED
    assert r.validation_code == ValidationCode.PERMISSION_DENIED


def test_multi_action_executes_via_existing_controller(live):
    e, ctrl = live
    ctrl.set_level("LIGHT_01", 60)
    r = e.handle("turn on the fan and lower the brightness")
    assert r.status == IntentStatus.EXECUTED
    assert dev(ctrl, "FAN_01").power is True
    assert dev(ctrl, "LIGHT_01").level == 40


# ─── Invalid ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["asdfgh", "do something", "", "   ", None])
def test_invalid_inputs_not_executed(live, text):
    e, ctrl = live
    before = [(d.device_id, d.state.power, d.state.level) for d in ctrl.registry.list_devices()]
    r = e.handle(text)
    assert r.status == IntentStatus.NEEDS_CLARIFICATION
    assert r.execution is None
    assert before == [(d.device_id, d.state.power, d.state.level) for d in ctrl.registry.list_devices()]


@pytest.mark.parametrize("text", ["open spotify", "delete all files", "shutdown computer", "run shell command"])
def test_out_of_scope_commands_blocked(engine, text):
    r = engine.parse(text)
    assert r.status == IntentStatus.REJECTED
    assert r.validation_code == ValidationCode.PERMISSION_DENIED


def test_out_of_range_value_rejected(live):
    e, _ = live
    r = e.handle("set the light to 150")
    assert r.status == IntentStatus.REJECTED and r.validation_code == ValidationCode.OUT_OF_RANGE


# ─── Validator (LLM-shaped output) ────────────────────────────────────────────

@pytest.mark.parametrize("d,code", [
    ({"intent": "hack_mainframe", "action": "x", "confidence": 0.99}, ValidationCode.UNSUPPORTED_INTENT),
    ({"intent": "device_control", "action": "explode", "entities": {"device": "fan"}, "confidence": 0.99},
     ValidationCode.UNSUPPORTED_ACTION),
    ({"intent": "device_control", "action": "turn_on", "confidence": 0.99}, ValidationCode.MISSING_ENTITY),
    ({"intent": "device_control", "action": "set_level", "entities": {"device": "fan", "value": "fast"},
      "confidence": 0.99}, ValidationCode.INVALID_PARAMETER),
    ({"intent": "device_control", "action": "set_level", "entities": {"device": "fan", "value": 9},
      "confidence": 0.99}, ValidationCode.OUT_OF_RANGE),
    ({"intent": "device_control", "action": "turn_on", "entities": {"device": "ac"}, "confidence": 0.99},
     ValidationCode.UNSUPPORTED_DEVICE),
    ({"intent": "device_control", "action": "turn_on", "entities": {"device": "fan", "password": "x"},
      "confidence": 0.99}, ValidationCode.INVALID_ENTITY),
    ({"intent": "system_command", "action": "run_shell", "confidence": 0.99}, ValidationCode.PERMISSION_DENIED),
])
def test_validator_rejects_unsafe_llm_output(d, code):
    v = IntentValidator().validate(StructuredCommand.from_dict(d), create_default_registry())
    assert not v.valid and v.code == code


# ─── Confidence & confirmation ────────────────────────────────────────────────

def test_confidence_policy_defaults():
    p = ConfidencePolicy()
    assert p.decide(0.95) == "execute"
    assert p.decide(0.85) == "execute"
    assert p.decide(0.84) == "confirm"
    assert p.decide(0.60) == "confirm"
    assert p.decide(0.59) == "rephrase"


def test_confidence_policy_invalid_config():
    with pytest.raises(ValueError):
        ConfidencePolicy(auto_execute_threshold=0.5, confirm_threshold=0.7)


def test_confidence_policy_from_config():
    class Cfg:
        INTENT_AUTO_EXECUTE_THRESHOLD = 0.9
        INTENT_CONFIRM_THRESHOLD = 0.5
    p = ConfidencePolicy.from_config(Cfg)
    assert p.decide(0.88) == "confirm"


@pytest.mark.parametrize("conf,status", [
    (0.95, IntentStatus.READY), (0.70, IntentStatus.NEEDS_CONFIRMATION), (0.40, IntentStatus.NEEDS_CLARIFICATION),
])
def test_engine_applies_confidence_bands(conf, status):
    e = IntentEngine(provider=StaticProvider(
        {"intent": "device_control", "action": "turn_on", "entities": {"device": "fan"}, "confidence": conf}))
    assert e.parse("whatever").status == status
    e.shutdown()


def test_turn_off_everything_requires_confirmation(live):
    e, ctrl = live
    ctrl.power_on("FAN_01")
    r = e.handle("Turn off everything.")
    assert r.status == IntentStatus.NEEDS_CONFIRMATION
    assert r.message == "I understood that you want to turn off all connected devices. Should I continue?"
    assert dev(ctrl, "FAN_01").power is True          # nothing executed yet
    r = e.handle("yes")
    assert r.status == IntentStatus.EXECUTED
    assert not any(d.state.power for d in ctrl.registry.list_devices())


def test_confirmation_cancel(live):
    e, ctrl = live
    ctrl.power_on("FAN_01")
    e.handle("turn off everything")
    assert e.handle("no").status == IntentStatus.CANCELLED
    assert dev(ctrl, "FAN_01").power is True
    assert not e.has_pending


def test_confirmation_expires(live):
    e, ctrl = live
    e.confirmation_timeout = 0.0
    ctrl.power_on("FAN_01")
    e.handle("turn off everything")
    time.sleep(0.01)
    assert e.confirm().status == IntentStatus.CANCELLED
    assert dev(ctrl, "FAN_01").power is True


def test_confirm_without_pending(live):
    e, _ = live
    assert e.confirm().status == IntentStatus.REJECTED


def test_new_command_abandons_pending(live):
    e, ctrl = live
    e.handle("turn off everything")
    assert e.handle("turn on the fan").status == IntentStatus.EXECUTED
    assert not e.has_pending


# ─── Provider abstraction & fallback ──────────────────────────────────────────

def test_llm_provider_output_used_when_valid():
    p = StaticProvider({"intent": "media_control", "action": "play", "entities": {}, "confidence": 0.9})
    e = IntentEngine(provider=p)
    r = e.parse("put on some tunes")
    assert r.source == "llm" and r.command.action == "play" and p.calls == 1
    e.shutdown()


def test_provider_timeout_falls_back_to_rules():
    e = IntentEngine(provider=TimeoutProvider())
    r = e.parse("turn on fan")
    assert r.source == "fallback" and r.status == IntentStatus.READY
    assert r.command.action == "turn_on"
    e.shutdown()


def test_hanging_provider_is_timed_out_and_falls_back():
    e = IntentEngine(provider=SlowProvider(), provider_timeout=0.1)
    t0 = time.time()
    r = e.parse("turn on fan")
    assert time.time() - t0 < 0.8
    assert r.source == "fallback" and r.command.action == "turn_on"
    e.shutdown()


def test_provider_failure_falls_back():
    e = IntentEngine(provider=BrokenProvider())
    r = e.parse("pause music")
    assert r.source == "fallback" and r.command.action == "pause"
    e.shutdown()


def test_malformed_provider_response_falls_back():
    e = IntentEngine(provider=StaticProvider({"intent": 42, "confidence": "lots"}))
    r = e.parse("decrease brightness")
    assert r.source == "fallback" and r.command.action == "decrease_brightness"
    e.shutdown()


def test_unsafe_provider_output_never_executes(live):
    _, ctrl = live
    e = IntentEngine(provider=StaticProvider({"intent": "system_command", "action": "delete_files",
                                               "confidence": 1.0}), executor=ActionRouterAdapter(ctrl))
    r = e.handle("clean up my disk")
    assert r.status == IntentStatus.REJECTED and r.execution is None
    e.shutdown()


def test_create_provider_without_key_uses_fallback():
    class Cfg:
        OPENAI_API_KEY = ""
        OPENAI_ENABLED = True
    assert create_provider("auto", Cfg) is None
    assert create_provider("fallback", Cfg) is None


# ─── Action router integration ────────────────────────────────────────────────

def test_execution_through_existing_controller(live):
    e, ctrl = live
    assert e.handle("turn on the bedroom fan").status == IntentStatus.EXECUTED
    assert dev(ctrl, "FAN_01").power is True
    assert e.handle("play music").status == IntentStatus.EXECUTED
    assert dev(ctrl, "MUSIC_01").power is True
    assert e.handle("set the fan to speed 2").status == IntentStatus.EXECUTED
    assert dev(ctrl, "FAN_01").level == 2
    # executions are recorded in the existing command history
    assert any(rec.device_id == "FAN_01" for rec in ctrl.command_history())


def test_emergency_stop_via_text(live):
    e, ctrl = live
    ctrl.power_on("LIGHT_01")
    assert e.handle("emergency stop").status == IntentStatus.EXECUTED
    assert not any(d.state.power for d in ctrl.registry.list_devices())


def test_navigation_uses_gesture_action_path(live):
    e, ctrl = live
    first = ctrl.registry.current().device_id
    assert e.handle("next device").status == IntentStatus.EXECUTED
    assert ctrl.registry.current().device_id != first


def test_face_command_without_ar_fails_gracefully(live):
    e, _ = live
    r = e.handle("show emoji")
    assert r.status == IntentStatus.EXECUTION_FAILED
    assert "Traceback" not in r.message


def test_execution_exception_is_contained():
    class ExplodingCtrl:
        @property
        def registry(self):
            raise RuntimeError("boom secret detail")
    e = IntentEngine(executor=ActionRouterAdapter(ExplodingCtrl()))
    r = e.handle("turn on fan")
    assert r.status in (IntentStatus.EXECUTION_FAILED, IntentStatus.ERROR)
    assert "boom" not in r.message
    e.shutdown()


# ─── Service / API boundary ───────────────────────────────────────────────────

def test_service_parse_response_shape():
    svc = IntentService(IntentEngine())
    resp = svc.parse({"input": "turn on the bedroom fan"})
    assert resp["success"] is True
    assert resp["intent"] == {"intent": "device_control", "action": "turn_on",
                              "entities": {"device": "fan", "location": "bedroom"},
                              "parameters": {}, "confidence": 0.95, "requires_confirmation": False}
    assert svc.parse({"input": 5})["status"] == "BAD_REQUEST"
    assert svc.parse("nope")["status"] == "BAD_REQUEST"


def test_http_api_parse_endpoint():
    svc = IntentService(IntentEngine())
    server = IntentAPIServer(svc, port=0).start()
    host, port = server.address
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/api/intent/parse",
            data=json.dumps({"input": "turn on the bedroom fan"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        body = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert body["success"] is True and body["intent"]["action"] == "turn_on"

        bad = urllib.request.Request(f"http://{host}:{port}/api/intent/parse", data=b"{oops",
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=5)
        assert ei.value.code == 400
    finally:
        server.stop()


def test_engine_is_thread_safe_for_concurrent_handles(live):
    e, ctrl = live
    threads = [threading.Thread(target=e.handle, args=("toggle the fan",)) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert dev(ctrl, "FAN_01").power is False   # even number of toggles


# ─── UI ───────────────────────────────────────────────────────────────────────

def test_hud_renders_intent_panel_states(live):
    import numpy as np
    from ui.hud import HUD
    e, _ = live
    hud = HUD()
    for text in [None, "turn on the fan", "turn off everything", "asdfgh", "open spotify"]:
        res = e.handle(text) if text is not None else None
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = hud.render(frame, 30.0, intent_result=res, intent_pending=e.has_pending,
                         intent_provider=e.provider_name, debug_intent=True)
        assert out.shape == (720, 1280, 3) and out.any()
