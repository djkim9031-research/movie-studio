import json
import os
from datetime import datetime
from types import SimpleNamespace

import httpx2
import pytest

from studio.claude.client import Claude, CredentialsError, Refused, read_api_key
from studio.claude.ledger import BudgetExceeded, Ledger
from studio.config import Config

CLOCK = datetime(2031, 2, 3)


def write_env(path, text, mode=0o600):
    path.write_text(text)
    os.chmod(path, mode)
    return path


def test_read_api_key_from_env_file(tmp_path):
    assert read_api_key(write_env(tmp_path / "a.env", "# key\nANTHROPIC_API_KEY=test-key\n")) == "test-key"


def test_read_api_key_refuses_a_group_readable_file(tmp_path):
    with pytest.raises(CredentialsError, match="0600"):
        read_api_key(write_env(tmp_path / "a.env", "ANTHROPIC_API_KEY=test-key\n", mode=0o640))


def test_read_api_key_refuses_an_other_readable_file(tmp_path):
    with pytest.raises(CredentialsError, match="0600"):
        read_api_key(write_env(tmp_path / "a.env", "ANTHROPIC_API_KEY=test-key\n", mode=0o604))


def test_read_api_key_names_a_missing_file(tmp_path):
    with pytest.raises(CredentialsError, match="a.env"):
        read_api_key(tmp_path / "a.env")


def sse(body, headers=None):
    """`body`, a whole message with text blocks, as the event stream the API sends for it."""
    events = [("message_start", {"type": "message_start",
                                 "message": {**body, "content": [], "stop_reason": None, "stop_sequence": None}})]
    for i, block in enumerate(body["content"]):
        events += [("content_block_start", {"type": "content_block_start", "index": i,
                                            "content_block": {"type": "text", "text": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": i,
                                            "delta": {"type": "text_delta", "text": block["text"]}}),
                   ("content_block_stop", {"type": "content_block_stop", "index": i})]
    events += [("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                                  "usage": {"output_tokens": body["usage"]["output_tokens"]}}),
               ("message_stop", {"type": "message_stop"})]
    text = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx2.Response(200, content=text.encode(), headers={"content-type": "text/event-stream", **(headers or {})})


def reply(stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "ok"}] if stop_reason == "end_turn" else [],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}


def make_config(tmp_path, month_cap=40.0):
    (tmp_path / "secrets").mkdir()
    write_env(tmp_path / "secrets" / "anthropic.env", "ANTHROPIC_API_KEY=test-key\n")
    return Config.model_validate({
        "data_root": str(tmp_path),
        "claude": {"routes": {"doctor": {"model": "claude-opus-5-5", "effort": "low"}},
                   "prices": {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2}},
                   "month_cap_usd": month_cap}})


def make_claude(tmp_path, monkeypatch, stop_reason="end_turn", month_cap=40.0, env=None):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    cfg = make_config(tmp_path, month_cap)
    seen = SimpleNamespace(bodies=[], urls=[])

    def handler(request):
        seen.urls.append(str(request.url))
        seen.bodies.append(json.loads(request.content))
        return sse(reply(stop_reason), headers={"request-id": "req_1"})

    ledger = Ledger(cfg.ledger_dir, cfg.claude, clock=lambda: CLOCK)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return Claude(cfg, http_client=http, ledger=ledger), ledger, seen


def say_hi(claude):
    return claude.create("doctor", film="_doctor", max_tokens=16, messages=[{"role": "user", "content": "hi"}])


def test_create_sends_the_route_model_and_effort_and_records_the_cost(tmp_path, monkeypatch):
    claude, ledger, seen = make_claude(tmp_path, monkeypatch)
    say_hi(claude)
    assert seen.bodies[0]["model"] == "claude-opus-5-5"
    assert seen.bodies[0]["output_config"] == {"effort": "low"}
    assert ledger.month_total("2031-02") == pytest.approx(4.0)


def test_create_streams_the_reply_and_records_its_request_id(tmp_path, monkeypatch):
    claude, ledger, seen = make_claude(tmp_path, monkeypatch)
    message = say_hi(claude)
    assert seen.bodies[0]["stream"] is True  # a long reply is not cut off by the client's 10-minute guard
    assert message.stop_reason == "end_turn" and message.content[0].text == "ok"
    assert ledger.entries("2031-02")[0]["request_id"] == "req_1"


def test_requests_go_to_the_anthropic_api_even_if_the_environment_names_another_host(tmp_path, monkeypatch):
    claude, _, seen = make_claude(tmp_path, monkeypatch, env={"ANTHROPIC_BASE_URL": "https://gateway.example.invalid"})
    say_hi(claude)
    assert seen.urls[0].startswith("https://api.anthropic.com/")


def test_custom_headers_from_the_environment_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Forward-To: elsewhere")
    with pytest.raises(CredentialsError, match="ANTHROPIC_CUSTOM_HEADERS"):
        Claude(make_config(tmp_path))


def test_create_checks_caps_before_calling(tmp_path, monkeypatch):
    claude, _, seen = make_claude(tmp_path, monkeypatch, month_cap=0.0)
    with pytest.raises(BudgetExceeded):
        say_hi(claude)
    assert seen.bodies == []


def test_refusal_raises_after_the_cost_is_recorded(tmp_path, monkeypatch):
    claude, ledger, _ = make_claude(tmp_path, monkeypatch, stop_reason="refusal")
    with pytest.raises(Refused):
        say_hi(claude)
    assert ledger.month_total("2031-02") == pytest.approx(4.0)


def test_a_paid_call_is_recorded_even_when_pricing_it_fails(tmp_path, monkeypatch):
    claude, ledger, _ = make_claude(tmp_path, monkeypatch)

    def broken(message, requested_model):
        raise ValueError("new usage shape")

    monkeypatch.setattr("studio.claude.client.hops_from_message", broken)
    with pytest.raises(ValueError, match="new usage shape"):
        say_hi(claude)
    rows = ledger.entries("2031-02")
    assert len(rows) == 1 and rows[0]["hops"][0]["priced"] is False
    assert ledger.month_total("2031-02") == pytest.approx(10.0)
