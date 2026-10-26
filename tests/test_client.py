import json
import os
from datetime import datetime

import httpx2
import pytest

from studio.claude.client import Claude, CredentialsError, Refused, read_api_key
from studio.claude.ledger import BudgetExceeded, Ledger
from studio.config import Config


def write_env(path, text, mode=0o600):
    path.write_text(text)
    os.chmod(path, mode)
    return path


def test_read_api_key_from_env_file(tmp_path):
    assert read_api_key(write_env(tmp_path / "a.env", "# key\nANTHROPIC_API_KEY=test-key\n")) == "test-key"


def test_read_api_key_refuses_a_group_readable_file(tmp_path):
    with pytest.raises(CredentialsError, match="0600"):
        read_api_key(write_env(tmp_path / "a.env", "ANTHROPIC_API_KEY=test-key\n", mode=0o640))


def test_read_api_key_names_a_missing_file(tmp_path):
    with pytest.raises(CredentialsError, match="a.env"):
        read_api_key(tmp_path / "a.env")


def reply(stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "ok"}] if stop_reason == "end_turn" else [],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}


def make_claude(tmp_path, monkeypatch, stop_reason="end_turn", month_cap=40.0):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    (tmp_path / "secrets").mkdir()
    write_env(tmp_path / "secrets" / "anthropic.env", "ANTHROPIC_API_KEY=test-key\n")
    cfg =Config.model_validate({
        "data_root": str(tmp_path),
        "claude": {"routes": {"doctor": {"model": "claude-opus-5-5", "effort": "low"}},
                   "prices": {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2}},
                   "month_cap_usd": month_cap}})
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=reply(stop_reason), headers={"request-id": "req_1"})

    ledger = Ledger(cfg.ledger_dir, cfg.claude, clock=lambda: datetime(2026, 10, 26))
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return Claude(cfg, http_client=http, ledger=ledger), ledger, sent


def test_create_sends_the_route_model_and_effort_and_records_the_cost(tmp_path, monkeypatch):
    claude, ledger, sent = make_claude(tmp_path, monkeypatch)
    claude.create("doctor", film="_doctor", max_tokens=16, messages=[{"role": "user", "content": "hi"}])
    assert sent[0]["model"] == "claude-opus-5-5"
    assert sent[0]["output_config"] == {"effort": "low"}
    assert ledger.month_total("2026-10") == pytest.approx(4.0)


def test_create_checks_caps_before_calling(tmp_path, monkeypatch):
    claude, _, sent = make_claude(tmp_path, monkeypatch, month_cap=0.0)
    with pytest.raises(BudgetExceeded):
        claude.create("doctor", film="_doctor", max_tokens=16, messages=[{"role": "user", "content": "hi"}])
    assert sent == []


def test_refusal_raises_after_the_cost_is_recorded(tmp_path, monkeypatch):
    claude, ledger, _ = make_claude(tmp_path, monkeypatch, stop_reason="refusal")
    with pytest.raises(Refused):
        claude.create("doctor", film="_doctor", max_tokens=16, messages=[{"role": "user", "content": "hi"}])
    assert ledger.month_total("2026-10") == pytest.approx(4.0)
