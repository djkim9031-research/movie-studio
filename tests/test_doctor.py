import json
import os
from datetime import datetime

import httpx2

from studio.claude.client import Claude
from studio.claude.ledger import Ledger
from studio.config import Config
from studio.doctor import Check, check_claude_cache, check_disk, check_driver, exit_code


def test_check_disk_passes_when_enough_space_is_free(tmp_path):
    assert check_disk(tmp_path, min_free_gb=0).ok


def test_check_disk_fails_and_reports_free_space(tmp_path):
    check = check_disk(tmp_path, min_free_gb=10**9)
    assert not check.ok and "GB free" in check.detail


def test_check_driver_needs_580_or_newer():
    assert check_driver("580.126.09").ok
    assert not check_driver("570.86.15").ok


def test_exit_code_is_nonzero_when_any_check_fails():
    assert exit_code([Check("a", True, ""), Check("b", True, "")]) == 0
    assert exit_code([Check("a", True, ""), Check("b", False, "")]) == 1


def claude_with_replies(tmp_path, monkeypatch, cache_reads):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    (tmp_path / "secrets").mkdir()
    env = tmp_path / "secrets" / "anthropic.env"
    env.write_text("ANTHROPIC_API_KEY=test-key\n")
    os.chmod(env, 0o600)
    cfg = Config.model_validate({
        "data_root": str(tmp_path),
        "claude": {"routes": {"doctor": {"model": "claude-opus-5-5", "effort": "low"}},
                   "prices": {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2}}}})
    replies, sent = iter(cache_reads), []

    def handler(request):
        sent.append(json.loads(request.content))
        read = next(replies)
        usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": read,
                 "cache_creation_input_tokens": 0 if read else 1500}
        body = {"id": "msg", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": usage}
        return httpx2.Response(200, json=body, headers={"request-id": "req"})

    ledger = Ledger(cfg.ledger_dir, cfg.claude, clock=lambda: datetime(2026, 10, 26))
    return Claude(cfg, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), ledger=ledger), sent


def test_cache_check_passes_when_the_second_call_reads_the_cache(tmp_path, monkeypatch):
    claude, sent = claude_with_replies(tmp_path, monkeypatch, cache_reads=[0, 1500])
    checks = check_claude_cache(claude)
    assert all(c.ok for c in checks)
    assert sent[0]["system"] == sent[1]["system"]
    assert sent[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_check_fails_when_nothing_is_read_back(tmp_path, monkeypatch):
    claude, _ = claude_with_replies(tmp_path, monkeypatch, cache_reads=[0, 0])
    assert not all(c.ok for c in check_claude_cache(claude))
