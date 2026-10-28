import json
import os
import subprocess
from datetime import datetime

import httpx2
import pytest

from studio.claude.client import Claude
from studio.claude.ledger import Ledger
from studio.config import Config, load_config
from studio.doctor import (Check, check_claude_cache, check_disk, check_driver, check_ffmpeg, check_gpu,
                           check_private, exit_code, guarded)
from studio.gpu.preflight import GpuMemory, GpuProcess, Verdict
from studio.media import ffmpeg


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


def test_check_private_fails_when_group_or_others_can_write(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    folder.chmod(0o775)
    check = check_private(folder)
    assert not check.ok and f"chmod 700 {folder}" in check.detail
    folder.chmod(0o700)
    assert check_private(folder).ok


def test_a_check_that_raises_becomes_a_fail_line():
    def broken():
        raise subprocess.CalledProcessError(9, ["nvidia-smi"], stderr="NVML init\nDriver/library version mismatch\n")

    assert guarded("gpu", broken) == [Check("gpu", False, "Driver/library version mismatch")]


@pytest.mark.skipif(not ffmpeg.binary(load_config()).exists(), reason="run: uv run studio fetch-ffmpeg")
def test_nvenc_test_encodes_are_skipped_while_the_gpu_is_busy(monkeypatch):
    def must_not_run(cfg, encoder):
        raise AssertionError("encoded on a busy GPU")

    monkeypatch.setattr(ffmpeg, "try_encode", must_not_run)
    encodes = [c for c in check_ffmpeg(load_config(), gpu_free=False) if c.name.startswith("encode ")]
    assert len(encodes) == 3
    assert all(not c.ok and c.detail == "skipped: GPU busy" for c in encodes)


def test_the_gpu_check_names_the_display_processes_it_let_through(monkeypatch):
    driver = subprocess.CompletedProcess([], 0, stdout="580.126.09\n")
    monkeypatch.setattr("studio.doctor.subprocess.run", lambda *a, **k: driver)
    monkeypatch.setattr("studio.doctor.run_preflight", lambda gpu: Verdict(
        ok=True, memory=GpuMemory(73415, 1705), display=[GpuProcess(26315, "/usr/NX/bin/nxnode.bin", 406)]))
    assert check_gpu(load_config())[1] == Check(
        "gpu idle", True, "no compute processes but the display's (pid 26315 /usr/NX/bin/nxnode.bin, 406 MiB), "
                          "70.0 GiB free")
