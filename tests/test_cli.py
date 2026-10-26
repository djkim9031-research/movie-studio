from datetime import datetime

from typer.testing import CliRunner

from studio.claude.ledger import Hop, Ledger
from studio.cli import app
from studio.config import load_config


def test_cost_prints_the_month_total_against_the_cap(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n"
                        "claude:\n  month_cap_usd: 40\n"
                        "  prices:\n    claude-opus-5-5: {input: 4.0, output: 20.0, cache_read: 0.2}\n")
    cfg = load_config(cfg_file)
    Ledger(cfg.ledger_dir, cfg.claude, clock=lambda: datetime(2026, 10, 26)).record(
        route="doctor", film="_doctor", hops=[Hop("claude-opus-5-5", input_tokens=250_000)])
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "cost", "--month", "2026-10"])
    assert result.exit_code == 0, result.output
    assert "2026-10: $1.00 of $40.00 cap, 1 calls" in result.output
    assert "_doctor" in result.output


def test_doctor_prints_every_check_even_when_one_of_them_raises(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\ndisk:\n  docker_root: {tmp_path / 'missing'}\n")
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "doctor"])
    assert result.exit_code == 1
    assert f"FAIL  disk {tmp_path / 'missing'}" in result.output
    assert "nvidia driver" in result.output and "ffmpeg" in result.output


def test_models_verify_passes_matching_files_and_fails_altered_ones(tmp_path):
    import hashlib
    weights = tmp_path / "models" / "comfy" / "vae" / "a.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"aaaa")
    digest = hashlib.sha256(b"aaaa").hexdigest()
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\nmodels:\n  demo:\n    repo: org/demo\n    revision: abc\n"
                        f"    files:\n      - {{path: vae/a.safetensors, size: 4, sha256: {digest}}}\n")
    ok = CliRunner().invoke(app, ["--config", str(cfg_file), "models", "verify", "demo"])
    assert ok.exit_code == 0 and "PASS  demo: 1 files match" in ok.output
    weights.write_bytes(b"bbbb")
    bad = CliRunner().invoke(app, ["--config", str(cfg_file), "models", "verify", "demo"])
    assert bad.exit_code == 1 and "FAIL  vae/a.safetensors: sha256" in bad.output


def test_render_test_refuses_while_the_container_is_down(monkeypatch):
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: False)
    result = CliRunner().invoke(app, ["render-test"])
    assert result.exit_code == 1
    assert "run: uv run studio gpu up" in result.output


def test_gpu_up_prints_the_local_address(monkeypatch):
    monkeypatch.setattr("studio.gpu.runtime.up", lambda cfg: None)
    result = CliRunner().invoke(app, ["gpu", "up"])
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8188" in result.output


def test_render_test_reports_a_failed_render_on_one_line(tmp_path, monkeypatch):
    from studio.gpu.comfy import RenderError
    from studio.gpu.preflight import Verdict

    class FailingClient:
        def run(self, workflow, timeout_s):
            raise RenderError("CLIPTextEncode: Failed to find C compiler")

    (tmp_path / "comfy-io" / "input").mkdir(parents=True)
    (tmp_path / "projects" / "_tests").mkdir(parents=True)
    (tmp_path / "projects" / "_tests" / "still.png").write_bytes(b"png")
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n")
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: True)
    monkeypatch.setattr("studio.gpu.runtime.container_pids", lambda: set())
    monkeypatch.setattr("studio.cli.run_preflight", lambda *a, **k: Verdict(ok=True))
    monkeypatch.setattr("studio.gpu.runtime.client", lambda cfg: FailingClient())
    monkeypatch.setattr("studio.render.VramSampler", lambda: _NoSampler())
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "render-test"])
    assert result.exit_code == 1
    assert "FAIL  CLIPTextEncode: Failed to find C compiler" in result.output
    assert "Traceback" not in result.output


class _NoSampler:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_render_test_reports_the_clip_and_the_absolute_gpu_memory_peak(tmp_path, monkeypatch):
    from studio.gpu.preflight import Verdict
    from studio.media.smoke import smoke_mux

    repo_cfg = load_config()
    if not (repo_cfg.bin_dir / "ffmpeg").exists():
        return
    out_dir = tmp_path / "comfy-io" / "output" / "studio"
    out_dir.mkdir(parents=True)
    smoke_mux(repo_cfg, out_dir / "render_test_00001_.mp4")
    (tmp_path / "comfy-io" / "input").mkdir(parents=True)
    (tmp_path / "projects" / "_tests").mkdir(parents=True)
    (tmp_path / "projects" / "_tests" / "still.png").write_bytes(b"png")
    (tmp_path / "bin").symlink_to(repo_cfg.bin_dir)

    class DoneClient:
        def run(self, workflow, timeout_s):
            return {"outputs": {"75": {"images": [{"filename": "render_test_00001_.mp4", "subfolder": "studio",
                                                   "type": "output"}]}}}

    class Peak:
        peak_mib = 43008

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n")
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: True)
    monkeypatch.setattr("studio.gpu.runtime.container_pids", lambda: set())
    monkeypatch.setattr("studio.cli.run_preflight", lambda *a, **k: Verdict(ok=True))
    monkeypatch.setattr("studio.gpu.runtime.client", lambda cfg: DoneClient())
    monkeypatch.setattr("studio.render.VramSampler", Peak)
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "render-test"])
    assert result.exit_code == 0, result.output
    assert "PASS  10.00 s 1280x720 24 fps, audio aac" in result.output
    assert "GPU memory peak 42.0 GiB on the card" in result.output
