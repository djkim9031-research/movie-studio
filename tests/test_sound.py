import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from studio import sound
from studio.config import REPO_CONFIG, Config


@pytest.fixture
def cfg(tmp_path):
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    return Config.model_validate({"data_root": str(tmp_path / "data"), "models": repo["models"]})


def ready(cfg, monkeypatch):
    """A worker that looks installed: environment, source, tokenizer, and every model set in place."""
    monkeypatch.setattr("studio.sound.models.present", lambda cfg, name: [])
    worker = sound.PrismAudio()
    for path in (worker.python(cfg), worker.home(cfg) / "src" / "predict.py",
                 cfg.data_root / "models" / "audio" / sound.TOKENIZER, worker.detector(cfg) / "__init__.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return worker


def test_problems_name_each_missing_piece(cfg):
    found = sound.PrismAudio().problems(cfg)
    assert any("t5gemma" in p and "uv run studio models fetch t5gemma" in p for p in found)
    assert any("prismaudio" in p and "uv run studio models fetch prismaudio" in p for p in found)
    assert any("worker environment" in p for p in found)


def test_the_t5gemma_set_is_a_model_folder_where_the_runner_looks(cfg):
    pinned = cfg.models["t5gemma"]
    targets = {f.target for f in pinned.files}
    needed = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json")
    assert pinned.repo == "google/t5gemma-l-l-ul2-it" and cfg.model_root("t5gemma") == sound.PrismAudio.audio_root(cfg)
    assert {f"{sound.T5GEMMA}/{name}" for name in needed} <= targets


def test_render_runs_the_runner_offline_in_the_worker_python_and_returns_the_track(cfg, tmp_path, monkeypatch):
    worker = ready(cfg, monkeypatch)
    calls = []

    def run(args, *, input, capture_output, text, timeout, env, cwd):
        calls.append((args, json.loads(input), env))
        Path(json.loads(input)["out"]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stdout='loading\n{"ok": true, "elapsed_s": 41.0, "peak_gib": 17.5}\n',
                               stderr="")

    monkeypatch.setattr("studio.sound.subprocess.run", run)
    clip, dest = tmp_path / "shot1.mp4", tmp_path / "shot1.wav"
    track = worker.render(cfg, clip=clip, prompt="Generate the sound of hooves on stone.", seconds=5, seed=41,
                          dest=dest)
    args, job, env = calls[0]
    assert args == [str(worker.python(cfg)), "-I", str(sound.RUNNER)]
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and "HF_TOKEN" not in env
    assert (job["clip"], job["out"], job["seconds"], job["seed"]) == (str(clip), str(dest), 5, 41)
    assert job["prompt"] == "Generate the sound of hooves on stone." and job["device"] == "cuda"
    assert job["weights"]["prismaudio"].endswith("audio/prismaudio/prismaudio.ckpt")
    assert job["weights"]["videoprism"].endswith("audio/videoprism/flax_lvt_large_f8r288_repeated.npz")
    assert job["weights"]["t5gemma"].endswith("audio/t5gemma-l-l-ul2-it")
    assert track.path == dest and track.peak_gib == 17.5


def test_render_reports_the_runner_error(cfg, tmp_path, monkeypatch):
    worker = ready(cfg, monkeypatch)
    monkeypatch.setattr("studio.sound.subprocess.run", lambda args, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="Traceback ...\nRuntimeError: CUDA out of memory\n"))
    with pytest.raises(sound.SoundError, match="shot1.mp4: RuntimeError: CUDA out of memory"):
        worker.render(cfg, clip=tmp_path / "shot1.mp4", prompt="x", seconds=5, seed=1, dest=tmp_path / "shot1.wav")


def test_render_reports_the_runner_error_line_and_keeps_the_log(cfg, tmp_path, monkeypatch):
    worker = ready(cfg, monkeypatch)
    # Upstream prints its run time at exit, after the runner's own error line.
    stderr = ("Traceback ...\nRuntimeError: unexpected audio shape (2, 440320)\n"
              "prismaudio: error: RuntimeError: unexpected audio shape (2, 440320)\n执行时间: 0.48 秒\n")
    monkeypatch.setattr("studio.sound.subprocess.run", lambda args, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr=stderr))
    log = tmp_path / "shot1.log"
    with pytest.raises(sound.SoundError) as failed:
        worker.render(cfg, clip=tmp_path / "shot1.mp4", prompt="x", seconds=10, seed=1, dest=tmp_path / "shot1.wav")
    assert str(failed.value) == f"PrismAudio failed on shot1.mp4: RuntimeError: unexpected audio shape (2, 440320); " \
                                f"log: {log}"
    assert log.read_text(encoding="utf-8") == stderr


def test_problems_name_a_missing_speech_detector(cfg, monkeypatch):
    worker = ready(cfg, monkeypatch)
    assert worker.problems(cfg) == []
    (worker.detector(cfg) / "__init__.py").unlink()
    worker.detector(cfg).rmdir()
    assert worker.problems(cfg) == [f"no speech detector in {worker.home(cfg)}; rerun workers/prismaudio/setup.sh"]


def test_speech_runs_the_detector_offline_in_the_worker_python(cfg, tmp_path, monkeypatch):
    worker = ready(cfg, monkeypatch)
    seen = {}

    def run(args, **kw):
        seen.update(args=args, **kw)
        return SimpleNamespace(returncode=0, stdout='{"speech": 0.42}\n', stderr="")

    monkeypatch.setattr("studio.sound.subprocess.run", run)
    assert worker.speech(cfg, tmp_path / "shot2.wav") == 0.42
    assert seen["args"] == [str(worker.python(cfg)), "-I", str(sound.SPEECH)]
    assert json.loads(seen["input"]) == {"wav": str(tmp_path / "shot2.wav")}
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and "HF_TOKEN" not in seen["env"]


def test_a_failed_speech_check_names_the_track(cfg, tmp_path, monkeypatch):
    worker = ready(cfg, monkeypatch)
    monkeypatch.setattr("studio.sound.subprocess.run",
                        lambda args, **kw: SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nOSError: bad wav"))
    with pytest.raises(sound.SoundError, match="speech check failed on shot2.wav: OSError: bad wav"):
        worker.speech(cfg, tmp_path / "shot2.wav")
