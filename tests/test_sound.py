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
                 cfg.data_root / "models" / "audio" / sound.TOKENIZER):
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
