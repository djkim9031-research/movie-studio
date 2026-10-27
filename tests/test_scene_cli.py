from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from studio.cli import app
from studio.config import REPO_CONFIG
from studio.gpu.preflight import Verdict
from studio.stills import Still
from tests.test_refs import write
from tests.test_scene import TOOLS, FakeClaude


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg_file = tmp_path / "studio.yaml"
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    cfg_file.write_text(yaml.safe_dump({"data_root": str(tmp_path / "data"), "models": repo["models"]}))
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: True)
    monkeypatch.setattr("studio.gpu.runtime.container_pids", lambda: set())
    monkeypatch.setattr("studio.cli.run_preflight", lambda *a, **k: Verdict(ok=True))
    monkeypatch.setattr("studio.cli.models.present", lambda cfg, name: [])
    monkeypatch.setattr("studio.cli.face_tools", lambda cfg: TOOLS)
    monkeypatch.setattr("studio.cli.Claude", lambda cfg: FakeClaude())

    def render_still(cfg, references, request, *, seed, size, name, dest):
        Image.new("RGB", size).save(dest)
        return Still(path=dest, caption="caption", elapsed_s=2.0, graph={})

    def render_clip(cfg, variant, start, prompt, seconds, seed, name, dest, size):
        dest.write_bytes(b"mp4")
        return SimpleNamespace(clip=dest, elapsed_s=27.0, peak_gib=42.7, reused_nodes=0)

    monkeypatch.setattr("studio.scene.render_still", render_still)
    monkeypatch.setattr("studio.scene.render_clip", render_clip)
    monkeypatch.setattr("studio.scene.prepare_still", lambda cfg, picture, out, size: picture)
    (tmp_path / "pictures").mkdir()
    pictures = [str(write(tmp_path / "pictures" / "a.jpg", (2000, 1500))),
                str(write(tmp_path / "pictures" / "b.jpg", (1800, 1500)))]
    return cfg_file, pictures, tmp_path / "data" / "projects" / "scenes"


def test_scene_stills_then_clips(setup):
    cfg_file, pictures, scenes = setup
    folder = Path(pictures[0]).parent
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "scene", "stills", str(folder), "--name", "moon",
                                      "--prompt", "me on the moon", "--count", "2"])
    assert result.exit_code == 0, result.output
    assert "PASS  2 stills" in result.output and str(scenes / "moon" / "review.html") in result.output
    assert "uv run studio scene clips --name moon" in result.output
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "scene", "clips", "--name", "moon", "2",
                                      "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert (scenes / "moon" / "clips" / "still2-seed7-ltx-int8-2stage-5s.mp4").exists()
    assert "PASS  1 clip" in result.output


def test_scene_stills_names_the_missing_model_set(setup, monkeypatch):
    cfg_file, pictures, _ = setup
    monkeypatch.setattr("studio.cli.models.present",
                        lambda cfg, name: ["x.safetensors: missing"] if name == "flux3-klein" else [])
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "scene", "stills", *pictures, "--name", "m",
                                      "--prompt", "x"])
    assert result.exit_code == 1 and "uv run studio models fetch flux3-klein" in result.output


def test_scene_clips_reports_an_unknown_scene(setup):
    cfg_file, _, _ = setup
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "scene", "clips", "--name", "nope", "1"])
    assert result.exit_code == 1 and "FAIL  no scene nope" in result.output
