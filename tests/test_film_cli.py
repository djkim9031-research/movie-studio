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
from tests.test_film_prompts import FakeClaude, plan
from tests.test_refs import write
from tests.test_scene import TOOLS


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
    monkeypatch.setattr("studio.cli.Claude", lambda cfg: FakeClaude(plan(3)))

    def render_still(cfg, references, request, *, seed, size, name, dest):
        Image.new("RGB", size).save(dest)
        return Still(path=dest, caption="caption", elapsed_s=2.0, graph={})

    def render_clip(cfg, variant, start, prompt, seconds, seed, name, dest, size):
        dest.write_bytes(b"mp4")
        return SimpleNamespace(clip=dest, elapsed_s=27.0, peak_gib=42.7, reused_nodes=0)

    monkeypatch.setattr("studio.film.render_still", render_still)
    monkeypatch.setattr("studio.film.render_clip", render_clip)
    monkeypatch.setattr("studio.film.prepare_still", lambda cfg, picture, out, size: picture)
    monkeypatch.setattr("studio.film.assemble.silent_film", lambda cfg, clips, out, **kw: out.write_bytes(b"film"))
    (tmp_path / "pictures").mkdir()
    write(tmp_path / "pictures" / "a.jpg", (2000, 1500))
    write(tmp_path / "pictures" / "b.jpg", (1800, 1500))
    return cfg_file, tmp_path / "pictures", tmp_path / "data" / "projects" / "films"


def make(cfg_file, pictures, *extra):
    return CliRunner().invoke(app, ["--config", str(cfg_file), "film", "make", str(pictures), "--name", "ride",
                                    "--prompt", "me riding a horse in front of the palace", "--shots", "3", *extra])


def test_film_make_names_the_missing_model_set(setup, monkeypatch):
    cfg_file, pictures, _ = setup
    monkeypatch.setattr("studio.cli.models.present",
                        lambda cfg, name: ["x.safetensors: missing"] if name == "wan22-i2v" else [])
    result = make(cfg_file, pictures)
    assert result.exit_code == 1 and "uv run studio models fetch wan22-i2v" in result.output


def test_film_make_reports_the_film_and_the_sound_step(setup):
    cfg_file, pictures, films = setup
    result = make(cfg_file, pictures)
    assert result.exit_code == 0, result.output
    assert 'PASS  "Ride at the Palace", 3 shots' in result.output and str(films / "ride" / "film.mp4") in result.output
    assert "uv run studio film sound --name ride" in result.output
    assert (films / "ride" / "film.mp4").exists()


def test_film_sound_lists_what_the_worker_still_needs(setup):
    cfg_file, pictures, _ = setup
    assert make(cfg_file, pictures).exit_code == 0
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "film", "sound", "--name", "ride"])
    assert result.exit_code == 1
    assert "FAIL  no VideoPrism text tokenizer" in result.output and "FAIL  no worker environment" in result.output


def test_film_sound_reports_an_unknown_film(setup):
    cfg_file, _, _ = setup
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "film", "sound", "--name", "nope"])
    assert result.exit_code == 1 and "FAIL  no film nope" in result.output
