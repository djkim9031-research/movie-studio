import yaml
from typer.testing import CliRunner

from studio.cli import app
from studio.config import REPO_CONFIG, Config
from tests.test_identity import make_photos
from tests.test_refs import ME, FakeTools, face, write


def setup(tmp_path, monkeypatch):
    cfg_file = tmp_path / "studio.yaml"
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    cfg_file.write_text(yaml.safe_dump({"data_root": str(tmp_path), "models": repo["models"]}))
    _, tools = make_photos(Config.model_validate({"data_root": str(tmp_path)}))
    monkeypatch.setattr("studio.cli.face_tools", lambda cfg: tools)
    monkeypatch.setattr("studio.cli.models.present", lambda cfg, name: [])
    return cfg_file, tools


def run(cfg_file, *args):
    return CliRunner().invoke(app, ["--config", str(cfg_file), *args])


def test_profile_reports_the_photos_used_and_the_genuine_spread(tmp_path, monkeypatch):
    cfg_file, _ = setup(tmp_path, monkeypatch)
    result = run(cfg_file, "identity", "profile", "--person", "me")
    assert result.exit_code == 0, result.output
    assert "PASS  profile of me: 4 photos" in result.output and "genuine pairs 6" in result.output
    assert (tmp_path / "cast" / "me" / "profile" / "sface.npz").exists()


def test_score_refuses_a_file_without_provenance_and_scores_a_camera_photo(tmp_path, monkeypatch):
    cfg_file, tools = setup(tmp_path, monkeypatch)
    assert run(cfg_file, "identity", "profile", "--person", "me").exit_code == 0
    stray = write(tmp_path / "flux_still.png", (1600, 1200))
    tools.by_size[(1200, 1600)] = [face(400, 300, 400, ME)]
    result = run(cfg_file, "identity", "score", "--person", "me", str(stray))
    assert result.exit_code == 1 and "REFUSED  flux_still.png: unknown origin (no provenance)" in result.output
    photo = tmp_path / "cast" / "me" / "photos" / "me0.jpg"
    result = run(cfg_file, "identity", "score", "--person", "me", str(photo))
    assert result.exit_code == 0, result.output
    assert "me0.jpg  similarity" in result.output and "percentile" in result.output


def test_bench_identity_runs_the_variant_and_prints_a_summary_per_motion(tmp_path, monkeypatch):
    cfg_file, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: True)
    monkeypatch.setattr("studio.gpu.runtime.container_pids", lambda: set())
    from studio.gpu.preflight import Verdict
    monkeypatch.setattr("studio.cli.run_preflight", lambda *a, **k: Verdict(ok=True))
    seen = {}

    def run(cfg, *, alias, variant, tools, run_dir, echo, count):
        seen.update(alias=alias, variant=variant, run_dir=run_dir, count=count)
        cells = [{"motion": m, "median": v, "mid_min": v - 0.1, "dips": d, "coverage": 0.9}
                 for m, v, d in [("still", 0.62, 0), ("still", 0.58, 0), ("turn", 0.40, 1), ("float", 0.50, 0)]]
        return {"variant": variant, "cells": cells}

    monkeypatch.setattr("studio.cli.bench.run", run)
    result = run_cli = CliRunner().invoke(app, ["--config", str(cfg_file), "bench", "identity", "--person", "me",
                                                "--name", "baseline", "--count", "2"])
    assert run_cli.exit_code == 0, result.output
    assert seen["variant"] == "ltx-int8-2stage" and seen["count"] == 2
    assert seen["run_dir"] == tmp_path / "bench" / "baseline"
    assert "still   median 0.600" in result.output and "turn    median 0.400" in result.output
    assert "dips 1" in result.output
