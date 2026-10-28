import pytest
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
    assert "me0.jpg  score" in result.output and "sface" in result.output


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


def test_bench_compare_prints_each_variant_against_the_first_and_writes_a_page(tmp_path, monkeypatch):
    import json
    from studio import identity
    from tests.test_bench import grid, results
    cfg_file, tools = setup(tmp_path, monkeypatch)
    assert run(cfg_file, "identity", "profile", "--person", "me").exit_code == 0
    digest = identity.digest(identity.load_profiles(Config.model_validate({"data_root": str(tmp_path)}), "me"))
    for name, data in [("base", results("ltx-int8-2stage", grid(0.0), digest=digest)),
                       ("r1", results("wan-4step-1022", grid(0.0, bump=0.4), digest=digest))]:
        (tmp_path / "bench" / name).mkdir(parents=True)
        (tmp_path / "bench" / name / "results.json").write_text(json.dumps(data))
    result = run(cfg_file, "bench", "compare", "--person", "me", "base", "r1")
    assert result.exit_code == 0, result.output
    assert "ltx-int8-2stage" in result.output and "baseline" in result.output
    assert "wan-4step-1022" in result.output and "passes screening" in result.output
    pages = list((tmp_path / "bench").glob("compare-*.html"))
    assert len(pages) == 1 and "r1/clips/p0-still.mp4" in pages[0].read_text()


def make_run(root, name, variant, cells, media):
    import json
    run_dir = root / "bench" / name
    (run_dir / "clips").mkdir(parents=True)
    for c in cells:
        media(run_dir / "clips" / c["clip"])
    (run_dir / "results.json").write_text(json.dumps({"variant": variant, "profile": "d", "cells": cells}))
    return run_dir


def test_blind_pairs_clips_by_cell_as_frame_strips_and_pick_reports(tmp_path, monkeypatch):
    from studio.config import load_config
    from studio.media import ffmpeg
    if not ffmpeg.binary(load_config()).exists():
        pytest.skip("run: uv run studio fetch-ffmpeg")
    cfg_file, _ = setup(tmp_path, monkeypatch)
    real = load_config()
    (tmp_path / "bin").symlink_to(real.bin_dir)

    def clip(path):
        ffmpeg.run(real, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=5",
                   "-pix_fmt", "yuv420p", str(path)).check_returncode()

    cell = lambda start, score: {"start": start, "motion": "still", "seed": 1, "clip": f"{start}.mp4",  # noqa: E731
                                 "median": score}
    make_run(tmp_path, "a", "ltx-int8-2stage", [cell("p1", 0.9), cell("p2", 0.1)], clip)
    make_run(tmp_path, "b", "wan-4step-1022", [cell("p1", 0.1), cell("p2", 0.2)], clip)
    result = run(cfg_file, "bench", "blind", "a", "b", "--name", "ab", "--seed", "4")
    assert result.exit_code == 0, result.output
    page = tmp_path / "bench" / "blind" / "ab" / "page.html"
    assert page.exists() and "ltx" not in page.read_text() and len(list(page.parent.glob("img/*.png"))) == 12
    import json
    key = json.loads((page.parent / "key.json").read_text())
    letter = {o["id"]: k for g in key["groups"] for k, o in g["options"].items()}
    result = run(cfg_file, "bench", "pick", "ab", f"1{letter['a/p1']}", "2=")
    assert result.exit_code == 0, result.output
    assert "ltx-int8-2stage 1" in result.output and "agree 1 of 1" in result.output


def test_blind_keyframes_groups_each_shot_across_engines(tmp_path, monkeypatch):
    cfg_file, _ = setup(tmp_path, monkeypatch)
    from PIL import Image

    def still(path):
        Image.new("RGB", (32, 18)).save(path)

    import json
    for name, engine, score in [("k1", "flux3", None), ("k2", "qwen-edit", 0.4), ("k3", "klein4b", 0.1)]:
        run_dir = tmp_path / "bench" / name
        (run_dir / "stills").mkdir(parents=True)
        still(run_dir / "stills" / "shot01-s1.png")
        (run_dir / "results.json").write_text(json.dumps({"engine": engine, "cells": [
            {"shot": 1, "seed": 1, "file": "shot01-s1.png", "score": score}]}))
    result = run(cfg_file, "bench", "blind-keyframes", "k1", "k2", "k3", "--name", "kf")
    assert result.exit_code == 0, result.output
    key = json.loads((tmp_path / "bench" / "blind" / "kf" / "key.json").read_text())
    assert len(key["groups"]) == 1 and len(key["groups"][0]["options"]) == 3


def test_cast_ingest_reports_kept_images_and_missing_angles(tmp_path, monkeypatch):
    cfg_file, _ = setup(tmp_path, monkeypatch)
    assert run(cfg_file, "identity", "profile", "--person", "me").exit_code == 0
    (tmp_path / "cast" / "me" / "consent.txt").write_text("me, 2026-10-26, local films\n")
    result = run(cfg_file, "cast", "ingest", "--person", "me")
    assert result.exit_code == 0, result.output
    assert "PASS  4 images in" in result.output and "missing angles: slightly turned" in result.output


def test_bench_identity_reports_a_busy_gpu_in_one_line(tmp_path, monkeypatch):
    from studio.gpu.preflight import Verdict

    cfg_file, _ = setup(tmp_path, monkeypatch)
    busy = "pid 4242 (python3) holds 9000 MiB on the GPU"
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: False)
    monkeypatch.setattr("studio.gpu.runtime.run_preflight", lambda gpu: Verdict(ok=False, problems=[busy]))
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "bench", "identity", "--person", "me"])
    assert result.exit_code == 1
    assert [line for line in result.output.splitlines() if line.startswith("FAIL")] == [f"FAIL  GPU not free: {busy}"]
