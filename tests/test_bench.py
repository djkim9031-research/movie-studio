import json
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from studio import bench, identity, provenance
from studio.config import REPO_CONFIG, Config
from studio.identity import ClipScore, Summary
from tests.test_identity import make_photos


@pytest.fixture
def cfg(tmp_path):
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    return Config.model_validate({"data_root": str(tmp_path), "models": repo["models"]})


def test_start_frames_are_face_framed_16_9_with_provenance_back_to_the_photo(cfg, tmp_path):
    _, tools = make_photos(cfg)
    starts = bench.choose_starts(cfg, "me", tools, count=2)
    assert len(starts) == 2
    photo, face = starts[0]
    out = tmp_path / "start.png"
    bench.make_start(cfg, photo, face, out)
    with Image.open(out) as image:
        assert image.size == bench.SIZE
    assert provenance.problems(cfg, out) == []


def test_run_renders_every_cell_scores_it_against_the_profile_without_its_start_and_saves_results(
        cfg, tmp_path, monkeypatch):
    _, tools = make_photos(cfg)
    identity.save_profile(cfg, identity.build_profile(cfg, "me", tools))
    rendered, scored = [], []

    def render(cfg, variant, start, prompt, seconds, seed, name, dest):
        rendered.append((start.name, prompt[:20], seed))
        dest.write_bytes(b"mp4 " + name.encode())
        return SimpleNamespace(elapsed_s=30.0)

    def score_clip(profiles, tools, path, pinned):
        scored.append((tuple(next(iter(profiles.values())).sources), pinned))
        return ClipScore(frames=[], summary=Summary(median=0.6, mid_min=0.5, dips=0, dip_seconds=0.0, coverage=0.9,
                                                    yaw_range=0.3, scored=100))

    monkeypatch.setattr(bench.identity, "score_clip", score_clip)
    monkeypatch.setattr(bench.identity, "score_image",
                        lambda profile, tools, path: identity.Frame(t=0.0, sim=0.7, yaw=0.0, px=400))
    run_dir = tmp_path / "run"
    results = bench.run(cfg, alias="me", variant="ltx-int8-2stage", tools=tools, run_dir=run_dir, count=2,
                        echo=print, render=render)
    assert len(results["cells"]) == 6 and len(rendered) == 6
    assert {c["motion"] for c in results["cells"]} == {"still", "turn", "float"}
    start_of = {c["clip"]: c["start"] for c in results["cells"]}
    for sources, pinned in scored:
        assert pinned == [0] and len(sources) == 3
    assert all(c["start"] + ".jpg" not in s for c, (s, _) in zip(results["cells"], scored))
    saved = json.loads((run_dir / "results.json").read_text())
    assert saved["variant"] == "ltx-int8-2stage"
    assert saved["profile"] == identity.digest(identity.load_profiles(cfg, "me"))
    clip = run_dir / "clips" / results["cells"][0]["clip"]
    assert provenance.params(clip)["pinned_frames"] == [0] and provenance.problems(cfg, clip) == []
    assert (run_dir / "review.html").exists() and len(start_of) == 6
    assert all(c["start_sim"] == 0.7 and c["drift"] == pytest.approx(-0.1) for c in results["cells"])


@pytest.mark.parametrize("variant,kind,detail", [
    ("ltx-int8-2stage", "template", None),
    ("ltx-bf16-2stage", "template", "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    ("ltx-bf16-1stage", "graph", ("LTXVAddGuide", 1)),
    ("ltx-bf16-1stage-loop", "graph", ("LTXVAddGuide", 2)),
    ("wan-4step-1022", "graph", ("LoraLoaderModelOnly", 2)),
    ("wan-20step", "graph", ("LoraLoaderModelOnly", 0)),
])
def test_each_variant_renders_through_its_own_graph(cfg, tmp_path, monkeypatch, variant, kind, detail):
    calls = []
    monkeypatch.setattr(bench, "render_i2v", lambda *a, **k: calls.append(("template", k.get("unet"))) or "ok")
    monkeypatch.setattr(bench, "render_graph",
                        lambda cfg, graph, images, **k: calls.append(("graph", graph, images)) or "ok")
    start = tmp_path / "start.png"
    assert bench.render(cfg, variant, start, "p", 5, 7, "x", tmp_path / "c.mp4") == "ok"
    assert calls[0][0] == kind
    if kind == "template":
        assert calls[0][1] == detail
    else:
        graph, images = calls[0][1], calls[0][2]
        assert sum(1 for n in graph.values() if n["class_type"] == detail[0]) == detail[1]
        assert images == {"START": start}


def test_every_variant_names_its_model_sets_and_pinned_frames():
    for name, spec in bench.VARIANTS.items():
        assert spec["models"] and spec["pinned"][0] == 0, name
    assert bench.VARIANTS["ltx-bf16-1stage-loop"]["pinned"] == [0, -1]


def results(variant, medians, start_sim=0.7, digest="d1"):
    cells = []
    for (start, motion), m in medians.items():
        cells.append({"start": start, "motion": motion, "seed": 1001, "clip": f"{start}-{motion}.mp4", "median": m,
                      "mid_min": m - 0.05, "dips": 0, "dip_seconds": 0.0, "coverage": 1.0, "yaw_range": 0.2,
                      "percentile": 50.0, "start_sim": start_sim, "drift": m - start_sim, "render_s": 30.0})
    return {"variant": variant, "alias": "me", "profile": digest, "seconds": 5, "cells": cells}


def grid(value, bump=0.0, photos=6):
    return {(f"p{i}", m): value + (bump if i < photos - 1 else -0.01)
            for i in range(photos) for m in ("still", "turn", "float")}


def test_compare_pairs_cells_and_applies_the_round_rule():
    base = results("base", grid(0.60))
    better = results("better", grid(0.60, bump=0.05))
    same = results("same", grid(0.60, bump=0.002))
    report = bench.compare([base, better, same], delta=0.02)
    rows = {r["variant"]: r for r in report["variants"]}
    assert rows["better"]["gain"] == pytest.approx(5 * 0.05 / 6, abs=1e-6)
    assert rows["better"]["photo_wins"] == 5 and rows["better"]["verdict"] == "passes screening"
    assert rows["same"]["verdict"] == "no clear gain"
    assert rows["base"]["verdict"] == "baseline"


def test_compare_refuses_runs_scored_against_different_profiles():
    with pytest.raises(ValueError, match="profile"):
        bench.compare([results("a", grid(0.6)), results("b", grid(0.6), digest="other")], delta=0.02)


def test_a_motion_that_gets_worse_fails_the_round_even_with_an_overall_gain():
    base = results("base", grid(0.60))
    medians = grid(0.60, bump=0.08)
    for i in range(6):
        medians[(f"p{i}", "turn")] = 0.55
    report = bench.compare([base, results("mixed", medians)], delta=0.02)
    row = [r for r in report["variants"] if r["variant"] == "mixed"][0]
    assert row["verdict"] == "a motion got worse"


def test_rescore_scores_every_clip_again_against_the_current_profiles(cfg, tmp_path, monkeypatch):
    _, tools = make_photos(cfg)
    identity.save_profile(cfg, identity.build_profile(cfg, "me", tools))
    run_dir = tmp_path / "run"
    (run_dir / "starts").mkdir(parents=True)
    (run_dir / "clips").mkdir()
    photo = cfg.data_root / "cast" / "me" / "photos" / "me1.jpg"
    start = run_dir / "starts" / "me1.png"
    start.write_bytes(b"start")
    provenance.write(cfg, start, kind="start", models=[], inputs=[photo], params={})
    clip = run_dir / "clips" / "me1-turn.mp4"
    clip.write_bytes(b"clip")
    provenance.write(cfg, clip, kind="clip", models=["ltx25-int8"], inputs=[start], params={"pinned_frames": [0, -1]})
    old = {"variant": "v", "profile": "old", "cells": [{"start": "me1", "motion": "turn", "seed": 1, "clip": clip.name,
                                                       "median": 0.6, "percentile": 40.0, "render_s": 30.0}]}
    (run_dir / "results.json").write_text(json.dumps(old))
    seen = []
    monkeypatch.setattr(bench.identity, "score_clip", lambda profiles, tools, path, pinned: seen.append(
        (sorted(next(iter(profiles.values())).sources), pinned)) or ClipScore(frames=[], summary=Summary(
            median=0.2, mid_min=-0.4, dips=1, dip_seconds=0.3, coverage=0.8, yaw_range=0.4, scored=90)))
    monkeypatch.setattr(bench.identity, "score_image",
                        lambda profiles, tools, path: identity.Frame(t=0.0, sim=0.5, yaw=0.0, px=400))
    results = bench.rescore(cfg, alias="me", tools=tools, run_dir=run_dir, echo=print)
    cell = results["cells"][0]
    assert seen == [(["me0.jpg", "me2.jpg", "me3.jpg"], [0, -1])]
    assert cell["median"] == 0.2 and cell["drift"] == pytest.approx(-0.3) and "percentile" not in cell
    assert results["profile"] == identity.digest(identity.load_profiles(cfg, "me"))
