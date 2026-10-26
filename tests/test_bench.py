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

    def score_clip(profile, tools, path, pinned):
        scored.append((tuple(profile.sources), pinned))
        return ClipScore(frames=[], summary=Summary(median=0.6, mid_min=0.5, dips=0, dip_seconds=0.0, coverage=0.9,
                                                    yaw_range=0.3, scored=100))

    monkeypatch.setattr(bench.identity, "score_clip", score_clip)
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
    assert saved["variant"] == "ltx-int8-2stage" and saved["profile"] == identity.load_profile(cfg, "me").digest
    clip = run_dir / "clips" / results["cells"][0]["clip"]
    assert provenance.params(clip)["pinned_frames"] == [0] and provenance.problems(cfg, clip) == []
    assert (run_dir / "review.html").exists() and len(start_of) == 6
