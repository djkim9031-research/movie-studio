import json
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from studio import identity, keyframe_bench as kb, provenance
from studio.config import REPO_CONFIG, Config
from tests.test_identity import make_photos


@pytest.fixture
def cfg(tmp_path):
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    return Config.model_validate({"data_root": str(tmp_path), "models": repo["models"]})


def fake_render(calls):
    def render(cfg, engine, refs, text, seed, name, dest):
        calls.append((engine, [r.rank for r in refs], text))
        Image.new("RGB", (64, 36)).save(dest)
        return SimpleNamespace(elapsed_s=12.0)
    return render


@pytest.mark.parametrize("engine,count,scored", [("flux3", 4, False), ("qwen-edit", 3, True), ("klein4b", 4, True)])
def test_each_engine_gets_its_best_references_and_only_apache_stills_are_scored(cfg, tmp_path, monkeypatch, engine,
                                                                                count, scored):
    _, tools = make_photos(cfg)
    identity.save_profile(cfg, identity.build_profile(cfg, "me", tools))
    monkeypatch.setattr(kb.identity, "score_image",
                        lambda profiles, tools, path: identity.Frame(t=0, sim=0.3, yaw=0.1, px=300, sims={"sface": 0.7}))
    calls = []
    results = kb.run(cfg, alias="me", engine=engine, tools=tools, run_dir=tmp_path / "run", echo=print,
                     render=fake_render(calls), limit=2)
    assert [ranks for _, ranks, _ in calls] == [list(range(1, count + 1))] * 2
    assert calls[0][2].startswith(f"Create a photograph of the person shown in images 1-{count}")
    cell = results["cells"][0]
    if scored:
        assert cell["score"] == 0.3 and cell["sims"] == {"sface": 0.7}
        assert provenance.problems(cfg, tmp_path / "run" / "stills" / cell["file"]) == []
    else:
        assert cell["score"] is None and "forbids face recognition" in cell["unscored"]
    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert saved["engine"] == engine and len(saved["cells"]) == 2
