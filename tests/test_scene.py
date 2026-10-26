import json
from types import SimpleNamespace

import pytest
from PIL import Image

from studio import scene
from studio.config import Config
from studio.stills import Still
from tests.test_refs import ME, ME_TOO, FakeTools, face, write

REPLY = {"still_request": "Create a photograph of the person shown in images 1-2 on the Moon.",
         "motion_prompt": "On the Moon, the camera stays still; the person holds the same thin smile."}


class FakeClaude:
    def __init__(self):
        self.calls = []

    def create(self, route, *, film, **params):
        self.calls.append((route, film, params))
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(REPLY))])


def pictures(tmp_path):
    return [write(tmp_path / "a.jpg", (2000, 1500)), write(tmp_path / "b.jpg", (1800, 1500)),
            write(tmp_path / "moon.png", (1400, 700), exif=False)]


TOOLS = FakeTools({(1500, 2000): [face(800, 400, 500, ME)], (1500, 1800): [face(700, 400, 500, ME_TOO)]})


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path / "data")})


@pytest.fixture
def fake_stills(monkeypatch):
    calls = []

    def render_still(cfg, references, request, *, seed, size, name, dest):
        calls.append(dict(references=references, request=request, seed=seed, size=size, name=name))
        Image.new("RGB", size, (10, 10, 10)).save(dest)
        return Still(path=dest, caption="A photograph of a person on the Moon.", elapsed_s=1.5, graph={"50": {}})

    monkeypatch.setattr("studio.scene.render_still", render_still)
    return calls


def test_make_stills_writes_references_prompts_stills_and_a_review_page(cfg, tmp_path, fake_stills):
    claude = FakeClaude()
    lines = []
    made = scene.make_stills(cfg, name="moon", prompt="me on the moon, thin smile", pictures=pictures(tmp_path),
                             count=2, portrait=False, claude=claude, tools=TOOLS, echo=lines.append)
    folder = cfg.projects_dir / "scenes" / "moon"
    assert [p.name for p in sorted((folder / "refs").iterdir())] == [
        "extra1.png", "extra1_square.png", "person1.png", "person2.png"]
    assert [c["request"] for c in fake_stills] == [REPLY["still_request"]] * 2
    assert [c["name"] for c in fake_stills] == ["moon_still1", "moon_still2"]
    assert fake_stills[0]["size"] == (1920, 1088) and len({c["seed"] for c in fake_stills}) == 2
    data = json.loads((folder / "scene.json").read_text())
    assert data["prompt"] == "me on the moon, thin smile" and data["motion_prompt"] == REPLY["motion_prompt"]
    assert [s["file"] for s in data["stills"]] == ["stills/still1.png", "stills/still2.png"]
    assert data["caption"] == "A photograph of a person on the Moon."
    assert (folder / "stills" / "still1.graph.json").exists()
    assert "stills/still2.png" in (folder / "review.html").read_text()
    assert claude.calls[0][1] == "scene-moon" and made["name"] == "moon"
    assert (folder.stat().st_mode & 0o777) == 0o700
    assert any(line.startswith("refs") and "a.jpg" in line for line in lines)


def test_make_stills_refuses_a_scene_that_exists(cfg, tmp_path, fake_stills):
    (cfg.projects_dir / "scenes" / "moon").mkdir(parents=True)
    (cfg.projects_dir / "scenes" / "moon" / "scene.json").write_text("{}")
    with pytest.raises(scene.SceneError, match="exists"):
        scene.make_stills(cfg, name="moon", prompt="x", pictures=pictures(tmp_path), count=1, portrait=False,
                          claude=FakeClaude(), tools=TOOLS, echo=print)


def test_make_stills_stops_before_any_render_when_no_picture_is_usable(cfg, tmp_path, fake_stills):
    other = FakeTools({(1500, 1600): [face(600, 400, 500, ME)], (1500, 1700): [face(600, 400, 500, ME_TOO)]})
    with pytest.raises(scene.SceneError):
        scene.make_stills(cfg, name="none", prompt="x", pictures=[], count=1, portrait=False,
                          claude=FakeClaude(), tools=other, echo=print)
    assert fake_stills == []


def test_make_clips_animates_the_chosen_stills_with_the_saved_motion_prompt(cfg, tmp_path, fake_stills, monkeypatch):
    scene.make_stills(cfg, name="moon", prompt="me on the moon", pictures=pictures(tmp_path), count=2,
                      portrait=False, claude=FakeClaude(), tools=TOOLS, echo=print)
    calls = []

    def render_i2v(cfg, still, prompt, seconds, seed, size, name, dest, cleanup=False):
        calls.append(dict(still=still, prompt=prompt, seconds=seconds, seed=seed, size=size, name=name,
                          cleanup=cleanup))
        dest.write_bytes(b"mp4")
        return SimpleNamespace(clip=dest, elapsed_s=27.0, peak_gib=42.7, reused_nodes=0)

    monkeypatch.setattr("studio.scene.render_i2v", render_i2v)
    monkeypatch.setattr("studio.scene.prepare_still", lambda cfg, picture, out, size: picture)
    scene.make_clips(cfg, name="moon", numbers=[2], seconds=5, seed=7, echo=print)
    folder = cfg.projects_dir / "scenes" / "moon"
    assert calls == [dict(still=folder / "stills" / "still2.png", prompt=REPLY["motion_prompt"], seconds=5, seed=7,
                          size=(1280, 704), name="moon_still2", cleanup=True)]
    data = json.loads((folder / "scene.json").read_text())
    assert data["clips"] == [{"still": 2, "seed": 7, "seconds": 5, "file": "clips/still2-seed7.mp4",
                              "render_s": 27.0}]
    assert "clips/still2-seed7.mp4" in (folder / "review.html").read_text()


def test_make_clips_refuses_a_still_that_does_not_exist(cfg, tmp_path, fake_stills):
    scene.make_stills(cfg, name="moon", prompt="x", pictures=pictures(tmp_path), count=1, portrait=False,
                      claude=FakeClaude(), tools=TOOLS, echo=print)
    with pytest.raises(scene.SceneError, match="no still 3"):
        scene.make_clips(cfg, name="moon", numbers=[3], seconds=5, seed=7, echo=print)


def test_scene_names_are_plain():
    assert scene.clean_name("My Moon Walk!") == "My-Moon-Walk"
    with pytest.raises(scene.SceneError):
        scene.clean_name("../..")


def test_make_stills_can_run_again_after_the_prompt_call_failed(cfg, tmp_path, fake_stills):
    class Down:
        def create(self, route, *, film, **params):
            raise scene.scene_prompts.PromptError("no reply")

    with pytest.raises(scene.scene_prompts.PromptError):
        scene.make_stills(cfg, name="moon", prompt="x", pictures=pictures(tmp_path), count=1, portrait=False,
                          claude=Down(), tools=TOOLS, echo=print)
    made = scene.make_stills(cfg, name="moon", prompt="x", pictures=pictures(tmp_path), count=1, portrait=False,
                             claude=FakeClaude(), tools=TOOLS, echo=print)
    assert len(made["stills"]) == 1


def test_make_stills_takes_a_folder_of_pictures(cfg, tmp_path, fake_stills):
    folder = tmp_path / "pictures"
    folder.mkdir()
    pictures(folder)
    (folder / "readme.txt").write_text("not a picture")
    lines = []
    scene.make_stills(cfg, name="moon", prompt="x", pictures=[folder], count=1, portrait=False,
                      claude=FakeClaude(), tools=TOOLS, echo=lines.append)
    refs_dir = cfg.projects_dir / "scenes" / "moon" / "refs"
    assert sorted(p.name for p in refs_dir.iterdir()) == ["extra1.png", "extra1_square.png", "person1.png",
                                                          "person2.png"]
    assert fake_stills and not any("readme.txt" in line for line in lines)
