import contextlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from studio import film, provenance
from studio.config import Config
from studio.stills import Still
from tests.test_film_prompts import FakeClaude, plan
from tests.test_scene import TOOLS, pictures


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path / "data")})


@pytest.fixture
def renders(monkeypatch):
    calls = []

    def render_still(cfg, references, request, *, seed, size, name, dest):
        calls.append(("still", name, request))
        Image.new("RGB", size, (10, 10, 10)).save(dest)
        return Still(path=dest, caption="caption", elapsed_s=1.5, graph={"50": {}})

    def render_clip(cfg, variant, start, prompt, seconds, seed, name, dest, size):
        calls.append(("clip", name, prompt, variant, seconds, size))
        dest.write_bytes(b"mp4")
        return SimpleNamespace(clip=dest, elapsed_s=27.0, peak_gib=42.7, reused_nodes=0)

    def silent_film(cfg, clips, out, *, seconds, fps):
        calls.append(("film", [c.name for c in clips], seconds, fps))
        out.write_bytes(b"film")

    monkeypatch.setattr("studio.film.render_still", render_still)
    monkeypatch.setattr("studio.film.render_clip", render_clip)
    monkeypatch.setattr("studio.film.prepare_still", lambda cfg, picture, out, size: picture)
    monkeypatch.setattr("studio.film.assemble.silent_film", silent_film)
    return calls


def make(cfg, tmp_path, claude, gpu=contextlib.nullcontext):
    return film.make(cfg, name="ride", prompt="me riding a horse in front of the palace", pictures=pictures(tmp_path),
                     shots=3, seconds=5, portrait=False, claude=claude, tools=TOOLS, echo=print, gpu=gpu)


def test_make_draws_every_still_before_any_clip_then_joins_the_clips_in_shot_order(cfg, tmp_path, renders):
    made = make(cfg, tmp_path, FakeClaude(plan(3)))
    assert [c[0] for c in renders] == ["still"] * 3 + ["clip"] * 3 + ["film"]
    assert [c[1] for c in renders[:6]] == ["ride_shot1", "ride_shot2", "ride_shot3"] * 2
    assert [c[2] for c in renders[:3]] == [s["still_request"] for s in plan(3)["shots"]]
    assert [c[2] for c in renders[3:6]] == [s["motion_prompt"] for s in plan(3)["shots"]]
    assert all(c[3:] == ("wan-4step-1022", 5, (1280, 704)) for c in renders[3:6])
    assert renders[6] == ("film", ["shot1.mp4", "shot2.mp4", "shot3.mp4"], 5, 16)
    folder = cfg.projects_dir / "films" / "ride"
    data = json.loads((folder / "film.json").read_text())
    assert made == data and data["title"] == "Ride at the Palace"
    assert [s["sound_prompt"] for s in data["shots"]] == [s["sound_prompt"] for s in plan(3)["shots"]]
    assert data["shots"][2]["still"] == "stills/shot3.png" and data["shots"][2]["clip"] == "clips/shot3.mp4"
    assert data["film"] == "film.mp4" and data["sound"] is None
    page = (folder / "review.html").read_text()
    assert "film.mp4" in page and "clips/shot3.mp4" in page and "Ride at the Palace" in page


def test_stills_and_clips_carry_their_lineage(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)))
    folder = cfg.projects_dir / "films" / "ride"
    clip = json.loads(provenance.sidecar(folder / "clips" / "shot1.mp4").read_text())
    assert [m["set"] for m in clip["models"]] == ["wan22-i2v", "wan22-lightx2v-1022"]
    assert clip["inputs"][0]["path"].endswith("stills/shot1.png")
    still = json.loads(provenance.sidecar(folder / "stills" / "shot1.png").read_text())
    assert [m["set"] for m in still["models"]] == ["flux3-klein"] and len(still["inputs"]) == 3


def test_the_shot_plan_is_saved_before_the_gpu_starts(cfg, tmp_path, renders):
    seen = []

    @contextlib.contextmanager
    def gpu():
        seen.append(json.loads((cfg.projects_dir / "films" / "ride" / "film.json").read_text())["title"])
        yield

    make(cfg, tmp_path, FakeClaude(plan(3)), gpu=gpu)
    assert seen == ["Ride at the Palace"]


def test_make_refuses_a_film_name_that_exists(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)))
    with pytest.raises(film.FilmError, match="exists"):
        make(cfg, tmp_path, FakeClaude(plan(3)))


class FakeWorker:
    models = ["prismaudio"]

    def __init__(self, problems=()):
        self.missing, self.calls = list(problems), []

    def problems(self, cfg):
        return self.missing

    def render(self, cfg, *, clip, prompt, seconds, seed, dest):
        self.calls.append((clip, prompt, seconds, seed))
        dest.write_bytes(b"wav")
        return SimpleNamespace(elapsed_s=3.0)


SHOT_MIX = {"lufs_made": -24.0, "gain_db": 8.0, "stereo": "narrowed", "width": 0.8, "correlation_before": -0.2,
            "correlation_after": 0.1, "lufs": -16.1}


def test_add_sound_makes_each_track_with_handles_then_mixes_them(cfg, tmp_path, renders, monkeypatch):
    make(cfg, tmp_path, FakeClaude(plan(3)))
    padded, mixed, said = [], [], []

    def padded_clip(cfg, clip, out, *, seconds, fps):
        padded.append((clip.name, seconds, fps))
        out.write_bytes(b"padded")
        return out

    def with_sound(cfg, silent, tracks, out, *, seconds):
        mixed.append((silent.name, [t.name for t in tracks], out.name, seconds))
        out.write_bytes(b"film with sound")
        return {"lufs": -16.0, "true_peak": -1.8, "gain_db": 0.4, "shots": [SHOT_MIX] * 3}

    monkeypatch.setattr("studio.film.assemble.padded_clip", padded_clip)
    monkeypatch.setattr("studio.film.assemble.with_sound", with_sound)
    worker = FakeWorker()
    data = film.add_sound(cfg, name="ride", worker=worker, echo=said.append, seed=40)
    # The worker hears each shot from its padded clip, 0.5 s longer at both ends: 5 s shots become 6 s tracks.
    assert padded == [(f"shot{n}.mp4", 5, 16) for n in (1, 2, 3)]
    assert not any(c.exists() for c, *_ in worker.calls)  # the padded clips are temporary
    assert [(c.name, p, s, seed) for c, p, s, seed in worker.calls] == \
        [(f"shot{n}.mp4", plan(3)["shots"][n - 1]["sound_prompt"], 6, 40 + n) for n in (1, 2, 3)]
    assert mixed == [("film.mp4", ["shot1.wav", "shot2.wav", "shot3.wav"], "film-sound.mp4", 5)]
    assert data["sound"]["film"] == "film-sound.mp4" and data["sound"]["true_peak"] == -1.8
    assert data["sound"]["shots"][0]["stereo"] == "narrowed" and data["shots"][0]["sound"] == "sound/shot1.wav"
    track = json.loads(provenance.sidecar(cfg.projects_dir / "films" / "ride" / "sound" / "shot1.wav").read_text())
    assert [m["set"] for m in track["models"]] == ["prismaudio"] and track["inputs"][0]["path"].endswith("shot1.mp4")
    assert track["params"]["handle_s"] == 0.5 and track["params"]["seconds"] == 5
    assert "film-sound.mp4" in (cfg.projects_dir / "films" / "ride" / "review.html").read_text()
    assert any("shot 3: -16.1 LUFS in the film" in line and "stereo narrowed" in line for line in said)
    assert any("film -16.0 LUFS, true peak -1.8 dBTP" in line for line in said)


def test_add_sound_refuses_when_the_worker_is_not_ready(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)))
    with pytest.raises(film.FilmError, match="not ready: model set t5gemma is missing"):
        film.add_sound(cfg, name="ride", worker=FakeWorker(["model set t5gemma is missing"]), echo=print)
