import contextlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from studio import film, provenance
from studio.config import Config
from studio.stills import Still
from tests.test_film_prompts import FakeClaude, plan
from tests.test_refs import write
from tests.test_scene import TOOLS, pictures


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path / "data")})


@pytest.fixture
def renders(monkeypatch):
    calls = []

    def render_still(cfg, references, request, *, seed, size, name, dest):
        calls.append(("still", name, request, [r.source for r in references]))
        Image.new("RGB", size, (10, 10, 10)).save(dest)
        return Still(path=dest, caption=f"caption {name}", elapsed_s=1.5, graph={"50": {}})

    def render_clip(cfg, variant, start, prompt, seconds, seed, name, dest, size):
        calls.append(("clip", name, prompt, variant, seconds, size, start.name))
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


def make(cfg, tmp_path, claude, gpu=contextlib.nullcontext, tools=TOOLS, **options):
    return film.make(cfg, name="ride", prompt="me riding a horse in front of the palace", pictures=pictures(tmp_path),
                     shots=3, seconds=5, portrait=False, claude=claude, tools=tools, echo=print, gpu=gpu, **options)


def test_make_draws_the_look_then_the_storyboard_from_it_then_the_takes(cfg, tmp_path, renders):
    made = make(cfg, tmp_path, FakeClaude(plan(3)), candidates=2, takes=2)
    stills = [c[1] for c in renders if c[0] == "still"]
    assert stills == ["ride_set1", "ride_set2", "ride_sheet1", "ride_sheet2",
                      *[f"ride_shot{n}-{c}" for n in (1, 2, 3) for c in (1, 2)]]
    clips = [(c[1], c[6]) for c in renders if c[0] == "clip"]
    assert clips == [(f"ride_shot{n}-{c}-{t}", f"shot{n}-{c}.png") for n in (1, 2, 3) for c in (1, 2) for t in (1, 2)]
    assert [c[0] for c in renders] == ["still"] * 10 + ["clip"] * 12 + ["film"]
    assert all(c[2:6] == (plan(3)["shots"][int(c[1][9]) - 1]["motion_prompt"], "wan-4step-1022", 5, (1280, 704))
               for c in renders if c[0] == "clip")
    # Tonight's film is joined from each shot's first still and first take; the rest wait for the user's picks.
    assert renders[-1] == ("film", ["shot1-1-1.mp4", "shot2-1-1.mp4", "shot3-1-1.mp4"], 5, 16)
    folder = cfg.projects_dir / "films" / "ride"
    data = json.loads((folder / "film.json").read_text())
    assert made == data and data["film"] == "film.mp4" and data["sound"] is None
    assert [c["file"] for c in data["look"]["set"]] == ["look/set1.png", "look/set2.png"]
    assert [c["file"] for c in data["look"]["sheet"]] == ["look/sheet1.png", "look/sheet2.png"]
    shot = data["shots"][1]
    assert [c["file"] for c in shot["stills"]] == ["stills/shot2-1.png", "stills/shot2-2.png"]
    assert [(t["still"], t["file"]) for t in shot["takes"]] == [(1, "clips/shot2-1-1.mp4"), (1, "clips/shot2-1-2.mp4"),
                                                                (2, "clips/shot2-2-1.mp4"), (2, "clips/shot2-2-2.mp4")]
    assert shot["still"] == "stills/shot2-1.png" and shot["clip"] == "clips/shot2-1-1.mp4"
    page = (folder / "review.html").read_text()
    assert all(f in page for f in ("film.mp4", "look/set2.png", "look/sheet1.png", "stills/shot3-2.png",
                                   "clips/shot3-2-2.mp4", "Ride at the Palace"))


def test_the_set_comes_from_the_place_pictures_the_sheet_from_the_photos_and_each_shot_from_both(cfg, tmp_path,
                                                                                                renders):
    make(cfg, tmp_path, FakeClaude(plan(3)), candidates=2, takes=1)
    refs = {c[1]: c[3] for c in renders if c[0] == "still"}
    assert refs["ride_set2"] == ["moon.png"]
    assert sorted(refs["ride_sheet1"]) == ["a.jpg", "b.jpg"]
    assert refs["ride_shot3-2"] == [*refs["ride_sheet1"], "look/sheet1.png", "look/set1.png"]
    requests = {c[1]: c[2] for c in renders if c[0] == "still"}
    assert requests["ride_set1"] == plan(3)["set_request"] and requests["ride_sheet2"] == plan(3)["sheet_request"]


def test_face_tools_see_only_the_input_pictures(cfg, tmp_path, renders):
    seen = []

    class Counting:
        def faces(self, bgr):
            seen.append(bgr.shape[:2])
            return TOOLS.faces(bgr)

    make(cfg, tmp_path, FakeClaude(plan(3)), tools=Counting(), candidates=2, takes=1)
    assert seen and set(seen) <= {(1500, 2000), (1500, 1800), (700, 1400)}  # never a 1920x1088 generated picture


def test_the_lineage_runs_from_the_photos_through_the_look_to_the_takes(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)), candidates=1, takes=1)
    folder = cfg.projects_dir / "films" / "ride"
    lineage = {f: json.loads(provenance.sidecar(folder / f).read_text())
               for f in ("look/set1.png", "look/sheet1.png", "stills/shot1-1.png", "clips/shot1-1-1.mp4")}
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["look/set1.png"]["inputs"]] == ["extra1.png"]
    assert len(lineage["look/sheet1.png"]["inputs"]) == 2
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["stills/shot1-1.png"]["inputs"]][-2:] == ["sheet1.png",
                                                                                                    "set1.png"]
    assert [m["set"] for m in lineage["stills/shot1-1.png"]["models"]] == ["flux3-klein"]
    assert lineage["clips/shot1-1-1.mp4"]["inputs"][0]["path"].endswith("stills/shot1-1.png")
    assert [m["set"] for m in lineage["clips/shot1-1-1.mp4"]["models"]] == ["wan22-i2v", "wan22-lightx2v-1022"]


def test_event_pictures_never_go_to_the_set_the_sheet_or_a_still(cfg, tmp_path, renders):
    trail = write(tmp_path / "trail.png", (1200, 800), exif=False)
    blast = write(tmp_path / "blast.png", (1300, 800), exif=False)
    claude = FakeClaude(plan(3))
    make(cfg, tmp_path, claude, candidates=1, takes=1, events=[trail, blast])
    refs = {c[1]: c[3] for c in renders if c[0] == "still"}
    assert refs["ride_set1"] == ["moon.png"]
    assert not any({"trail.png", "blast.png"} & set(r) for r in refs.values())
    text = claude.calls[0][2]["messages"][0]["content"][0]["text"]
    assert 'Event pictures: 1: file name "trail"; 2: file name "blast".' in text


def test_a_film_needs_a_picture_of_the_place_before_any_gpu_work(cfg, tmp_path, renders):
    with pytest.raises(film.FilmError, match="picture of the place"):
        film.make(cfg, name="ride", prompt="me riding a horse", pictures=pictures(tmp_path)[:2], shots=3, seconds=5,
                  portrait=False, claude=FakeClaude(plan(3)), tools=TOOLS, echo=print)
    assert renders == []


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
    assert padded == [(f"shot{n}-1-1.mp4", 5, 16) for n in (1, 2, 3)]
    assert not any(c.exists() for c, *_ in worker.calls)  # the padded clips are temporary
    assert [(c.name, p, s, seed) for c, p, s, seed in worker.calls] == \
        [(f"shot{n}-1-1.mp4", plan(3)["shots"][n - 1]["sound_prompt"], 6, 40 + n) for n in (1, 2, 3)]
    assert mixed == [("film.mp4", ["shot1.wav", "shot2.wav", "shot3.wav"], "film-sound.mp4", 5)]
    assert data["sound"]["film"] == "film-sound.mp4" and data["sound"]["true_peak"] == -1.8
    assert data["sound"]["shots"][0]["stereo"] == "narrowed" and data["shots"][0]["sound"] == "sound/shot1.wav"
    track = json.loads(provenance.sidecar(cfg.projects_dir / "films" / "ride" / "sound" / "shot1.wav").read_text())
    assert [m["set"] for m in track["models"]] == ["prismaudio"] and \
        track["inputs"][0]["path"].endswith("shot1-1-1.mp4")
    assert track["params"]["handle_s"] == 0.5 and track["params"]["seconds"] == 5
    assert "film-sound.mp4" in (cfg.projects_dir / "films" / "ride" / "review.html").read_text()
    assert any("shot 3: -16.1 LUFS in the film" in line and "stereo narrowed" in line for line in said)
    assert any("film -16.0 LUFS, true peak -1.8 dBTP" in line for line in said)


def test_add_sound_refuses_when_the_worker_is_not_ready(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)))
    with pytest.raises(film.FilmError, match="not ready: model set t5gemma is missing"):
        film.add_sound(cfg, name="ride", worker=FakeWorker(["model set t5gemma is missing"]), echo=print)
