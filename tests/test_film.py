import contextlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from studio import film, provenance
from studio.config import Config
from studio.stills import Still
from tests.test_film_prompts import LIGHT, PLACE, WARDROBE, FakeClaude, plan
from tests.test_refs import write
from tests.test_scene import TOOLS, pictures


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path / "data")})


TRIAGE = {"motion": 1.2, "still_head_s": 0.0, "still_tail_s": 0.25, "color_drift": 0.8}


@pytest.fixture
def renders(monkeypatch):
    calls = []

    def upsample(cfg, references, request, *, name):
        calls.append(("upsample", name))
        return f"Caption: {request}"

    def render_still(cfg, references, request, *, seed, size, name, dest, caption=None):
        calls.append(("still", name, request, [r.source for r in references], caption))
        Image.new("RGB", size, (10, 10, 10)).save(dest)
        return Still(path=dest, caption=caption or f"caption {name}", elapsed_s=1.5, graph={"50": {}})

    def paint_event(cfg, base, box, references, caption, *, seed, name, dest):
        calls.append(("paint", name, base.name, box, [r.source for r in references], caption))
        Image.open(base).save(dest)
        return Still(path=dest, caption=caption, elapsed_s=2.0, graph={"60": {}})

    def render_clip(cfg, variant, start, prompt, seconds, seed, name, dest, size):
        calls.append(("clip", name, prompt, variant, seconds, size, start.name))
        dest.write_bytes(b"mp4")
        return SimpleNamespace(clip=dest, elapsed_s=27.0, peak_gib=42.7, reused_nodes=0)

    def silent_film(cfg, clips, out, *, seconds, fps):
        calls.append(("film", [c.name for c in clips], seconds, fps))
        out.write_bytes(b"film")

    monkeypatch.setattr("studio.film.upsample", upsample)
    monkeypatch.setattr("studio.film.render_still", render_still)
    monkeypatch.setattr("studio.film.paint_event", paint_event)
    monkeypatch.setattr("studio.film.render_clip", render_clip)
    monkeypatch.setattr("studio.film.prepare_still", lambda cfg, picture, out, size: picture)
    monkeypatch.setattr("studio.film.triage.clip", lambda cfg, clip, fps: dict(TRIAGE))
    monkeypatch.setattr("studio.film.assemble.silent_film", silent_film)
    return calls


def make(cfg, tmp_path, claude, gpu=contextlib.nullcontext, tools=TOOLS, **options):
    return film.make(cfg, name="ride", prompt="me riding a horse in front of the palace", pictures=pictures(tmp_path),
                     shots=3, seconds=5, portrait=False, claude=claude, tools=tools, echo=print, gpu=gpu, **options)


def two_views(count=3):
    """plan(count) with a second view, the hall, for shot 2."""
    told = plan(count)
    told["bible"]["views"].append({"id": "hall", "request": f"Create a photograph of the place shown in image 1, "
                                                            f"looking at the hall: {PLACE}, {LIGHT}, no people"})
    told["shots"][1]["view"] = "hall"
    return told


def test_make_draws_the_look_and_views_then_each_shot_from_its_view_then_takes_of_the_picked_stills(cfg, tmp_path,
                                                                                                    renders):
    made = make(cfg, tmp_path, FakeClaude(plan(3)), candidates=2, takes=3)
    stills = [c[1] for c in renders if c[0] == "still"]
    assert stills == ["ride_set1", "ride_set2", "ride_sheet1", "ride_sheet2", "ride_view-gate1", "ride_view-gate2",
                      *[f"ride_shot{n}-{c}" for n in (1, 2, 3) for c in (1, 2)]]
    assert [c[1] for c in renders if c[0] == "upsample"] == stills  # every caption is read before drawing
    clips = [(c[1], c[6]) for c in renders if c[0] == "clip"]
    assert clips == [(f"ride_shot{n}-1-{t}", f"shot{n}-1.png") for n in (1, 2, 3) for t in (1, 2, 3)]
    assert renders[-1] == ("film", ["shot1-1-1.mp4", "shot2-1-1.mp4", "shot3-1-1.mp4"], 5, 16)
    folder = cfg.projects_dir / "films" / "ride"
    data = json.loads((folder / "film.json").read_text())
    assert made == data and data["film"] == "film.mp4" and data["sound"] is None
    assert data["bible"]["wardrobe"] == WARDROBE
    assert [c["file"] for c in data["look"]["views"]["gate"]] == ["look/view-gate1.png", "look/view-gate2.png"]
    shot = data["shots"][1]
    assert shot["view"] == "gate" and [c["file"] for c in shot["stills"]] == ["stills/shot2-1.png", "stills/shot2-2.png"]
    assert [(t["still"], t["take"], t["file"]) for t in shot["takes"]] == [(1, t, f"clips/shot2-1-{t}.mp4") for t in (1, 2, 3)]
    assert shot["takes"][0]["triage"] == TRIAGE
    assert shot["still"] == "stills/shot2-1.png" and shot["clip"] == "clips/shot2-1-1.mp4"
    page = (folder / "review.html").read_text()
    assert all(f in page for f in ("film.mp4", "look/set2.png", "look/sheet1.png", "look/view-gate2.png",
                                   "stills/shot3-2.png", "clips/shot3-1-3.mp4", "motion 1.2", "Ride at the Palace"))


def test_views_come_from_the_first_set_picture_and_each_shot_from_its_own_view(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(two_views()), candidates=1, takes=1)
    refs = {c[1]: c[3] for c in renders if c[0] == "still"}
    assert refs["ride_set1"] == ["moon.png"] and sorted(refs["ride_sheet1"]) == ["a.jpg", "b.jpg"]
    assert refs["ride_view-gate1"] == ["look/set1.png"] and refs["ride_view-hall1"] == ["look/set1.png"]
    assert refs["ride_shot1-1"] == [*refs["ride_sheet1"], "look/sheet1.png", "look/view-gate1.png"]
    assert refs["ride_shot2-1"] == [*refs["ride_sheet1"], "look/sheet1.png", "look/view-hall1.png"]


def test_a_caption_that_drops_a_locked_phrase_gets_it_back_before_drawing(cfg, tmp_path, renders, monkeypatch):
    monkeypatch.setattr("studio.film.upsample", lambda cfg, references, request, *, name: "A rider by a gate.")
    make(cfg, tmp_path, FakeClaude(plan(3)), candidates=1, takes=1)
    captions = {c[1]: c[4] for c in renders if c[0] == "still"}
    assert captions["ride_set1"] == f"A rider by a gate. Place: {PLACE}. Light: {LIGHT}."
    assert captions["ride_sheet1"] == f"A rider by a gate. Clothes: {WARDROBE}."
    assert captions["ride_shot2-1"] == f"A rider by a gate. Clothes: {WARDROBE}. Place: {PLACE}. Light: {LIGHT}."
    data = json.loads((cfg.projects_dir / "films" / "ride" / "film.json").read_text())
    assert data["shots"][1]["stills"][0]["caption_fixed"] == ["wardrobe", "place", "light"]
    assert data["look"]["set"][0]["caption_fixed"] == ["place", "light"]


def test_an_event_is_painted_into_its_box_and_the_clip_starts_from_the_painted_still(cfg, tmp_path, renders):
    blast = write(tmp_path / "blast.png", (1300, 800), exif=False)
    told = plan(3)
    phrase = "a small orange flash on the horizon line in the right third, about 4% of the frame width"
    told["shots"][1].update(event=phrase, event_box=[0.70, 0.30, 0.74, 0.36], event_pictures=[1])
    make(cfg, tmp_path, FakeClaude(told), candidates=1, takes=1, events=[blast])
    paints = [c for c in renders if c[0] == "paint"]
    assert len(paints) == 1
    _, name, base, box, refs, caption = paints[0]
    assert (name, base, box, refs) == ("ride_shot2-1-event", "shot2-1-base.png", [0.70, 0.30, 0.74, 0.36],
                                       ["blast.png"])
    assert phrase in caption and LIGHT in caption
    starts = {c[1]: c[6] for c in renders if c[0] == "clip"}
    assert starts["ride_shot2-1-1"] == "shot2-1.png"
    folder = cfg.projects_dir / "films" / "ride"
    assert (folder / "stills" / "shot2-1-base.png").exists()
    shot = json.loads((folder / "film.json").read_text())["shots"][1]
    assert shot["stills"][0]["painted"] == {"base": "stills/shot2-1-base.png", "box": [0.70, 0.30, 0.74, 0.36]}
    lineage = json.loads(provenance.sidecar(folder / "stills" / "shot2-1.png").read_text())
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["inputs"]] == ["shot2-1-base.png", "extra2.png"]


def test_pick_changes_the_film_and_takes_fill_in_what_a_pick_needs(cfg, tmp_path, renders):
    make(cfg, tmp_path, FakeClaude(plan(3)), candidates=2, takes=1)
    said = []
    film.pick(cfg, name="ride", choices={2: (2, None)}, echo=said.append)
    data = json.loads((cfg.projects_dir / "films" / "ride" / "film.json").read_text())
    assert data["shots"][1]["still"] == "stills/shot2-2.png" and data["shots"][1]["clip"] is None
    assert data["film"] is None and any("shot 2: still 2 has no takes" in line for line in said)
    del renders[:]
    film.takes(cfg, name="ride", count=2, echo=said.append)
    assert [c[1] for c in renders if c[0] == "clip"] == ["ride_shot1-1-2", "ride_shot2-2-1", "ride_shot2-2-2",
                                                         "ride_shot3-1-2"]
    assert renders[-1] == ("film", ["shot1-1-1.mp4", "shot2-2-1.mp4", "shot3-1-1.mp4"], 5, 16)
    film.pick(cfg, name="ride", choices={3: (1, 2)}, echo=said.append)
    assert renders[-1] == ("film", ["shot1-1-1.mp4", "shot2-2-1.mp4", "shot3-1-2.mp4"], 5, 16)
    with pytest.raises(film.FilmError, match="shot 3 has no take 5 of still 1"):
        film.pick(cfg, name="ride", choices={3: (1, 5)}, echo=said.append)
    with pytest.raises(film.FilmError, match="shot 4 is not in the film"):
        film.pick(cfg, name="ride", choices={4: (1, None)}, echo=said.append)


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
               for f in ("look/set1.png", "look/sheet1.png", "look/view-gate1.png", "stills/shot1-1.png",
                         "clips/shot1-1-1.mp4")}
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["look/set1.png"]["inputs"]] == ["extra1.png"]
    assert len(lineage["look/sheet1.png"]["inputs"]) == 2
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["look/view-gate1.png"]["inputs"]] == ["set1.png"]
    assert [i["path"].rsplit("/", 1)[1] for i in lineage["stills/shot1-1.png"]["inputs"]][-2:] == ["sheet1.png",
                                                                                                    "view-gate1.png"]
    assert lineage["clips/shot1-1-1.mp4"]["inputs"][0]["path"].endswith("stills/shot1-1.png")
    assert [m["set"] for m in lineage["clips/shot1-1-1.mp4"]["models"]] == ["wan22-i2v", "wan22-lightx2v-1022"]


def test_event_pictures_never_go_to_the_set_the_sheet_the_views_or_a_drawn_still(cfg, tmp_path, renders):
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
