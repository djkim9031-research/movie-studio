"""A short film from a prompt and pictures, look first: Claude's shot plan and continuity bible (text only), then set
pictures, character sheets and a picture of each view of the place, every shot's still drawn from them (an event
painted into its box), takes of each shot's picked still, and the film joined from the picks. Sound is added to a
finished film by a second step. Pictures and frames stay on this machine."""

import contextlib
import html
import json
import math
import re
import secrets
import tempfile
from pathlib import Path

from studio import bench, film_prompts, provenance, refs
from studio.bench import render as render_clip
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.media import assemble, triage
from studio.refs import MAX_REFERENCES, Reference
from studio.render import LANDSCAPE as CLIP_LANDSCAPE
from studio.render import PORTRAIT as CLIP_PORTRAIT
from studio.render import prepare_still
from studio.scene import clean_name, references
from studio.stills import LANDSCAPE, PORTRAIT, paint_event, render_still, upsample

VIDEO = "wan-4step-1022"
FPS = 16  # Wan 2.2's frame rate
MODEL_SETS = ["face-detect", "face-id", "flux3-klein", *bench.VARIANTS[VIDEO]["models"]]
ALL_REFERENCES = 30  # pictures a film may use in all; each still takes at most MAX_REFERENCES of them
LOCKED = {"wardrobe": "Clothes", "place": "Place", "light": "Light"}
SPEECH_MAX = 0.1  # share of a generated track heard as speech above which it is made again, then dropped
COMMON = {"with", "from", "that", "this", "under", "over", "into", "onto", "their", "there", "each", "very", "some",
          "while", "near", "about"}


class FilmError(RuntimeError):
    pass


def film_dir(cfg: Config, name: str) -> Path:
    return cfg.projects_dir / "films" / clean_name(name)


def _save(folder: Path, data: dict) -> None:
    write_private(folder / "film.json", json.dumps(data, indent=2) + "\n")
    write_private(folder / "review.html", review_page(data))


def load(cfg: Config, name: str) -> tuple[Path, dict]:
    folder = film_dir(cfg, name)
    if not (folder / "film.json").is_file():
        raise FilmError(f"no film {clean_name(name)} in {folder.parent}")
    return folder, json.loads((folder / "film.json").read_text(encoding="utf-8"))


def _models(cfg: Config, names: list[str]) -> list[str]:
    return [f"{n} {cfg.models[n].revision[:12]}" if n in cfg.models else n for n in names]


def _look_reference(path: Path) -> Reference:
    """A drawn picture as a reference for the stills drawn from it, padded to a square for the upsampler. It is never
    looked at for faces: the FLUX license bars biometric processing of what the model makes."""
    _, square = refs.extra_versions(refs.load(path))
    return Reference("extra", path, refs.save_png(square, path.with_name(f"{path.stem}_square.png")),
                     f"{path.parent.name}/{path.name}")


def fix_caption(caption: str, locked: dict[str, str]) -> tuple[str, list[str]]:
    """`caption` with every locked phrase it lost added back as a short sentence, and the names of those phrases. A
    phrase is lost when one of its words of four letters or more is missing from the caption."""
    low = caption.lower()
    lost = [name for name, phrase in locked.items()
            if any(w not in low for w in re.findall(r"[a-z]{4,}", phrase.lower()) if w not in COMMON)]
    if not lost:
        return caption, []
    return caption.rstrip() + " " + " ".join(f"{LOCKED[n]}: {locked[n].rstrip('.')}." for n in lost), lost


def _draw(cfg: Config, folder: Path, references: list[Reference], request: str, *, seed: int,
          size: tuple[int, int], name: str, dest: Path, locked: dict[str, str]) -> dict:
    """The upsampler's caption checked for the locked phrases and fixed, then the still drawn from it. Returns the
    still's entry for film.json."""
    caption, lost = fix_caption(upsample(cfg, references, request, name=name), locked)
    still = render_still(cfg, references, request, seed=seed, size=size, name=name, dest=dest, caption=caption)
    write_private(dest.with_suffix(".graph.json"), json.dumps(still.graph, indent=1) + "\n")
    provenance.write(cfg, dest, kind="still", models=["flux3-klein"], inputs=[r.file for r in references],
                     params={"seed": seed, "size": list(size)})
    entry = {"file": str(dest.relative_to(folder)), "seed": seed, "caption": still.caption,
             "s": round(still.elapsed_s, 1)}
    return entry | ({"caption_fixed": lost} if lost else {})


def _paint(cfg: Config, folder: Path, shot: dict, light: str, base: Path, dest: Path, events: list[Reference],
           name: str) -> dict:
    """The shot's event painted into its box on `base`, from the event pictures it names."""
    chosen = [events[i - 1] for i in shot["event_pictures"]]
    seed = secrets.randbelow(2**31)
    painted = paint_event(cfg, base, shot["event_box"], chosen, f"{shot['event'].rstrip('.')}. {light.rstrip('.')}.",
                          seed=seed, name=name, dest=dest)
    write_private(dest.with_suffix(".graph.json"), json.dumps(painted.graph, indent=1) + "\n")
    provenance.write(cfg, dest, kind="still", models=["flux3-klein"], inputs=[base, *(r.file for r in chosen)],
                     params={"seed": seed, "box": shot["event_box"]})
    return {"base": str(base.relative_to(folder)), "box": shot["event_box"]}


def _picked(shot: dict) -> int:
    return next(i for i, still in enumerate(shot["stills"], start=1) if still["file"] == shot["still"])


def _clip_size(data: dict) -> tuple[int, int]:
    return CLIP_PORTRAIT if data["orientation"] == "portrait" else CLIP_LANDSCAPE


def _takes(cfg: Config, folder: Path, data: dict, shot: dict, count: int, echo) -> None:
    """Takes of the shot's picked still until it has `count`, each with its triage numbers."""
    n, k = shot["n"], _picked(shot)
    for t in range(len([t for t in shot["takes"] if t["still"] == k]) + 1, count + 1):
        seed, dest = secrets.randbelow(2**31), private_dir(folder / "clips") / f"shot{n}-{k}-{t}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            start = prepare_still(cfg, folder / shot["still"], Path(tmp) / "start.png", _clip_size(data))
            result = render_clip(cfg, VIDEO, start, shot["motion_prompt"], data["seconds"], seed,
                                 f"{data['name']}_shot{n}-{k}-{t}", dest, size=_clip_size(data))
        provenance.write(cfg, dest, kind="clip", models=bench.VARIANTS[VIDEO]["models"],
                         inputs=[folder / shot["still"]], params={"seed": seed, "seconds": data["seconds"]})
        take = {"still": k, "take": t, "file": f"clips/{dest.name}", "seed": seed, "s": round(result.elapsed_s, 1),
                "triage": triage.clip(cfg, dest, data["fps"])}
        shot["takes"].append(take)
        if not any(t["file"] == shot["clip"] and t["still"] == k for t in shot["takes"]):
            shot["clip"] = take["file"]
        _save(folder, data)
        echo(f"clip  {n}/{len(data['shots'])}  still {k} take {t}/{count}  seed {seed}, {result.elapsed_s:.1f} s, "
             f"motion {take['triage']['motion']} -> {dest}")


def _rebuild(cfg: Config, folder: Path, data: dict, echo) -> None:
    """The silent film from the picks, once every shot has a take of its picked still; any sound is then stale."""
    waiting = [s for s in data["shots"] if not s.get("clip")]
    for s in waiting:
        echo(f"shot {s['n']}: still {_picked(s)} has no takes; run: uv run studio film takes --name {data['name']}")
    if waiting:
        data["film"] = None
    else:
        clips = [folder / s["clip"] for s in data["shots"]]
        assemble.silent_film(cfg, clips, folder / "film.mp4", seconds=data["seconds"], fps=data["fps"])
        provenance.write(cfg, folder / "film.mp4", kind="film", models=[], inputs=clips,
                         params={"seconds": data["seconds"]})
        data["film"] = "film.mp4"
    data["sound"] = None
    _save(folder, data)


def make(cfg: Config, *, name: str, prompt: str, pictures: list[Path], shots: int, seconds: int, portrait: bool,
         claude, tools, echo, gpu=contextlib.nullcontext, candidates: int = 3, takes: int = 4,
         events: list[Path] = ()) -> dict:
    """References and the plan first (no GPU). Then inside `gpu()`: `candidates` set pictures and character sheets,
    `candidates` pictures of each view drawn from the first set picture, `candidates` stills per shot drawn from the
    photos, the first sheet and the shot's view (with its event painted into its box), and `takes` clips of each
    shot's picked still (the first until the user picks), so the GPU swaps models once. Then the film from the
    picks; pick() and takes() change them later."""
    name = clean_name(name)
    folder = film_dir(cfg, name)
    if (folder / "film.json").exists():
        raise FilmError(f"film {name} exists in {folder}; pick another name")
    report = references([*pictures, *events], folder, tools, echo, limit=ALL_REFERENCES)
    shown = {Path(e).name for e in events}
    people = [r for r in report.used if r.kind == "person"]
    places = [r for r in report.used if r.kind == "extra" and not r.faces_blurred and r.source not in shown]
    event_refs = [r for r in report.used if r.kind == "extra" and r.source in shown]
    missing = shown - {r.source for r in event_refs}
    if not people:
        raise FilmError("a film needs at least one photo of the person")
    if not places:
        raise FilmError("the set is drawn from pictures of the place: add at least one picture of the place without "
                        "people")
    if missing:
        raise FilmError(f"event pictures not usable: {', '.join(sorted(missing))}")
    if len(places) > MAX_REFERENCES:
        raise FilmError(f"{len(places)} pictures of the place; one still takes at most {MAX_REFERENCES}")
    if len(event_refs) + 1 > MAX_REFERENCES:
        raise FilmError(f"{len(event_refs)} event pictures; an event is painted from at most {MAX_REFERENCES - 1}")
    orientation = "portrait" if portrait else "landscape"
    plan = film_prompts.write(claude, film=f"film-{name}", user_prompt=prompt, report=report,
                              orientation=orientation, shots=shots, seconds=seconds, events=shown)
    bible = plan.bible
    role = {id(r): "person" for r in people} | {id(r): "place" for r in places} | {id(r): "event" for r in event_refs}
    data = {"name": name, "prompt": prompt, "title": plan.title, "orientation": orientation, "seconds": seconds,
            "fps": FPS, "video": VIDEO, "candidates": candidates, "takes": takes,
            "references": [{"kind": r.kind, "role": role.get(id(r), "unused"), "file": f"refs/{r.file.name}",
                            "source": r.source} for r in report.used],
            "skipped": [{"source": s, "reason": why} for s, why in report.skipped],
            "models": {"stills": _models(cfg, ["flux3-klein"]),
                       "clips": _models(cfg, bench.VARIANTS[VIDEO]["models"])},
            "bible": bible.model_dump(),
            "look": {"set_request": plan.set_request, "sheet_request": plan.sheet_request, "set": [], "sheet": [],
                     "views": {v.id: [] for v in bible.views}},
            "shots": [{"n": n, **shot.model_dump(), "stills": [], "takes": [], "still": None, "clip": None}
                      for n, shot in enumerate(plan.shots, start=1)],
            "film": None, "sound": None}
    _save(folder, data)
    echo(f"plan  \"{plan.title}\", {shots} shots and {len(bible.views)} views by Claude, saved in "
         f"{folder / 'film.json'}")
    echo(f"look  set from {len(places)} pictures of the place, sheet from {len(people)} photos, "
         f"{len(event_refs)} event pictures to paint from")
    look_dir, stills_dir = private_dir(folder / "look"), private_dir(folder / "stills")
    size = PORTRAIT if portrait else LANDSCAPE
    words = {k: getattr(bible, k) for k in LOCKED}
    with gpu():
        for kind, chosen, request, locked in (("set", places, plan.set_request, ("place", "light")),
                                              ("sheet", people, plan.sheet_request, ("wardrobe",))):
            for c in range(1, candidates + 1):
                seed, dest = secrets.randbelow(2**31), look_dir / f"{kind}{c}.png"
                entry = _draw(cfg, folder, chosen, request, seed=seed, size=size, name=f"{name}_{kind}{c}",
                              dest=dest, locked={k: words[k] for k in locked})
                data["look"][kind].append(entry)
                _save(folder, data)
                echo(f"look  {kind} {c}/{candidates}  seed {seed}, {entry['s']} s -> {dest}")
        sheet, plate = _look_reference(look_dir / "sheet1.png"), _look_reference(look_dir / "set1.png")
        views = {}
        for view in bible.views:
            slug = clean_name(view.id)
            for c in range(1, candidates + 1):
                seed, dest = secrets.randbelow(2**31), look_dir / f"view-{slug}{c}.png"
                entry = _draw(cfg, folder, [plate], view.request, seed=seed, size=size, name=f"{name}_view-{slug}{c}",
                              dest=dest, locked={k: words[k] for k in ("place", "light")})
                data["look"]["views"][view.id].append(entry)
                _save(folder, data)
                echo(f"look  view {view.id} {c}/{candidates}  seed {seed}, {entry['s']} s -> {dest}")
            views[view.id] = _look_reference(look_dir / f"view-{slug}1.png")
        for shot in data["shots"]:
            n = shot["n"]
            person = shot["still_request"].strip().lower().startswith("create a photograph of the person")
            locked = {k: words[k] for k in (("wardrobe",) if person else ()) + ("place", "light")}
            for c in range(1, candidates + 1):
                seed, dest = secrets.randbelow(2**31), stills_dir / f"shot{n}-{c}.png"
                base = stills_dir / f"shot{n}-{c}-base.png" if shot["event"] else dest
                entry = _draw(cfg, folder, [*people, sheet, views[shot["view"]]], shot["still_request"], seed=seed,
                              size=size, name=f"{name}_shot{n}-{c}", dest=base, locked=locked)
                entry["file"] = str(dest.relative_to(folder))
                if shot["event"]:
                    entry["painted"] = _paint(cfg, folder, shot, bible.light, base, dest, event_refs,
                                              name=f"{name}_shot{n}-{c}-event")
                shot["stills"].append(entry)
                shot["still"] = shot["still"] or entry["file"]
                _save(folder, data)
                echo(f"still {n}/{shots}  candidate {c}/{candidates}  seed {seed}, {entry['s']} s -> {dest}")
        for shot in data["shots"]:
            _takes(cfg, folder, data, shot, takes, echo)
    _rebuild(cfg, folder, data, echo)
    return data


def pick(cfg: Config, *, name: str, choices: dict[int, tuple[int, int | None]], echo) -> dict:
    """`choices` maps a shot to (still, take); take None means the still's first take. The film is cut again from
    the picks once every shot has a take of its picked still."""
    folder, data = load(cfg, name)
    by = {s["n"]: s for s in data["shots"]}
    for n, (still, take) in sorted(choices.items()):
        if n not in by:
            raise FilmError(f"shot {n} is not in the film (shots 1-{len(by)})")
        if not 1 <= still <= len(by[n]["stills"]):
            raise FilmError(f"shot {n} has no still {still} (it has {len(by[n]['stills'])})")
        if take is not None and not any(t["still"] == still and t["take"] == take for t in by[n]["takes"]):
            raise FilmError(f"shot {n} has no take {take} of still {still}")
    for n, (still, take) in choices.items():
        shot = by[n]
        shot["still"] = shot["stills"][still - 1]["file"]
        mine = [t for t in shot["takes"] if t["still"] == still and (take is None or t["take"] == take)]
        shot["clip"] = mine[0]["file"] if mine else None
    _rebuild(cfg, folder, data, echo)
    return data


def takes(cfg: Config, *, name: str, count: int, echo, gpu=contextlib.nullcontext) -> dict:
    """Takes of every shot's picked still until each has `count`, then the film from the picks."""
    folder, data = load(cfg, name)
    with gpu():
        for shot in data["shots"]:
            _takes(cfg, folder, data, shot, count, echo)
    _rebuild(cfg, folder, data, echo)
    return data


def _gated(cfg: Config, worker, *, clip: Path, prompt: str, seconds: int, seed: int, dest: Path):
    """A track made and checked for speech, made once more with another seed when it sounds like speech. Returns the
    speech share, the seed used, the time taken and whether the track is clean."""
    for attempt in (0, 1):
        made = worker.render(cfg, clip=clip, prompt=prompt, seconds=seconds, seed=seed + 1000 * attempt, dest=dest)
        heard = worker.speech(cfg, dest)
        if heard < SPEECH_MAX:
            break
    return heard, seed + 1000 * attempt, made.elapsed_s, heard < SPEECH_MAX


def _bed_picture(data: dict) -> str:
    """The picture the bed is made from: the first shot's view, else the set picture, else its still."""
    look, first = data.get("look", {}), data["shots"][0]
    views = look.get("views", {})
    if views.get(first.get("view")):
        return views[first["view"]][0]["file"]
    return look["set"][0]["file"] if look.get("set") else first["still"]


def _bed(cfg: Config, folder: Path, data: dict, worker, base: int, sound_dir: Path, echo) -> tuple[Path, dict]:
    """The ambience under the whole film: BED_SEGMENT-second tracks made from a still picture of the place with the
    bible's bed prompt, each checked for speech, joined by crossfades."""
    length = len(data["shots"]) * data["seconds"]
    count = max(1, math.ceil((length - assemble.BED_CROSSFADE) / (assemble.BED_SEGMENT - assemble.BED_CROSSFADE)))
    prompt = (data.get("bible") or {}).get("bed") or data["shots"][0]["sound_prompt"]
    picture = folder / _bed_picture(data)
    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        still = assemble.still_clip(cfg, picture, Path(tmp) / "bed.mp4", seconds=assemble.BED_SEGMENT, fps=data["fps"])
        for i in range(count):
            dest = sound_dir / f"bed{i + 1}.wav"
            heard, used, _, clean = _gated(cfg, worker, clip=still, prompt=prompt, seconds=assemble.BED_SEGMENT,
                                           seed=base + 100 + i, dest=dest)
            provenance.write(cfg, dest, kind="sound", models=worker.models, inputs=[picture],
                             params={"seed": used, "seconds": assemble.BED_SEGMENT})
            segments.append({"file": f"sound/{dest.name}", "seed": used, "speech": heard})
            if not clean:
                echo(f"bed   segment {i + 1}: speech-like twice ({heard:.2f} of the track); kept, the film needs a bed")
    bed = assemble.join_bed(cfg, [folder / seg["file"] for seg in segments], sound_dir / "bed.wav")
    echo(f"bed   {count} segments of {assemble.BED_SEGMENT} s from {picture.name}, speech "
         + ", ".join(f"{seg['speech']:.2f}" for seg in segments))
    return bed, {"bed_prompt": prompt, "bed_segments": segments}


def add_sound(cfg: Config, *, name: str, worker, echo, seed: int | None = None, bed_file: Path | None = None) -> dict:
    """The film's sound in layers: one ambience bed under every cut (from `bed_file`, a recording, or made from a
    still picture of the place), Foley for each shot that has its own sound (made FOLEY_HANDLE seconds longer at both
    ends), every generated track checked for speech, and the film mixed as one programme."""
    folder, data = load(cfg, name)
    if not data.get("film"):
        raise FilmError(f"film {data['name']} has no finished picture yet")
    problems = worker.problems(cfg)
    if problems:
        raise FilmError(f"the sound worker is not ready: {'; '.join(problems)}")
    base = seed if seed is not None else secrets.randbelow(2**31 - 2000)
    sound_dir = private_dir(folder / "sound")
    seconds, shots = data["seconds"], data["shots"]
    if bed_file:
        bed = assemble.bed_from_file(cfg, bed_file, sound_dir / "bed.wav", seconds=len(shots) * seconds)
        about = {"bed_file": str(bed_file)}
        echo(f"bed   from {bed_file}")
    else:
        bed, about = _bed(cfg, folder, data, worker, base, sound_dir, echo)
    tracks = {}
    with tempfile.TemporaryDirectory() as tmp:
        for shot in shots:
            n = shot["n"]
            shot.update(sound=None, speech=None, dropped=None)
            if not shot.get("foley", True):
                continue
            dest, clip = sound_dir / f"shot{n}.wav", folder / shot["clip"]
            padded = assemble.padded_clip(cfg, clip, Path(tmp) / clip.name, seconds=seconds, fps=data["fps"],
                                          handle=assemble.FOLEY_HANDLE)
            heard, used, took, clean = _gated(cfg, worker, clip=padded, prompt=shot["sound_prompt"],
                                              seconds=seconds + int(2 * assemble.FOLEY_HANDLE), seed=base + n,
                                              dest=dest)
            padded.unlink()
            shot.update(sound_seed=used, sound_s=round(took, 1), speech=heard)
            if clean:
                provenance.write(cfg, dest, kind="sound", models=worker.models, inputs=[clip],
                                 params={"seed": used, "seconds": seconds, "handle_s": assemble.FOLEY_HANDLE})
                shot["sound"], tracks[n] = f"sound/{dest.name}", dest
                echo(f"foley shot {n}: seed {used}, speech {heard:.2f}, {took:.1f} s -> {dest}")
            else:
                shot["dropped"] = "speech"
                echo(f"foley shot {n}: speech-like twice ({heard:.2f} of the track); dropped, the bed carries the "
                     "shot")
            _save(folder, data)
    out = folder / "film-sound.mp4"
    mix = assemble.with_layers(cfg, folder / data["film"], bed, tracks, out, shots=len(shots), seconds=seconds)
    data["sound"] = {"film": out.name, **about, **mix}
    _save(folder, data)
    for n, s in enumerate(mix["shots"], start=1):
        echo(f"mix   shot {n}: {s['lufs']:.1f} LUFS in the film, "
             + (f"Foley gain {s['gain_db']:+.1f} dB, stereo {s['stereo']}" if s["foley"] else "bed only"))
    echo(f"mix   film {mix['lufs']:.1f} LUFS, true peak {mix['true_peak']:.1f} dBTP after AAC")
    return data


def review_page(data: dict) -> str:
    """A local page: the films, the look, then each shot's stills, takes and prompts, the picks marked. Never
    published: it shows real people."""
    e = html.escape

    def figure(label: str, media: str) -> str:
        return f"<figure><figcaption>{e(label)}</figcaption>{media}</figure>"

    def image(file: str) -> str:
        return f'<img src="{e(file)}" alt="{e(file)}" loading="lazy">'

    def video(file: str, sound: bool = False) -> str:
        return (f'<video src="{e(file)}" controls playsinline preload="metadata"'
                f'{"" if sound else " loop muted"}></video>')

    films = "".join(figure(label, video(file, sound=True)) for label, file in
                    (("Film", data.get("film")), ("Film with sound", (data.get("sound") or {}).get("film"))) if file)
    look = data.get("look", {})
    looks = "".join(figure(f"{kind.capitalize()} {i}{' (used)' if i == 1 else ''}", image(c["file"]))
                    for kind in ("set", "sheet") for i, c in enumerate(look.get(kind, []), start=1))
    looks += "".join(figure(f"View {view} {i}{' (used)' if i == 1 else ''}", image(c["file"]))
                     for view, made in look.get("views", {}).items() for i, c in enumerate(made, start=1))
    bible = data.get("bible", {})
    requests = (f'<details><summary>Look requests</summary><p><b>Set:</b> {e(look.get("set_request", ""))}</p>'
                f'<p><b>Character sheet:</b> {e(look.get("sheet_request", ""))}</p>'
                + "".join(f"<p><b>View {e(v['id'])}:</b> {e(v['request'])}</p>" for v in bible.get("views", []))
                + "".join(f"<p><b>{e(k.capitalize())}:</b> {e(bible[k])}</p>"
                          for k in ("wardrobe", "place", "light", "bed") if k in bible)
                + "</details>" if look else "")

    def numbers(t: dict) -> str:
        n = t.get("triage")
        return (f" · motion {n['motion']}, still {n['still_head_s']}/{n['still_tail_s']} s, color drift "
                f"{n['color_drift']}" if n else "")
    shots = []
    for s in data["shots"]:
        stills = "".join(figure(f"Still {i}{' (picked)' if c['file'] == s.get('still') else ''}", image(c["file"]))
                         for i, c in enumerate(s.get("stills", []), start=1))
        takes = "".join(figure(f"Still {t['still']}, take {t['take']}{' (picked)' if t['file'] == s.get('clip') else ''}"
                               f"{numbers(t)}", video(t["file"])) for t in s.get("takes", []))
        shots.append(f'<h2>Shot {s["n"]}</h2><div class="grid">{stills}{takes}</div>'
                     f'<details><summary>Prompts</summary>'
                     f'<p><b>Still request:</b> {e(s["still_request"])}</p>'
                     f'<p><b>Motion prompt:</b> {e(s["motion_prompt"])}</p>'
                     f'<p><b>Sound prompt:</b> {e(s["sound_prompt"])}</p>'
                     + (f'<p><b>View:</b> {e(s["view"])}; <b>ends:</b> {e(s["end_state"])}</p>' if "view" in s else "")
                     + (f'<p><b>Event:</b> {e(s["event"])} in {s["event_box"]}</p>' if s.get("event") else "")
                     + "</details>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(data["title"])}</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 28px 0 0; }}
  p, details {{ color: var(--muted); max-width: 80ch; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap: 16px;
          margin-top: 12px; }}
  figure {{ margin: 0; background: var(--card); border-radius: 10px; padding: 12px; }}
  figcaption {{ font-weight: 600; margin-bottom: 8px; color: var(--text); }}
  img, video {{ display: block; width: 100%; height: auto; border-radius: 6px; background: #000; }}
</style>
</head>
<body>
<h1>{e(data["title"])}</h1>
<p>{e(data["prompt"])}</p>
<div class="grid">
{films}
</div>
<h2>Look</h2>
<div class="grid">
{looks}
</div>
{requests}
{chr(10).join(shots)}
</body>
</html>
"""
