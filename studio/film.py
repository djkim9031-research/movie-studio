"""A short film from a prompt and pictures, look first: Claude's shot plan (text only), then a set picture and a
character sheet, every shot's still drawn from them, several takes per still, and the film joined from the picks.
Sound is added to a finished film by a second step. Pictures and frames stay on this machine."""

import contextlib
import html
import json
import secrets
import tempfile
from pathlib import Path

from studio import bench, film_prompts, provenance, refs
from studio.bench import render as render_clip
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.media import assemble
from studio.refs import MAX_REFERENCES, Reference
from studio.render import LANDSCAPE as CLIP_LANDSCAPE
from studio.render import PORTRAIT as CLIP_PORTRAIT
from studio.render import prepare_still
from studio.scene import clean_name, references
from studio.stills import LANDSCAPE, PORTRAIT, render_still

VIDEO = "wan-4step-1022"
FPS = 16  # Wan 2.2's frame rate
MODEL_SETS = ["face-detect", "face-id", "flux3-klein", *bench.VARIANTS[VIDEO]["models"]]
ALL_REFERENCES = 30  # pictures a film may use in all; each still takes at most MAX_REFERENCES of them


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


def _draw(cfg: Config, references: list[Reference], request: str, *, seed: int, size: tuple[int, int], name: str,
          dest: Path):
    still = render_still(cfg, references, request, seed=seed, size=size, name=name, dest=dest)
    write_private(dest.with_suffix(".graph.json"), json.dumps(still.graph, indent=1) + "\n")
    provenance.write(cfg, dest, kind="still", models=["flux3-klein"], inputs=[r.file for r in references],
                     params={"seed": seed, "size": list(size)})
    return still


def make(cfg: Config, *, name: str, prompt: str, pictures: list[Path], shots: int, seconds: int, portrait: bool,
         claude, tools, echo, gpu=contextlib.nullcontext, candidates: int = 3, takes: int = 2,
         events: list[Path] = ()) -> dict:
    """References and the shot plan first (no GPU). Then inside `gpu()`: `candidates` set pictures and character
    sheets, `candidates` stills per shot drawn from the first set picture and sheet (plus the event pictures), and
    `takes` clips per still, so the GPU swaps models once. The film joins each shot's first still's first take; the
    other candidates and takes wait for the user's picks."""
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
    if len(event_refs) > MAX_REFERENCES:
        raise FilmError(f"{len(event_refs)} event pictures; an event is painted from at most {MAX_REFERENCES}")
    orientation = "portrait" if portrait else "landscape"
    plan = film_prompts.write(claude, film=f"film-{name}", user_prompt=prompt, report=report,
                              orientation=orientation, shots=shots, seconds=seconds, events=shown)
    role = {id(r): "person" for r in people} | {id(r): "place" for r in places} | {id(r): "event" for r in event_refs}
    data = {"name": name, "prompt": prompt, "title": plan.title, "orientation": orientation, "seconds": seconds,
            "fps": FPS, "video": VIDEO, "candidates": candidates, "takes": takes,
            "references": [{"kind": r.kind, "role": role.get(id(r), "unused"), "file": f"refs/{r.file.name}",
                            "source": r.source} for r in report.used],
            "skipped": [{"source": s, "reason": why} for s, why in report.skipped],
            "models": {"stills": _models(cfg, ["flux3-klein"]),
                       "clips": _models(cfg, bench.VARIANTS[VIDEO]["models"])},
            "look": {"set_request": plan.set_request, "sheet_request": plan.sheet_request, "set": [], "sheet": []},
            "shots": [{"n": n, **shot.model_dump(), "stills": [], "takes": [], "still": None, "clip": None}
                      for n, shot in enumerate(plan.shots, start=1)],
            "film": None, "sound": None}
    _save(folder, data)
    echo(f"plan  \"{plan.title}\", {shots} shots by Claude, saved in {folder / 'film.json'}")
    echo(f"look  set from {len(places)} pictures of the place, sheet from {len(people)} photos, "
         f"{len(event_refs)} event pictures for the shots")
    look_dir, stills_dir, clips_dir = (private_dir(folder / d) for d in ("look", "stills", "clips"))
    size = PORTRAIT if portrait else LANDSCAPE
    clip_size = CLIP_PORTRAIT if portrait else CLIP_LANDSCAPE
    with gpu():
        for kind, chosen, request in (("set", places, plan.set_request), ("sheet", people, plan.sheet_request)):
            for c in range(1, candidates + 1):
                seed, dest = secrets.randbelow(2**31), look_dir / f"{kind}{c}.png"
                still = _draw(cfg, chosen, request, seed=seed, size=size, name=f"{name}_{kind}{c}", dest=dest)
                data["look"][kind].append({"file": f"look/{dest.name}", "seed": seed, "caption": still.caption,
                                           "s": round(still.elapsed_s, 1)})
                _save(folder, data)
                echo(f"look  {kind} {c}/{candidates}  seed {seed}, {still.elapsed_s:.1f} s -> {dest}")
        anchors = [_look_reference(look_dir / "sheet1.png"), _look_reference(look_dir / "set1.png")]
        for shot in data["shots"]:
            n = shot["n"]
            for c in range(1, candidates + 1):
                seed, dest = secrets.randbelow(2**31), stills_dir / f"shot{n}-{c}.png"
                still = _draw(cfg, [*people, *anchors], shot["still_request"], seed=seed, size=size,
                              name=f"{name}_shot{n}-{c}", dest=dest)
                shot["stills"].append({"file": f"stills/{dest.name}", "seed": seed, "caption": still.caption,
                                       "s": round(still.elapsed_s, 1)})
                shot["still"] = shot["stills"][0]["file"]
                _save(folder, data)
                echo(f"still {n}/{shots}  candidate {c}/{candidates}  seed {seed}, {still.elapsed_s:.1f} s -> {dest}")
        for shot in data["shots"]:
            n = shot["n"]
            for c, drawn in enumerate(shot["stills"], start=1):
                for t in range(1, takes + 1):
                    seed, dest = secrets.randbelow(2**31), clips_dir / f"shot{n}-{c}-{t}.mp4"
                    with tempfile.TemporaryDirectory() as tmp:
                        start = prepare_still(cfg, folder / drawn["file"], Path(tmp) / "start.png", clip_size)
                        result = render_clip(cfg, VIDEO, start, shot["motion_prompt"], seconds, seed,
                                             f"{name}_shot{n}-{c}-{t}", dest, size=clip_size)
                    provenance.write(cfg, dest, kind="clip", models=bench.VARIANTS[VIDEO]["models"],
                                     inputs=[folder / drawn["file"]], params={"seed": seed, "seconds": seconds})
                    shot["takes"].append({"still": c, "take": t, "file": f"clips/{dest.name}", "seed": seed,
                                          "s": round(result.elapsed_s, 1)})
                    shot["clip"] = shot["takes"][0]["file"]
                    _save(folder, data)
                    echo(f"clip  {n}/{shots}  still {c} take {t}/{takes}  seed {seed}, {result.elapsed_s:.1f} s -> "
                         f"{dest}")
    clips = [folder / shot["clip"] for shot in data["shots"]]
    assemble.silent_film(cfg, clips, folder / "film.mp4", seconds=seconds, fps=FPS)
    provenance.write(cfg, folder / "film.mp4", kind="film", models=[], inputs=clips, params={"seconds": seconds})
    data["film"] = "film.mp4"
    _save(folder, data)
    return data


def add_sound(cfg: Config, *, name: str, worker, echo, seed: int | None = None) -> dict:
    """One track per shot from its clip and sound prompt, made assemble.HANDLE seconds longer at both ends so the
    mix can crossfade at the cuts, then the film again with the tracks mixed in."""
    folder, data = load(cfg, name)
    if not data.get("film"):
        raise FilmError(f"film {data['name']} has no finished picture yet")
    problems = worker.problems(cfg)
    if problems:
        raise FilmError(f"the sound worker is not ready: {'; '.join(problems)}")
    base = seed if seed is not None else secrets.randbelow(2**31 - 100)
    sound_dir = private_dir(folder / "sound")
    seconds = data["seconds"]
    with tempfile.TemporaryDirectory() as tmp:
        for shot in data["shots"]:
            n = shot["n"]
            dest = sound_dir / f"shot{n}.wav"
            clip = folder / shot["clip"]
            padded = assemble.padded_clip(cfg, clip, Path(tmp) / clip.name, seconds=seconds, fps=data["fps"])
            made = worker.render(cfg, clip=padded, prompt=shot["sound_prompt"],
                                 seconds=seconds + int(2 * assemble.HANDLE), seed=base + n, dest=dest)
            padded.unlink()
            provenance.write(cfg, dest, kind="sound", models=worker.models, inputs=[clip],
                             params={"seed": base + n, "seconds": seconds, "handle_s": assemble.HANDLE})
            shot.update(sound=f"sound/{dest.name}", sound_seed=base + n, sound_s=round(made.elapsed_s, 1))
            _save(folder, data)
            echo(f"sound {n}/{len(data['shots'])}  seed {base + n}, {made.elapsed_s:.1f} s -> {dest}")
    out = folder / "film-sound.mp4"
    mix = assemble.with_sound(cfg, folder / data["film"], [folder / s["sound"] for s in data["shots"]], out,
                              seconds=seconds)
    data["sound"] = {"film": out.name, **mix}
    _save(folder, data)
    for n, s in enumerate(mix["shots"], start=1):
        echo(f"mix   shot {n}: {s['lufs']:.1f} LUFS in the film (made at {s['lufs_made']:.1f}, gain "
             f"{s['gain_db']:+.1f} dB), stereo {s['stereo']}, correlation {s['correlation_before']:+.2f} -> "
             f"{s['correlation_after']:+.2f}")
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
    requests = (f'<details><summary>Look requests</summary><p><b>Set:</b> {e(look.get("set_request", ""))}</p>'
                f'<p><b>Character sheet:</b> {e(look.get("sheet_request", ""))}</p></details>' if look else "")
    shots = []
    for s in data["shots"]:
        stills = "".join(figure(f"Still {i}{' (picked)' if c['file'] == s.get('still') else ''}", image(c["file"]))
                         for i, c in enumerate(s.get("stills", []), start=1))
        takes = "".join(figure(f"Still {t['still']}, take {t['take']}{' (picked)' if t['file'] == s.get('clip') else ''}",
                               video(t["file"])) for t in s.get("takes", []))
        shots.append(f'<h2>Shot {s["n"]}</h2><div class="grid">{stills}{takes}</div>'
                     f'<details><summary>Prompts</summary>'
                     f'<p><b>Still request:</b> {e(s["still_request"])}</p>'
                     f'<p><b>Motion prompt:</b> {e(s["motion_prompt"])}</p>'
                     f'<p><b>Sound prompt:</b> {e(s["sound_prompt"])}</p></details>')
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
