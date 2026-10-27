"""A short film from a prompt and pictures: Claude's shot plan (text only), one FLUX 3 Klein still and one Wan 2.2 clip
per shot, joined into one MP4; sound is added to a finished film by a second step. Pictures and frames stay on this
machine."""

import contextlib
import html
import json
import secrets
import tempfile
from pathlib import Path

from studio import bench, film_prompts, provenance
from studio.bench import render as render_clip
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.media import assemble
from studio.render import LANDSCAPE as CLIP_LANDSCAPE
from studio.render import PORTRAIT as CLIP_PORTRAIT
from studio.render import prepare_still
from studio.scene import clean_name, references
from studio.stills import LANDSCAPE, PORTRAIT, render_still

VIDEO = "wan-4step-1022"
FPS = 16  # Wan 2.2's frame rate


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


def make(cfg: Config, *, name: str, prompt: str, pictures: list[Path], shots: int, seconds: int, portrait: bool,
         claude, tools, echo, gpu=contextlib.nullcontext) -> dict:
    """References and the shot plan first (no GPU), then inside `gpu()` every still and then every clip (one model
    swap), then the silent film."""
    name = clean_name(name)
    folder = film_dir(cfg, name)
    if (folder / "film.json").exists():
        raise FilmError(f"film {name} exists in {folder}; pick another name")
    report = references(pictures, folder, tools, echo)
    orientation = "portrait" if portrait else "landscape"
    plan = film_prompts.write(claude, film=f"film-{name}", user_prompt=prompt, report=report,
                              orientation=orientation, shots=shots, seconds=seconds)
    data = {"name": name, "prompt": prompt, "title": plan.title, "orientation": orientation, "seconds": seconds,
            "fps": FPS, "video": VIDEO,
            "references": [{"kind": r.kind, "file": f"refs/{r.file.name}", "source": r.source} for r in report.used],
            "skipped": [{"source": s, "reason": why} for s, why in report.skipped],
            "models": {"stills": _models(cfg, ["flux3-klein"]),
                       "clips": _models(cfg, bench.VARIANTS[VIDEO]["models"])},
            "shots": [{"n": n, **shot.model_dump(), "still": None, "still_seed": None, "caption": "", "clip": None,
                       "clip_seed": None} for n, shot in enumerate(plan.shots, start=1)],
            "film": None, "sound": None}
    _save(folder, data)
    echo(f"plan  \"{plan.title}\", {shots} shots by Claude, saved in {folder / 'film.json'}")
    stills_dir, clips_dir = private_dir(folder / "stills"), private_dir(folder / "clips")
    size = PORTRAIT if portrait else LANDSCAPE
    clip_size = CLIP_PORTRAIT if portrait else CLIP_LANDSCAPE
    with gpu():
        for shot in data["shots"]:
            n, seed = shot["n"], secrets.randbelow(2**31)
            dest = stills_dir / f"shot{n}.png"
            still = render_still(cfg, report.used, shot["still_request"], seed=seed, size=size,
                                 name=f"{name}_shot{n}", dest=dest)
            write_private(stills_dir / f"shot{n}.graph.json", json.dumps(still.graph, indent=1) + "\n")
            provenance.write(cfg, dest, kind="still", models=["flux3-klein"], inputs=[r.file for r in report.used],
                             params={"seed": seed, "size": list(size)})
            shot.update(still=f"stills/{dest.name}", still_seed=seed, caption=still.caption,
                        still_s=round(still.elapsed_s, 1))
            _save(folder, data)
            echo(f"still {n}/{shots}  seed {seed}, {still.elapsed_s:.1f} s -> {dest}")
        for shot in data["shots"]:
            n, seed = shot["n"], secrets.randbelow(2**31)
            dest = clips_dir / f"shot{n}.mp4"
            with tempfile.TemporaryDirectory() as tmp:
                start = prepare_still(cfg, folder / shot["still"], Path(tmp) / "start.png", clip_size)
                result = render_clip(cfg, VIDEO, start, shot["motion_prompt"], seconds, seed, f"{name}_shot{n}",
                                     dest, size=clip_size)
            provenance.write(cfg, dest, kind="clip", models=bench.VARIANTS[VIDEO]["models"],
                             inputs=[folder / shot["still"]], params={"seed": seed, "seconds": seconds})
            shot.update(clip=f"clips/{dest.name}", clip_seed=seed, clip_s=round(result.elapsed_s, 1))
            _save(folder, data)
            echo(f"clip  {n}/{shots}  seed {seed}, {result.elapsed_s:.1f} s -> {dest}")
    clips = [folder / shot["clip"] for shot in data["shots"]]
    assemble.silent_film(cfg, clips, folder / "film.mp4", seconds=seconds, fps=FPS)
    provenance.write(cfg, folder / "film.mp4", kind="film", models=[], inputs=clips, params={"seconds": seconds})
    data["film"] = "film.mp4"
    _save(folder, data)
    return data


def add_sound(cfg: Config, *, name: str, worker, echo, seed: int | None = None) -> dict:
    """One track per shot from its clip and sound prompt, then the film again with the tracks mixed in."""
    folder, data = load(cfg, name)
    if not data.get("film"):
        raise FilmError(f"film {data['name']} has no finished picture yet")
    problems = worker.problems(cfg)
    if problems:
        raise FilmError(f"the sound worker is not ready: {'; '.join(problems)}")
    base = seed if seed is not None else secrets.randbelow(2**31 - 100)
    sound_dir = private_dir(folder / "sound")
    for shot in data["shots"]:
        n = shot["n"]
        dest = sound_dir / f"shot{n}.wav"
        made = worker.render(cfg, clip=folder / shot["clip"], prompt=shot["sound_prompt"], seconds=data["seconds"],
                             seed=base + n, dest=dest)
        provenance.write(cfg, dest, kind="sound", models=worker.models, inputs=[folder / shot["clip"]],
                         params={"seed": base + n, "seconds": data["seconds"]})
        shot.update(sound=f"sound/{dest.name}", sound_seed=base + n, sound_s=round(made.elapsed_s, 1))
        _save(folder, data)
        echo(f"sound {n}/{len(data['shots'])}  seed {base + n}, {made.elapsed_s:.1f} s -> {dest}")
    out = folder / "film-sound.mp4"
    assemble.with_sound(cfg, folder / data["film"], [folder / s["sound"] for s in data["shots"]], out,
                        seconds=data["seconds"])
    data["sound"] = {"film": out.name, "lufs": assemble.TARGET_LUFS, "true_peak": assemble.TRUE_PEAK}
    _save(folder, data)
    return data


def review_page(data: dict) -> str:
    """A local page: the film, then each shot with its still, clip and prompts. Never published: it shows real
    people."""
    e = html.escape
    films = "".join(f'<figure><figcaption>{label}</figcaption><video src="{e(file)}" controls playsinline '
                    f'preload="metadata"></video></figure>'
                    for label, file in (("Film", data.get("film")),
                                        ("Film with sound", (data.get("sound") or {}).get("film"))) if file)
    cards = []
    for s in data["shots"]:
        media = (f'<img src="{e(s["still"])}" alt="Shot {s["n"]} still">' if s.get("still") else "") + \
                (f'<video src="{e(s["clip"])}" controls loop muted playsinline preload="metadata"></video>'
                 if s.get("clip") else "")
        cards.append(f'<figure><figcaption>Shot {s["n"]}</figcaption>{media}'
                     f'<details><summary>Prompts</summary>'
                     f'<p><b>Still request:</b> {e(s["still_request"])}</p>'
                     f'<p><b>Motion prompt:</b> {e(s["motion_prompt"])}</p>'
                     f'<p><b>Sound prompt:</b> {e(s["sound_prompt"])}</p></details>'
                     f'<p class="meta">still seed {s.get("still_seed")}, clip seed {s.get("clip_seed")}</p></figure>')
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
  p, details {{ color: var(--muted); max-width: 80ch; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 20px;
          margin-top: 20px; }}
  figure {{ margin: 0; background: var(--card); border-radius: 10px; padding: 12px; }}
  figcaption {{ font-weight: 600; margin-bottom: 8px; color: var(--text); }}
  img, video {{ display: block; width: 100%; height: auto; border-radius: 6px; background: #000; margin-top: 8px; }}
  .meta {{ font-size: 13px; margin: 6px 0 0; }}
</style>
</head>
<body>
<h1>{e(data["title"])}</h1>
<p>{e(data["prompt"])}</p>
<div class="grid">
{films}
{chr(10).join(cards)}
</div>
</body>
</html>
"""
