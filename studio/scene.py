"""A scene from a prompt and pictures: references, Claude's two prompts and FLUX 3 Klein stills; then LTX-2.5 clips of
the stills the user picks. Pictures and frames stay on this machine; Claude gets the prompt text only."""

import contextlib
import html
import json
import re
import secrets
import tempfile
from pathlib import Path

from studio import bench, refs, scene_prompts
from studio.bench import render as render_clip
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.render import LANDSCAPE as CLIP_LANDSCAPE
from studio.render import PORTRAIT as CLIP_PORTRAIT
from studio.render import prepare_still
from studio.stills import LANDSCAPE, PORTRAIT, render_still


class SceneError(RuntimeError):
    pass


def clean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if not cleaned:
        raise SceneError(f"not a usable scene name: {name!r}")
    return cleaned


def scene_dir(cfg: Config, name: str) -> Path:
    return cfg.projects_dir / "scenes" / clean_name(name)


def _save(folder: Path, data: dict) -> None:
    write_private(folder / "scene.json", json.dumps(data, indent=2) + "\n")
    write_private(folder / "review.html", review_page(data))


def load(cfg: Config, name: str) -> tuple[Path, dict]:
    folder = scene_dir(cfg, name)
    if not (folder / "scene.json").is_file():
        raise SceneError(f"no scene {clean_name(name)} in {folder.parent}")
    return folder, json.loads((folder / "scene.json").read_text(encoding="utf-8"))


def _models(cfg: Config) -> dict:
    return {key: f"{name} {cfg.models[name].revision[:12]}" for key, name in
            (("stills", "flux3-klein"), ("clips", "ltx25-int8")) if name in cfg.models}


def references(pictures: list[Path], folder: Path, tools, echo) -> refs.Report:
    """The usable pictures as references in folder/refs, each one reported."""
    try:
        pictures, unreadable = refs.gather(pictures)
    except ValueError as error:
        raise SceneError(str(error)) from None
    for source, reason in unreadable:
        echo(f"refs  skipped {source}: {reason}")
    if not pictures:
        raise SceneError("no pictures found")
    private_dir(folder)
    report = refs.prepare(pictures, folder / "refs", tools)
    for ref in report.used:
        if ref.kind == "person":
            detail = f"face {ref.face_px} px, sharpness {ref.sharpness:.0f}"
        else:
            detail = "other people, faces blurred" if ref.faces_blurred else "no person"
        echo(f"refs  {ref.file.name} from {ref.source} ({detail})")
    for source, reason in report.skipped:
        echo(f"refs  skipped {source}: {reason}")
    if not report.used:
        raise SceneError("none of the pictures can be used")
    return report


def make_stills(cfg: Config, *, name: str, prompt: str, pictures: list[Path], count: int, portrait: bool, claude,
                tools, echo, gpu=contextlib.nullcontext) -> dict:
    """References and prompts first (no GPU), then `count` stills inside `gpu()`."""
    name = clean_name(name)
    folder = scene_dir(cfg, name)
    # A folder without scene.json is a run that stopped before its prompts were written; it may run again.
    if (folder / "scene.json").exists():
        raise SceneError(f"scene {name} exists in {folder}; pick another name")
    report = references(pictures, folder, tools, echo)
    orientation = "portrait" if portrait else "landscape"
    size = PORTRAIT if portrait else LANDSCAPE
    prompts = scene_prompts.write(claude, film=f"scene-{name}", user_prompt=prompt, report=report,
                                  orientation=orientation)
    data = {"name": name, "prompt": prompt, "orientation": orientation, "size": list(size),
            "references": [{"kind": r.kind, "file": f"refs/{r.file.name}", "square": f"refs/{r.square.name}",
                            "source": r.source, "face_px": r.face_px, "sharpness": r.sharpness} for r in report.used],
            "skipped": [{"source": s, "reason": why} for s, why in report.skipped],
            "still_request": prompts.still_request, "motion_prompt": prompts.motion_prompt, "caption": "",
            "models": _models(cfg), "stills": [], "clips": []}
    _save(folder, data)
    echo(f"prompts  written by Claude, saved in {folder / 'scene.json'}")
    stills_dir = private_dir(folder / "stills")
    with gpu():
        for n in range(1, count + 1):
            seed = secrets.randbelow(2**31)
            dest = stills_dir / f"still{n}.png"
            still = render_still(cfg, report.used, prompts.still_request, seed=seed, size=size,
                                 name=f"{name}_still{n}", dest=dest)
            write_private(stills_dir / f"still{n}.graph.json", json.dumps(still.graph, indent=1) + "\n")
            data["caption"] = still.caption
            data["stills"].append({"n": n, "seed": seed, "file": f"stills/still{n}.png",
                                   "render_s": round(still.elapsed_s, 1)})
            _save(folder, data)
            echo(f"still {n}/{count}  seed {seed}, {still.elapsed_s:.1f} s -> {dest}")
    return data


def make_clips(cfg: Config, *, name: str, numbers: list[int], seconds: int, seed: int | None, echo,
               gpu=contextlib.nullcontext, video: str = "ltx-int8-2stage") -> dict:
    """One clip per chosen still with the scene's motion prompt, rendered by a bench video variant; one seed for all,
    random when None."""
    if video not in bench.VARIANTS:
        raise SceneError(f"unknown video variant {video}; known: {', '.join(bench.VARIANTS)}")
    folder, data = load(cfg, name)
    stills = {s["n"]: s for s in data["stills"]}
    for n in numbers:
        if n not in stills:
            raise SceneError(f"no still {n} in scene {data['name']}; it has {sorted(stills)}")
    size = CLIP_PORTRAIT if data["orientation"] == "portrait" else CLIP_LANDSCAPE
    clips_dir = private_dir(folder / "clips")
    with gpu():
        for n in numbers:
            clip_seed = seed if seed is not None else secrets.randbelow(2**31)
            dest = clips_dir / f"still{n}-seed{clip_seed}-{video}-{seconds}s.mp4"
            with tempfile.TemporaryDirectory() as tmp:
                start = prepare_still(cfg, folder / stills[n]["file"], Path(tmp) / "start.png", size)
                result = render_clip(cfg, video, start, data["motion_prompt"], seconds, clip_seed,
                                     f"{data['name']}_still{n}", dest, size=size)
            data["clips"].append({"still": n, "seed": clip_seed, "seconds": seconds, "video": video,
                                  "file": f"clips/{dest.name}", "render_s": round(result.elapsed_s, 1)})
            _save(folder, data)
            echo(f"clip of still {n} ({video})  seed {clip_seed}, {result.elapsed_s:.1f} s -> {dest}")
    return data


def compare_page(cfg: Config, names: list[str]) -> str:
    """Scenes side by side, stills and clips, for a before and after look. It shows real people: never published."""
    e = html.escape
    columns = []
    for name in names:
        folder, data = load(cfg, name)
        stills = "".join(f'<img src="{e(folder.name)}/{e(s["file"])}" alt="still {s["n"]}">' for s in data["stills"])
        clips = "".join(f'<video src="{e(folder.name)}/{e(c["file"])}" controls loop playsinline preload="metadata">'
                        f'</video><p class="meta">still {c["still"]}, {e(c.get("video", "ltx-int8-2stage"))}, '
                        f'{c["seconds"]} s</p>' for c in data["clips"])
        columns.append(f'<section><h2>{e(data["name"])}</h2><p>{e(data["prompt"])}</p>{stills}{clips}</section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scene comparison</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr)); gap: 16px; }}
  section {{ background: var(--card); border-radius: 10px; padding: 12px; }}
  img, video {{ display: block; width: 100%; height: auto; border-radius: 6px; margin-top: 8px; background: #000; }}
  p {{ color: var(--muted); }} .meta {{ font-size: 12px; margin: 4px 0 0; }}
</style></head>
<body><h1>Before and after</h1><main>{"".join(columns)}</main></body></html>
"""


def review_page(data: dict) -> str:
    """A local page: each still with its clips, and every prompt used. Never published: it shows real people."""
    e = html.escape
    cards = []
    for s in data["stills"]:
        clips = "".join(f'<video src="{e(c["file"])}" controls loop playsinline preload="metadata"></video>'
                        f'<p class="meta">{e(c.get("video", "ltx-int8-2stage"))}, seed {c["seed"]}, {c["seconds"]} s</p>'
                        for c in data["clips"] if c["still"] == s["n"])
        cards.append(f'<figure><figcaption>Still {s["n"]}</figcaption><img src="{e(s["file"])}" alt="Still {s["n"]}">'
                     f'{clips}<p class="meta">seed {s["seed"]}</p></figure>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(data["name"])} scene</title>
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
<h1>{e(data["name"])}</h1>
<p>{e(data["prompt"])}</p>
<details><summary>Prompts and caption</summary>
<p><b>Still request (Claude):</b> {e(data["still_request"])}</p>
<p><b>Caption (local prompt upsampler):</b> {e(data["caption"])}</p>
<p><b>Motion prompt (Claude):</b> {e(data["motion_prompt"])}</p>
</details>
<div class="grid">
{chr(10).join(cards)}
</div>
</body>
</html>
"""
