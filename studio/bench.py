"""The identity bench: the person's best real photos as face-framed start frames, fixed motions, prompts and seeds
(config/bench/identity.yaml). A variant renders every cell; each clip gets provenance and is scored against the
person's profile without that cell's start photo."""

import html
import json
import os
from pathlib import Path

import cv2
import yaml

from studio import identity, provenance, refs
from studio.config import Config
from studio.fs import private_dir
from studio.render import render_i2v

SPEC = Path(__file__).resolve().parent.parent / "config" / "bench" / "identity.yaml"
SIZE = (1280, 704)

# Variant name -> the model sets that make its clips; how each renders is in render().
VARIANTS = {"ltx-int8-2stage": ["ltx25-int8"]}


def choose_starts(cfg: Config, alias: str, tools, count: int = 6) -> list[tuple[Path, refs.Face]]:
    """The person's best faces among their real photos: sharp, large, frontal, no near-copies."""
    pictures, _ = refs.gather([cfg.data_root / "cast" / alias / "photos"], limit=identity.MAX_PHOTOS)
    faces = {p.name: tools.faces(refs.load(p)) for p in pictures}
    by_name = {p.name: p for p in pictures}
    return [(by_name[name], face) for name, face in refs.best_faces(refs.find_subject(faces), limit=count)]


def make_start(cfg: Config, photo: Path, face: refs.Face, dest: Path) -> Path:
    bgr = refs.load(photo)
    x0, y0, x1, y1 = refs.face_frame(bgr.shape, face)
    method = cv2.INTER_AREA if x1 - x0 >= SIZE[0] else cv2.INTER_LANCZOS4
    refs.save_png(cv2.resize(bgr[y0:y1, x0:x1], SIZE, interpolation=method), dest)
    provenance.write(cfg, dest, kind="start", models=[], inputs=[photo],
                     params={"window": [x0, y0, x1, y1], "size": list(SIZE)})
    return dest


def render(cfg: Config, variant: str, start: Path, prompt: str, seconds: int, seed: int, name: str, dest: Path):
    if variant == "ltx-int8-2stage":
        return render_i2v(cfg, start, prompt, seconds, seed, SIZE, name, dest, cleanup=True)
    raise ValueError(f"unknown variant {variant}; known: {', '.join(VARIANTS)}")


def _write(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def run(cfg: Config, *, alias: str, variant: str, tools, run_dir: Path, echo, count: int = 6, seeds=None,
        render=render) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; known: {', '.join(VARIANTS)}")
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    profile = identity.load_profile(cfg, alias)
    starts_dir, clips_dir = private_dir(run_dir / "starts"), private_dir(run_dir / "clips")
    results = {"variant": variant, "alias": alias, "profile": profile.digest, "seconds": spec["seconds"],
               "cells": []}
    for photo, face in choose_starts(cfg, alias, tools, count):
        start = make_start(cfg, photo, face, starts_dir / f"{photo.stem}.png")
        cell_profile = profile.without(photo.name)
        for motion, prompt in spec["motions"].items():
            for seed in seeds or spec["seeds"]:
                clip = clips_dir / f"{photo.stem}-{motion}-{variant}-s{seed}.mp4"
                made = render(cfg, variant, start, prompt, spec["seconds"], seed, f"bench_{photo.stem}_{motion}",
                              clip)
                provenance.write(cfg, clip, kind="clip", models=VARIANTS[variant], inputs=[start],
                                 params={"variant": variant, "motion": motion, "seed": seed, "prompt": prompt,
                                         "seconds": spec["seconds"], "pinned_frames": [0]})
                problems = provenance.problems(cfg, clip)
                if problems:
                    raise RuntimeError("; ".join(problems))
                s = identity.score_clip(cell_profile, tools, clip, pinned=[0]).summary
                cell = {"start": photo.stem, "motion": motion, "seed": seed, "clip": clip.name,
                        "median": s.median, "mid_min": s.mid_min, "dips": s.dips, "dip_seconds": s.dip_seconds,
                        "coverage": round(s.coverage, 3), "yaw_range": round(s.yaw_range, 3),
                        "percentile": None if s.median is None else identity.percentile(cell_profile, s.median),
                        "render_s": round(made.elapsed_s, 1)}
                results["cells"].append(cell)
                _write(run_dir / "results.json", json.dumps(results, indent=1) + "\n")
                _write(run_dir / "review.html", review_page(results))
                echo(line(cell))
    return results


def line(cell: dict) -> str:
    if cell["median"] is None:
        return f"{cell['start']:<10} {cell['motion']:<6} no scorable face (coverage {cell['coverage']:.0%})"
    return (f"{cell['start']:<10} {cell['motion']:<6} median {cell['median']:.3f} (p{cell['percentile']:.0f}) "
            f"min {cell['mid_min']:.3f} dips {cell['dips']} coverage {cell['coverage']:.0%} "
            f"yaw {cell['yaw_range']:.2f} {cell['render_s']:.0f} s")


def review_page(results: dict) -> str:
    """A local page of every cell with its numbers. It shows a real person: never published."""
    e = html.escape
    cards = "\n".join(
        f'<figure><figcaption>{e(c["start"])} · {e(c["motion"])}</figcaption>'
        f'<video src="clips/{e(c["clip"])}" controls loop muted playsinline preload="metadata"></video>'
        f'<p>{e(line(c))}</p></figure>' for c in results["cells"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Identity bench</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap: 16px; }}
  figure {{ margin: 0; background: var(--card); border-radius: 10px; padding: 10px; }}
  figcaption {{ font-weight: 600; margin-bottom: 6px; }}
  video {{ display: block; width: 100%; height: auto; border-radius: 6px; background: #000; }}
  p {{ color: var(--muted); font: 12px/1.4 ui-monospace, monospace; margin: 6px 0 0; }}
</style></head>
<body><h1>Identity bench: {e(results["variant"])}</h1>
<main>
{cards}
</main></body></html>
"""
