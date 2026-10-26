"""The keyframe bake-off: the same shot prompts, seeds and reference photos for each engine. Stills from the Apache
engines are face-scored; FLUX 3 Klein's are judged by eye only, since its license forbids biometric processing of its
outputs."""

import html
import json
from pathlib import Path

import yaml

from studio import identity, keyframes, provenance, refs, stills
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.video_graphs import render_graph

SPEC = Path(__file__).resolve().parent.parent / "config" / "bench" / "keyframes.yaml"

ENGINES = {
    "flux3": {"models": ["flux3-klein"], "refs": 6, "size": stills.LANDSCAPE},
    "qwen-edit": {"models": ["qwen-edit-2511", "qwen-image-encoders"], "refs": 3, "size": (1664, 928)},
    "klein4b": {"models": ["flux2-klein-4b"], "refs": 4, "size": stills.LANDSCAPE},
}


def request(prompt: str, count: int) -> str:
    return (f"Create a photograph of the person shown in images 1-{count} (all {count} show the same person) {prompt} "
            f"Keep the person's exact face, identity, skin tone and hair.")


def render(cfg: Config, engine: str, references: list, text: str, seed: int, name: str, dest: Path):
    size = ENGINES[engine]["size"]
    if engine == "flux3":
        return stills.render_still(cfg, references, text, seed=seed, size=size, name=name, dest=dest)
    images = {f"R{i}": ref.file for i, ref in enumerate(references, start=1)}
    if engine == "qwen-edit":
        graph = keyframes.qwen_edit(prompt=text, refs=list(images), size=size, seed=seed, prefix=f"keyframes/{name}")
    elif engine == "klein4b":
        graph = keyframes.flux2_klein(prompt=text, refs=list(images), size=size, seed=seed, prefix=f"keyframes/{name}")
    else:
        raise ValueError(f"unknown engine {engine}; known: {', '.join(ENGINES)}")
    return render_graph(cfg, graph, images, name=name, dest=dest)


def run(cfg: Config, *, alias: str, engine: str, tools, run_dir: Path, echo, render=render,
        limit: int | None = None) -> dict:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine}; known: {', '.join(ENGINES)}")
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    profiles = identity.load_profiles(cfg, alias)
    pictures, _ = refs.gather([cfg.data_root / "cast" / alias / "photos"], limit=identity.MAX_PHOTOS)
    report = refs.prepare(pictures, run_dir / "refs", tools)
    by_name = {p.name: p for p in pictures}
    for ref in report.used:
        provenance.write(cfg, ref.file, kind="reference", models=[], inputs=[by_name[ref.source]], params={})
    chosen = report.best(ENGINES[engine]["refs"])
    results = {"engine": engine, "alias": alias, "profile": identity.digest(profiles),
               "refs": [r.source for r in chosen], "cells": []}
    stills_dir = private_dir(run_dir / "stills")
    for shot, prompt in enumerate(spec["prompts"][:limit], start=1):
        for seed in spec["seeds"]:
            dest = stills_dir / f"shot{shot:02d}-s{seed}.png"
            text = request(prompt, len(chosen))
            made = render(cfg, engine, chosen, text, seed, f"kf_{engine}_{shot:02d}", dest)
            provenance.write(cfg, dest, kind="keyframe", models=ENGINES[engine]["models"],
                             inputs=[r.file for r in chosen], params={"engine": engine, "seed": seed, "prompt": text})
            cell = {"shot": shot, "seed": seed, "file": dest.name, "render_s": round(made.elapsed_s, 1)}
            problems = provenance.problems(cfg, dest)
            if problems:
                cell.update(score=None, unscored="; ".join(problems))
            else:
                frame = identity.score_image(profiles, tools, dest)
                cell.update(score=frame.sim, sims=frame.sims, face_px=frame.px, yaw=frame.yaw)
            results["cells"].append(cell)
            write_private(run_dir / "results.json", json.dumps(results, indent=1) + "\n")
            write_private(run_dir / "review.html", review_page(results))
            echo(line(cell))
    return results


def line(cell: dict) -> str:
    head = f"shot {cell['shot']:02d} seed {cell['seed']}"
    if cell["score"] is None:
        reason = "no face found" if "unscored" not in cell else "not scored: license"
        return f"{head}  {reason}  {cell['render_s']:.0f} s"
    cosines = " ".join(f"{k} {v:.3f}" for k, v in sorted(cell["sims"].items()))
    return f"{head}  score {cell['score']:+.2f}  {cosines}  face {cell['face_px']:.0f} px  {cell['render_s']:.0f} s"


def review_page(results: dict) -> str:
    """Every still with its number and score. It shows a real person, so it stays on this machine."""
    e = html.escape
    cards = "".join(f'<figure><img src="stills/{e(c["file"])}" alt="shot {c["shot"]}"><p>{e(line(c))}</p></figure>'
                    for c in results["cells"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keyframe bake-off</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap: 14px; }}
  figure {{ margin: 0; background: var(--card); border-radius: 10px; padding: 10px; }}
  img {{ display: block; width: 100%; height: auto; border-radius: 6px; }}
  p {{ color: var(--muted); font: 12px/1.4 ui-monospace, monospace; margin: 6px 0 0; }}
</style></head>
<body><h1>Keyframes: {e(results["engine"])}</h1><main>{cards}</main></body></html>
"""
