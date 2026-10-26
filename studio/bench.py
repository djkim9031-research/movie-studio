"""The identity bench: the person's best real photos as face-framed start frames, fixed motions, prompts and seeds
(config/bench/identity.yaml). A variant renders every cell; each clip gets provenance and is scored against the
person's profile without that cell's start photo."""

import html
import json
from pathlib import Path

import cv2
import yaml

from studio import identity, provenance, refs
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.render import render_i2v
from studio.video_graphs import WAN_LORAS_1022, ltx_single, render_graph, wan_i2v

SPEC = Path(__file__).resolve().parent.parent / "config" / "bench" / "identity.yaml"
SIZE = (1280, 704)
# The round rule's least gain worth having: a quarter of a genuine standard deviation (scores are z).
DELTA = 0.25
LTX_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"

# Variant -> the model sets that make its clips and the frames copied from the start image (-1 is the last frame).
VARIANTS = {
    "ltx-int8-2stage": {"models": ["ltx25-int8"], "pinned": [0]},
    "ltx-bf16-2stage": {"models": ["ltx25-int8", "ltx25-bf16"], "pinned": [0]},
    "ltx-bf16-1stage": {"models": ["ltx25-int8", "ltx25-bf16"], "pinned": [0]},
    "ltx-bf16-1stage-loop": {"models": ["ltx25-int8", "ltx25-bf16"], "pinned": [0, -1]},
    "wan-4step-1022": {"models": ["wan22-i2v", "wan22-lightx2v-1022"], "pinned": [0]},
    "wan-20step": {"models": ["wan22-i2v"], "pinned": [0]},
}


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
    """LTX two-stage runs the exported template (int8 as today, or the BF16 transformer); the other variants run
    graphs built from ComfyUI's single-stage LTX and Wan 2.2 templates."""
    if variant == "ltx-int8-2stage":
        return render_i2v(cfg, start, prompt, seconds, seed, SIZE, name, dest, cleanup=True)
    if variant == "ltx-bf16-2stage":
        return render_i2v(cfg, start, prompt, seconds, seed, SIZE, name, dest, cleanup=True, unet=LTX_BF16)
    prefix = f"bench/{name}"
    if variant in ("ltx-bf16-1stage", "ltx-bf16-1stage-loop"):
        last = "START" if variant.endswith("loop") else None
        graph = ltx_single(prompt=prompt, first="START", last=last, size=SIZE, seconds=seconds, seed=seed,
                           unet=LTX_BF16, prefix=prefix)
    elif variant == "wan-4step-1022":
        graph = wan_i2v(prompt=prompt, start="START", size=SIZE, seconds=seconds, seed=seed, steps=4, cfg=1.0,
                        split=2, loras=WAN_LORAS_1022, prefix=prefix)
    elif variant == "wan-20step":
        graph = wan_i2v(prompt=prompt, start="START", size=SIZE, seconds=seconds, seed=seed, steps=20, cfg=3.5,
                        split=10, loras=None, prefix=prefix)
    else:
        raise ValueError(f"unknown variant {variant}; known: {', '.join(VARIANTS)}")
    return render_graph(cfg, graph, {"START": start}, name=name, dest=dest)


def score_cell(cell: dict, clip: Path, start: Path, profiles: dict, tools, pinned: list[int]) -> dict:
    """The clip's scores against `profiles` (which leave out the cell's start photo) and its drift from the start
    frame, all in the combined z scale."""
    s = identity.score_clip(profiles, tools, clip, pinned=pinned).summary
    start_sim = identity.score_image(profiles, tools, start).sim
    cell.update(median=s.median, mid_min=s.mid_min, dips=s.dips, dip_seconds=s.dip_seconds,
                coverage=round(s.coverage, 3), yaw_range=round(s.yaw_range, 3), by_scorer=s.by_scorer,
                start_sim=start_sim, drift=None if s.median is None or start_sim is None else s.median - start_sim)
    return cell


def _source(profiles: dict, stem: str) -> str:
    return next(source for source in next(iter(profiles.values())).sources if Path(source).stem == stem)


def run(cfg: Config, *, alias: str, variant: str, tools, run_dir: Path, echo, count: int = 6, seeds=None,
        render=render) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; known: {', '.join(VARIANTS)}")
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    profiles = identity.load_profiles(cfg, alias)
    starts_dir, clips_dir = private_dir(run_dir / "starts"), private_dir(run_dir / "clips")
    results = {"variant": variant, "alias": alias, "profile": identity.digest(profiles),
               "scorers": sorted(profiles), "seconds": spec["seconds"], "cells": []}
    for photo, face in choose_starts(cfg, alias, tools, count):
        start = make_start(cfg, photo, face, starts_dir / f"{photo.stem}.png")
        cell_profiles = identity.without(profiles, photo.name)
        for motion, prompt in spec["motions"].items():
            for seed in seeds or spec["seeds"]:
                clip = clips_dir / f"{photo.stem}-{motion}-{variant}-s{seed}.mp4"
                made = render(cfg, variant, start, prompt, spec["seconds"], seed, f"bench_{photo.stem}_{motion}",
                              clip)
                pinned = VARIANTS[variant]["pinned"]
                provenance.write(cfg, clip, kind="clip", models=VARIANTS[variant]["models"], inputs=[start],
                                 params={"variant": variant, "motion": motion, "seed": seed, "prompt": prompt,
                                         "seconds": spec["seconds"], "pinned_frames": pinned})
                problems = provenance.problems(cfg, clip)
                if problems:
                    raise RuntimeError("; ".join(problems))
                cell = score_cell({"start": photo.stem, "motion": motion, "seed": seed, "clip": clip.name,
                                   "render_s": round(made.elapsed_s, 1)}, clip, start, cell_profiles, tools, pinned)
                results["cells"].append(cell)
                write_private(run_dir / "results.json", json.dumps(results, indent=1) + "\n")
                write_private(run_dir / "review.html", review_page(results))
                echo(line(cell))
    return results


def rescore(cfg: Config, *, alias: str, tools, run_dir: Path, echo) -> dict:
    """Scores a finished run again from its clips against the person's current profiles."""
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    profiles = identity.load_profiles(cfg, alias)
    for cell in results["cells"]:
        clip, start = run_dir / "clips" / cell["clip"], run_dir / "starts" / f"{cell['start']}.png"
        problems = provenance.problems(cfg, clip)
        if problems:
            raise RuntimeError("; ".join(problems))
        pinned = provenance.params(clip).get("pinned_frames", [0])
        score_cell(cell, clip, start, identity.without(profiles, _source(profiles, cell["start"])), tools, pinned)
        cell.pop("percentile", None)
        echo(line(cell))
    results.update(profile=identity.digest(profiles), scorers=sorted(profiles))
    write_private(run_dir / "results.json", json.dumps(results, indent=1) + "\n")
    write_private(run_dir / "review.html", review_page(results))
    return results


def line(cell: dict) -> str:
    """Scores are z against the person's genuine pairs (0 is the middle of their real photo pairs); drift is the
    change from the start frame; each scorer's median cosine follows."""
    if cell["median"] is None:
        return f"{cell['start']:<10} {cell['motion']:<6} no scorable face (coverage {cell['coverage']:.0%})"
    drift = f"drift {cell['drift']:+.2f} " if cell.get("drift") is not None else ""
    cosines = " ".join(f"{name} {value:.3f}" for name, value in sorted(cell.get("by_scorer", {}).items()))
    return (f"{cell['start']:<10} {cell['motion']:<6} score {cell['median']:+.2f} {drift}min {cell['mid_min']:+.2f} "
            f"dips {cell['dips']} coverage {cell['coverage']:.0%} yaw {cell['yaw_range']:.2f} {cosines} "
            f"{cell['render_s']:.0f} s")


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


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compare(runs: list[dict], delta: float) -> dict:
    """Each run against the first, cell by cell (same start photo, motion and seed), by the round rule: a mean gain
    of at least `delta`, a win on all but at most one start photo, and no motion losing more than delta / 2."""
    if len({run["profile"] for run in runs}) > 1:
        raise ValueError("these runs were scored against different profiles; run: uv run studio bench rescore")
    key = lambda c: (c["start"], c["motion"], c["seed"])  # noqa: E731
    base = {key(c): c for c in runs[0]["cells"] if c["median"] is not None}
    rows = []
    for i, run in enumerate(runs):
        cells = [c for c in run["cells"] if c["median"] is not None]
        motions = {}
        for motion in dict.fromkeys(c["motion"] for c in cells):
            mc = [c for c in cells if c["motion"] == motion]
            motions[motion] = {"median": _mean([c["median"] for c in mc]), "drift": _mean([c.get("drift") for c in mc]),
                               "min": _mean([c["mid_min"] for c in mc]), "dips": sum(c["dips"] for c in mc),
                               "coverage": _mean([c["coverage"] for c in mc]), "n": len(mc)}
        row = {"variant": run["variant"], "motions": motions, "render_s": _mean([c["render_s"] for c in cells])}
        if i == 0:
            row["verdict"] = "baseline"
            rows.append(row)
            continue
        pairs = [(c, base[key(c)]) for c in cells if key(c) in base]
        gains = [c["median"] - b["median"] for c, b in pairs]
        by_photo, by_motion = {}, {}
        for (c, b), g in zip(pairs, gains):
            by_photo.setdefault(c["start"], []).append(g)
            by_motion.setdefault(c["motion"], []).append(g)
        row.update(gain=_mean(gains), pairs=len(pairs), photos=len(by_photo),
                   photo_wins=sum(1 for g in by_photo.values() if _mean(g) > 0),
                   motion_gain={m: _mean(g) for m, g in by_motion.items()})
        if any(g < -delta / 2 for g in row["motion_gain"].values()):
            row["verdict"] = "a motion got worse"
        elif row["gain"] is not None and row["gain"] >= delta and row["photo_wins"] >= row["photos"] - 1:
            row["verdict"] = "passes screening"
        else:
            row["verdict"] = "no clear gain"
        rows.append(row)
    return {"delta": delta, "variants": rows}


def compare_page(runs: list[tuple[str, dict]], report: dict) -> str:
    """Every cell's clips side by side, run by run, with the numbers. Labelled, for analysis; it shows a real person,
    so it stays on this machine."""
    e = html.escape
    cells = {}
    for name, data in runs:
        for c in data["cells"]:
            cells.setdefault((c["start"], c["motion"], c["seed"]), {})[name] = c
    rows = []
    for (start, motion, seed), by_run in cells.items():
        figures = "".join(
            f'<figure><figcaption>{e(data["variant"])}</figcaption>'
            f'<video src="{e(name)}/clips/{e(by_run[name]["clip"])}" controls loop muted playsinline '
            f'preload="metadata"></video><p>{e(line(by_run[name]))}</p></figure>'
            for name, data in runs if name in by_run)
        rows.append(f'<section><h2>{e(start)} · {e(motion)} · seed {seed}</h2><div class="row">{figures}</div>'
                    f'</section>')
    verdicts = "".join(f"<li>{e(r['variant'])}: {e(r['verdict'])}</li>" for r in report["variants"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bench comparison</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }} h2 {{ font-size: 15px; margin: 24px 0 8px; }}
  .row {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap: 12px; }}
  figure {{ margin: 0; background: var(--card); border-radius: 10px; padding: 10px; }}
  figcaption {{ font-weight: 600; margin-bottom: 6px; }}
  video {{ display: block; width: 100%; height: auto; border-radius: 6px; background: #000; }}
  p, li {{ color: var(--muted); font: 12px/1.4 ui-monospace, monospace; }}
</style></head>
<body><h1>Bench comparison</h1><ul>{verdicts}</ul>
{"".join(rows)}
</body></html>
"""
