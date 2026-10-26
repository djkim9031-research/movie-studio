"""Blind comparisons: groups of options (stills, or frame strips of clips) shown as letters with no labels, names or
scores; the user's picks are typed into the CLI and checked against the scores afterwards."""

import html
import json
import random
import shutil
from pathlib import Path

from studio.fs import private_dir, write_private

LETTERS = "ABCDEFGH"


def make(groups: list[list[dict]], out: Path, seed: int) -> dict:
    """`groups`: lists of options {"id", "run", "images": [paths], "score"}; each group's options are shuffled."""
    out = private_dir(out)
    img = private_dir(out / "img")
    rng = random.Random(seed)
    key, sections = {"groups": []}, []
    for g, options in enumerate(groups, start=1):
        options = list(options)
        rng.shuffle(options)
        entry, rows = {"options": {}}, []
        for letter, option in zip(LETTERS, options):
            names = []
            for k, source in enumerate(option["images"], start=1):
                name = f"g{g:03d}{letter}{k}{Path(source).suffix}"
                shutil.copyfile(source, img / name)
                names.append(name)
            entry["options"][letter] = {"id": option["id"], "run": option["run"], "score": option.get("score")}
            frames = "".join(f'<img src="img/{n}" alt="">' for n in names)
            rows.append(f'<div class="option"><b>{letter}</b><div class="strip">{frames}</div></div>')
        key["groups"].append(entry)
        sections.append(f'<section><h2>Group {g}</h2>{"".join(rows)}</section>')
    write_private(out / "key.json", json.dumps(key, indent=1) + "\n")
    write_private(out / "page.html", _page("".join(sections), len(groups)))
    return key


def pick(out: Path, picks: list[str], delta: float) -> dict:
    """Picks like "1A", "2B" or "3=" (no preference). Wins per run, and how often the user's choice is also the
    higher score where the scores differ by more than `delta`."""
    key = json.loads((Path(out) / "key.json").read_text(encoding="utf-8"))
    wins, agree, disagree, ties = {}, 0, 0, 0
    for option in (o for g in key["groups"] for o in g["options"].values()):
        wins.setdefault(option["run"], 0)
    for item in picks:
        number, choice = item[:-1], item[-1].upper()
        if not number.isdigit() or not 1 <= int(number) <= len(key["groups"]):
            raise ValueError(f"{item}: no group {number}")
        options = key["groups"][int(number) - 1]["options"]
        if choice == "=":
            ties += 1
            continue
        if choice not in options:
            raise ValueError(f"{item}: group {number} has options {', '.join(options)}")
        chosen = options[choice]
        wins[chosen["run"]] += 1
        others = [o["score"] for letter, o in options.items() if letter != choice and o["score"] is not None]
        if chosen["score"] is not None and others and abs(chosen["score"] - max(others)) > delta:
            if chosen["score"] > max(others):
                agree += 1
            else:
                disagree += 1
    report = {"picks": picks, "wins": wins, "agree": agree, "disagree": disagree, "ties": ties}
    write_private(Path(out) / "picks.json", json.dumps(report, indent=1) + "\n")
    return report


def _page(sections: str, count: int) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind comparison</title>
<style>
  :root {{ color-scheme: dark; --bg: #0b0d12; --card: #161a22; --text: #e8eaf0; --muted: #9aa3b5; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  p {{ color: var(--muted); max-width: 70ch; }}
  section {{ background: var(--card); border-radius: 10px; padding: 12px; margin: 16px 0; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .option {{ display: flex; gap: 10px; align-items: center; margin: 8px 0; }}
  .option b {{ font-size: 20px; width: 1.5em; }}
  .strip {{ display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 6px; flex: 1; }}
  img {{ width: 100%; height: auto; border-radius: 6px; }}
</style></head>
<body><h1>Which looks more like the person?</h1>
<p>{count} groups. For each, pick the option that looks most like the real person (ignore everything else), or "="
when you can't tell. Then run, for example: uv run studio bench pick NAME 1A 2B 3=</p>
{sections}
</body></html>
"""
