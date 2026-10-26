from datetime import datetime
from pathlib import Path

import typer

from studio import models
from studio.claude.ledger import Ledger, summarize
from studio.config import Config, load_config
from studio.doctor import exit_code, run_doctor
from studio.media import ffmpeg
from studio.media.smoke import TARGET_LUFS, smoke_mux

# Tracebacks must never print locals: the Claude client holds the API key.
app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False,
                  help="Short films from a written story.")
models_app = typer.Typer(no_args_is_help=True, help="Pinned model weights.")
app.add_typer(models_app, name="models")
_state: dict[str, Config] = {}


@app.callback()
def main(config: Path = typer.Option(None, "--config", help="Config file; default config/studio.yaml.")):
    _state["cfg"] = load_config(config)


@app.command()
def doctor(claude: bool = typer.Option(False, "--claude",
                                       help="Also check the Claude API key and prompt cache (a few cents).")):
    """Check the GPU, disks, FFmpeg and, with --claude, the Claude API."""
    checks = []
    for c in run_doctor(_state["cfg"], with_claude=claude):
        typer.echo(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<22} {c.detail}")
        checks.append(c)
    raise typer.Exit(exit_code(checks))


@app.command()
def cost(month: str = typer.Option(None, help="YYYY-MM; default this month.")):
    """Claude API spend for a month, by film and by route."""
    cfg = _state["cfg"]
    month = month or f"{datetime.now():%Y-%m}"
    s = summarize(Ledger(cfg.ledger_dir, cfg.claude).entries(month))
    typer.echo(f"{month}: ${s['total']:.2f} of ${cfg.claude.month_cap_usd:.2f} cap, {s['calls']} calls")
    for film, usd in sorted(s["by_film"].items()):
        typer.echo(f"  film  {film:<24} ${usd:.2f}")
    for route, usd in sorted(s["by_route"].items()):
        typer.echo(f"  route {route:<24} ${usd:.2f}")
    if s["unpriced_calls"]:
        typer.echo(f"  {s['unpriced_calls']} calls used a model missing from the price table")


@app.command("smoke-mux")
def smoke_mux_command(out: Path = typer.Option(None, help="Output file; default projects/_smoke/smoke.mp4.")):
    """Write a 10 s test pattern with a tone at the loudness target and measure it."""
    cfg = _state["cfg"]
    out = out or cfg.projects_dir / "_smoke" / "smoke.mp4"
    integrated, peak = smoke_mux(cfg, out)
    ok = abs(integrated - TARGET_LUFS) <= 0.5
    typer.echo(f"{'PASS' if ok else 'FAIL'}  {out}: {integrated:.1f} LUFS (target {TARGET_LUFS:.0f}), "
               f"true peak {peak} dBFS")
    raise typer.Exit(0 if ok else 1)


@app.command("fetch-ffmpeg")
def fetch_ffmpeg():
    """Download the pinned FFmpeg build, check its sha256 and install ffmpeg and ffprobe."""
    for path in ffmpeg.install(_state["cfg"]):
        typer.echo(path)


def _report_models(name: str, problems: list[str]) -> None:
    if problems:
        for problem in problems:
            typer.echo(f"FAIL  {problem}")
        raise typer.Exit(1)
    typer.echo(f"PASS  {name}: {len(_state['cfg'].models[name].files)} files match their pinned size and sha256")


@models_app.command("verify")
def models_verify(name: str):
    """Check a pinned model set's files by size and sha256."""
    _report_models(name, models.verify(_state["cfg"], name))


@models_app.command("fetch")
def models_fetch(name: str):
    """Download a pinned model set with the Hugging Face token in secrets/hf.env, then verify it."""
    _report_models(name, models.fetch(_state["cfg"], name))
