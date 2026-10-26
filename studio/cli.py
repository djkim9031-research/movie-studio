from datetime import datetime
from pathlib import Path

import typer

from studio.claude.ledger import Ledger, summarize
from studio.config import Config, load_config
from studio.doctor import exit_code, run_doctor
from studio.media import ffmpeg
from studio.media.smoke import TARGET_LUFS, smoke_mux

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Short films from a written story.")
_state: dict[str, Config] = {}


@app.callback()
def main(config: Path = typer.Option(None, "--config", help="Config file; default config/studio.yaml.")):
    _state["cfg"] = load_config(config)


@app.command()
def doctor(claude: bool = typer.Option(False, "--claude",
                                       help="Also check the Claude API key and prompt cache (a few cents).")):
    """Check the GPU, disks, FFmpeg and, with --claude, the Claude API."""
    checks = run_doctor(_state["cfg"], with_claude=claude)
    for c in checks:
        typer.echo(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<22} {c.detail}")
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
