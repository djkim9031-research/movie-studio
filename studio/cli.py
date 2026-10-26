import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import typer

from studio import models
from studio.fs import private_dir
from studio.gpu import runtime
from studio.gpu.comfy import output_files
from studio.gpu.preflight import run_preflight
from studio.gpu.vram import VramSampler
from studio.workflow import fill, strip_prompt_enhancer
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
gpu_app = typer.Typer(no_args_is_help=True, help="The ComfyUI container on this GPU.")
app.add_typer(gpu_app, name="gpu")
WORKFLOWS = Path(__file__).resolve().parent.parent / "config" / "workflows"
RENDER_TEST_PROMPT = ("A slow push-in toward a misty pond at morning. Fog drifts low across the water, willow "
                      "leaves and tall meadow grass sway in a light breeze, soft sunlight. Quiet outdoor ambience: "
                      "light wind and distant birds.")
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


@gpu_app.command("build")
def gpu_build():
    """Build the ComfyUI image from docker/comfyui with the pins in config/studio.yaml."""
    runtime.build(_state["cfg"])


@gpu_app.command("up")
def gpu_up():
    """Check the GPU is free, start the container and wait until ComfyUI answers."""
    cfg = _state["cfg"]
    runtime.up(cfg)
    typer.echo(f"ComfyUI is up at http://127.0.0.1:{cfg.comfyui.port}")


@gpu_app.command("down")
def gpu_down():
    """Stop the container and wait until its processes have left the GPU."""
    runtime.down(_state["cfg"])
    typer.echo("ComfyUI is down; the GPU is free")


@gpu_app.command("status")
def gpu_status():
    """Whether the container runs, and what ComfyUI reports about itself and the GPU."""
    cfg = _state["cfg"]
    if not runtime.is_running():
        typer.echo("ComfyUI is down")
        return
    stats = runtime.client(cfg).wait_ready(10)
    system, device = stats.get("system", {}), (stats.get("devices") or [{}])[0]
    typer.echo(f"ComfyUI {system.get('comfyui_version')} with PyTorch {system.get('pytorch_version')}, "
               f"{device.get('vram_free', 0) / 2**30:.1f} of {device.get('vram_total', 0) / 2**30:.1f} GiB free, "
               f"container pids {sorted(runtime.container_pids())}")


@app.command("render-test")
def render_test(seconds: int = typer.Option(5, help="Clip length in whole seconds (24 fps gives 8n+1 frames)."),
                seed: int = typer.Option(42, help="Noise seed."),
                still: Path = typer.Option(None, help="Start frame; default projects/_tests/still.png."),
                prompt: str = typer.Option(RENDER_TEST_PROMPT, help="What happens in the clip.")):
    """Render one LTX-2.5 image-to-video clip and report time, GPU memory and the file."""
    cfg = _state["cfg"]
    if not runtime.is_running():
        typer.echo("FAIL  ComfyUI is down; run: uv run studio gpu up")
        raise typer.Exit(1)
    verdict = run_preflight(cfg.gpu.agent_service, min_free_gib=0, own_pids=runtime.container_pids())
    if not verdict.ok:
        typer.echo("FAIL  " + "; ".join(verdict.problems))
        raise typer.Exit(1)
    template_path, slots_path = WORKFLOWS / "ltx25_i2v.json", WORKFLOWS / "ltx25_i2v.slots.json"
    if not template_path.exists() or not slots_path.exists():
        typer.echo(f"FAIL  {template_path.name} and its slot map are missing; export the workflow first")
        raise typer.Exit(1)
    still = still or cfg.projects_dir / "_tests" / "still.png"
    if not still.exists():
        typer.echo(f"FAIL  no start frame at {still}")
        raise typer.Exit(1)
    shutil.copyfile(still, cfg.comfy_io_dir / "input" / "render_test.png")
    workflow = fill(strip_prompt_enhancer(json.loads(template_path.read_text())), json.loads(slots_path.read_text()),
                    {"prompt": prompt, "image": "render_test.png", "seconds": seconds, "width": 1280, "height": 704,
                     "seed": seed, "prefix": "studio/render_test"})
    started = time.monotonic()
    with VramSampler() as vram:
        entry = runtime.client(cfg).run(workflow, timeout_s=1800)
    elapsed = time.monotonic() - started
    videos = [f for f in output_files(entry) if f[1].endswith(".mp4")]
    if not videos:
        typer.echo(f"FAIL  the run finished without an MP4: {output_files(entry)}")
        raise typer.Exit(1)
    subfolder, filename, _ = videos[0]
    out = private_dir(cfg.projects_dir / "_tests") / f"ltx25_render_test_seed{seed}.mp4"
    shutil.copyfile(cfg.comfy_io_dir / "output" / subfolder / filename, out)
    info = ffmpeg.probe(cfg, out)
    typer.echo(f"PASS  {info['duration_s']:.2f} s {info['width']}x{info['height']} {info['fps']:g} fps, "
               f"audio {info['audio']}, {elapsed:.1f} s wall, GPU memory peak "
               f"{vram.peak_over_baseline_gib:.1f} GiB over the display baseline")
    typer.echo(f"      {out}")
