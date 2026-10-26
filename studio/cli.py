import contextlib
import re
import secrets
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
import typer
from pydantic import ValidationError

from studio import models, refs, scene
from studio.claude.client import Claude, CredentialsError, Refused
from studio.claude.ledger import BudgetExceeded, Ledger, summarize
from studio.config import Config, load_config
from studio.doctor import exit_code, run_doctor
from studio.fs import private_dir
from studio.gpu import runtime
from studio.gpu.comfy import RenderError, WorkflowError
from studio.gpu.preflight import run_preflight
from studio.media import ffmpeg
from studio.media.smoke import TARGET_LUFS, smoke_mux
from studio.render import LANDSCAPE, image_size, prepare_still, render_i2v, target_size
from studio.scene_prompts import PromptError

# Tracebacks must never print locals: the Claude client holds the API key.
app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False,
                  help="Short films from a written story.")
models_app = typer.Typer(no_args_is_help=True, help="Pinned model weights.")
app.add_typer(models_app, name="models")
gpu_app = typer.Typer(no_args_is_help=True, help="The ComfyUI container on this GPU.")
app.add_typer(gpu_app, name="gpu")
scene_app = typer.Typer(no_args_is_help=True, help="A scene from a prompt and pictures: stills, then clips of the "
                                                   "stills you pick.")
app.add_typer(scene_app, name="scene")
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
    typer.echo(f"ComfyUI is up at {cfg.comfyui.base_url}")


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


def _report_clip(result) -> None:
    info = ffmpeg.probe(_state["cfg"], result.clip)
    peak = f"{result.peak_gib:.1f} GiB on the card" if result.peak_gib is not None else "not measured"
    reused = f", {result.reused_nodes} nodes reused from an earlier run" if result.reused_nodes else ""
    typer.echo(f"PASS  {info['duration_s']:.2f} s {info['width']}x{info['height']} {info['fps']:g} fps, "
               f"audio {info['audio']}, {result.elapsed_s:.1f} s wall, GPU memory peak {peak}{reused}")
    typer.echo(f"      {result.clip}")


def _gpu_ours_or_exit(cfg: Config) -> None:
    verdict = run_preflight(cfg.gpu.agent_service, min_free_gib=0, own_pids=runtime.container_pids())
    if not verdict.ok:
        typer.echo("FAIL  " + "; ".join(verdict.problems))
        raise typer.Exit(1)


@app.command("render-test")
def render_test(seconds: int = typer.Option(5, help="Clip length in whole seconds (24 fps gives 8n+1 frames)."),
                seed: int = typer.Option(42, help="Noise seed."),
                still: Path = typer.Option(None, help="Start frame, 1280x704; default projects/_tests/still.png."),
                prompt: str = typer.Option(RENDER_TEST_PROMPT, help="What happens in the clip.")):
    """Render one LTX-2.5 image-to-video clip and report time, GPU memory and the file."""
    cfg = _state["cfg"]
    if not runtime.is_running():
        typer.echo("FAIL  ComfyUI is down; run: uv run studio gpu up")
        raise typer.Exit(1)
    _gpu_ours_or_exit(cfg)
    still = still or cfg.projects_dir / "_tests" / "still.png"
    if not still.exists():
        typer.echo(f"FAIL  no start frame at {still}")
        raise typer.Exit(1)
    out = private_dir(cfg.projects_dir / "_tests") / f"ltx25_render_test_seed{seed}.mp4"
    try:
        result = render_i2v(cfg, still, prompt, seconds, seed, LANDSCAPE, "render_test", out)
    except (RenderError, WorkflowError, TimeoutError) as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    _report_clip(result)


@app.command("make-video")
def make_video(picture: Path = typer.Argument(..., help="Start frame: any photo or image."),
               prompt: str = typer.Option(..., help="What happens in the clip, including its sound."),
               seconds: int = typer.Option(5, help="Clip length in whole seconds."),
               seed: int = typer.Option(None, help="Noise seed; random when omitted, printed either way."),
               name: str = typer.Option(None, help="Output name; default the picture's file name."),
               keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running for another clip.")):
    """Make one clip with sound from a picture and a prompt. Everything runs on this machine."""
    cfg = _state["cfg"]
    if not picture.exists():
        typer.echo(f"FAIL  no picture at {picture}")
        raise typer.Exit(1)
    seed = seed if seed is not None else secrets.randbelow(2**31)
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name or picture.stem).strip("-") or "clip"
    started_here = not runtime.is_running()
    if started_here:
        runtime.up(cfg)
    try:
        _gpu_ours_or_exit(cfg)
        size = target_size(*image_size(cfg, picture))
        clip = private_dir(cfg.projects_dir / "videos") / f"{name}-seed{seed}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            still = prepare_still(cfg, picture, Path(tmp) / "start.png", size)
            try:
                result = render_i2v(cfg, still, prompt, seconds, seed, size, name, clip)
            except (RenderError, WorkflowError, TimeoutError) as error:
                typer.echo(f"FAIL  {error}")
                raise typer.Exit(1)
        clip.with_suffix(".txt").write_text(
            f"prompt: {prompt}\nseconds: {seconds}\nseed: {seed}\nsize: {size[0]}x{size[1]}\npicture: {picture}\n"
            f"model: LTX-2.5 distilled int8, image {runtime.image_tag(cfg)}\n"
            f"render: {result.elapsed_s:.1f} s\n")
        _report_clip(result)
    finally:
        if started_here and not keep_up:
            runtime.down(cfg)


SCENE_ERRORS = (scene.SceneError, PromptError, RenderError, WorkflowError, TimeoutError, Refused, CredentialsError,
                BudgetExceeded, anthropic.APIError, ValidationError)


def face_tools(cfg: Config) -> refs.FaceTools:
    detect, identify = cfg.models["face-detect"], cfg.models["face-id"]
    return refs.FaceTools(cfg.model_root("face-detect") / detect.files[0].path,
                          cfg.model_root("face-id") / identify.files[0].path)


def _models_or_exit(cfg: Config, names: list[str]) -> None:
    for name in names:
        problems = models.present(cfg, name) if name in cfg.models else [f"{name} is not in the config"]
        if problems:
            typer.echo(f"FAIL  model set {name}: {problems[0]}; run: uv run studio models fetch {name}")
            raise typer.Exit(1)


@contextlib.contextmanager
def _comfy_session(cfg: Config, keep_up: bool):
    """ComfyUI for the length of a command: started here if it was down, stopped again unless --keep-up."""
    started_here = not runtime.is_running()
    if started_here:
        runtime.up(cfg)
    try:
        _gpu_ours_or_exit(cfg)
        yield
    finally:
        if started_here and not keep_up:
            runtime.down(cfg)


@scene_app.command("stills")
def scene_stills(pictures: list[Path] = typer.Argument(..., help="Photos of the person, and pictures of places or "
                                                                 "things to use."),
                 prompt: str = typer.Option(..., help="The scene in your words: place, look, expression, motion."),
                 name: str = typer.Option(..., help="Scene name; its folder is projects/scenes/NAME."),
                 count: int = typer.Option(4, min=1, max=12, help="How many stills to make."),
                 portrait: bool = typer.Option(False, "--portrait", help="9:16 instead of 16:9."),
                 keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """References from your pictures, prompts from Claude (text only), then FLUX 3 Klein stills."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["face-detect", "face-id", "flux3-klein"])
    try:
        made = scene.make_stills(cfg, name=name, prompt=prompt, pictures=pictures, count=count, portrait=portrait,
                                 claude=Claude(cfg), tools=face_tools(cfg), echo=typer.echo,
                                 gpu=lambda: _comfy_session(cfg, keep_up))
    except SCENE_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    folder = scene.scene_dir(cfg, made["name"])
    typer.echo(f"PASS  {len(made['stills'])} stills; review: {folder / 'review.html'}")
    typer.echo(f"      next: uv run studio scene clips --name {made['name']} STILL... (still numbers from the review)")


@scene_app.command("clips")
def scene_clips(numbers: list[int] = typer.Argument(..., help="Still numbers to animate, as on the review page."),
                name: str = typer.Option(..., help="Scene name."),
                seconds: int = typer.Option(5, min=1, max=10, help="Clip length in whole seconds."),
                seed: int = typer.Option(None, help="Noise seed for every clip; random when omitted."),
                keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """LTX-2.5 clips with sound of the chosen stills, with the scene's motion prompt."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["ltx25-int8"])
    try:
        scene.load(cfg, name)
        made = scene.make_clips(cfg, name=name, numbers=numbers, seconds=seconds, seed=seed, echo=typer.echo,
                                gpu=lambda: _comfy_session(cfg, keep_up))
    except SCENE_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    count = len(numbers)
    typer.echo(f"PASS  {count} clip{'s' if count > 1 else ''}; review: "
               f"{scene.scene_dir(cfg, made['name']) / 'review.html'}")
