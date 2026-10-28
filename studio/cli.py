import contextlib
import json
import re
import secrets
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
import typer
from pydantic import ValidationError

from studio import bench, blind, cast, film, identity, keyframe_bench, models, provenance, refs, scene, sound
from studio.claude.client import Claude, CredentialsError, Refused
from studio.claude.ledger import BudgetExceeded, Ledger, summarize
from studio.config import Config, load_config
from studio.doctor import exit_code, run_doctor
from studio.fs import private_dir, write_private
from studio.gpu import runtime
from studio.gpu.comfy import RenderError, WorkflowError
from studio.gpu.preflight import run_preflight
from studio.media import assemble, ffmpeg
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
identity_app = typer.Typer(no_args_is_help=True, help="Face profiles of real people and identity scores, on this "
                                                         "machine only.")
app.add_typer(identity_app, name="identity")
bench_app = typer.Typer(no_args_is_help=True, help="Fixed benches that measure variants against each other.")
app.add_typer(bench_app, name="bench")
cast_app = typer.Typer(no_args_is_help=True, help="A person's capture: photos and videos in cast/ALIAS.")
app.add_typer(cast_app, name="cast")
film_app = typer.Typer(no_args_is_help=True, help="A short film from a prompt and pictures: Claude's shot plan, one "
                                                  "still and one Wan clip per shot, one MP4; sound added after.")
app.add_typer(film_app, name="film")
RENDER_TEST_PROMPT = ("A slow push-in toward a misty pond at morning. Fog drifts low across the water, willow "
                      "leaves and tall meadow grass sway in a light breeze, soft sunlight. Quiet outdoor ambience: "
                      "light wind and distant birds.")
_state: dict[str, Config] = {}


def exit_on_kill() -> None:
    """A kill (SIGTERM) leaves the way Ctrl+C does, through each command's cleanup: copies of pictures removed from
    ComfyUI's folders, the container stopped."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


@app.callback()
def main(config: Path = typer.Option(None, "--config", help="Config file; default config/studio.yaml.")):
    exit_on_kill()
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
    try:
        problems = models.fetch(_state["cfg"], name)
    except models.DiskFloor as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    _report_models(name, problems)


@gpu_app.command("build")
def gpu_build():
    """Build the ComfyUI image from docker/comfyui with the pins in config/studio.yaml."""
    runtime.build(_state["cfg"])


@gpu_app.command("up")
def gpu_up():
    """Check the GPU is free, start the container and wait until ComfyUI answers."""
    cfg = _state["cfg"]
    _up_or_exit(cfg)
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


def _up_or_exit(cfg: Config) -> None:
    try:
        runtime.up(cfg)
    except runtime.GpuBusy as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)


def _gpu_ours_or_exit(cfg: Config) -> None:
    verdict = run_preflight(cfg.gpu, min_free_gib=0, own_pids=runtime.container_pids())
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
        _up_or_exit(cfg)
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
                BudgetExceeded, anthropic.APIError, ValidationError, runtime.GpuBusy)


def face_tools(cfg: Config) -> refs.FaceTools:
    """YuNet and SFace, plus AuraFace once its model set is downloaded."""
    detect, identify = cfg.models["face-detect"], cfg.models["face-id"]
    extra = {}
    if "auraface" in cfg.models and not models.present(cfg, "auraface"):
        extra["auraface"] = refs.AuraFace(cfg.model_root("auraface") / cfg.models["auraface"].files[0].target)
    return refs.FaceTools(cfg.model_root("face-detect") / detect.files[0].path,
                          cfg.model_root("face-id") / identify.files[0].path, extra=extra)


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
def scene_stills(pictures: list[Path] = typer.Argument(..., help="A folder of pictures, or picture files: photos of "
                                                                 "the person and pictures of places or things."),
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
                video: str = typer.Option("ltx-int8-2stage", help=f"Video variant: {', '.join(bench.VARIANTS)}."),
                keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """Clips of the chosen stills with the scene's motion prompt, by a bench video variant (LTX-2.5 or Wan 2.2)."""
    cfg = _state["cfg"]
    if video not in bench.VARIANTS:
        typer.echo(f"FAIL  unknown video variant {video}; known: {', '.join(bench.VARIANTS)}")
        raise typer.Exit(1)
    _models_or_exit(cfg, bench.VARIANTS[video]["models"])
    try:
        scene.load(cfg, name)
        made = scene.make_clips(cfg, name=name, numbers=numbers, seconds=seconds, seed=seed, echo=typer.echo,
                                gpu=lambda: _comfy_session(cfg, keep_up), video=video)
    except SCENE_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    count = len(numbers)
    typer.echo(f"PASS  {count} clip{'s' if count > 1 else ''}; review: "
               f"{scene.scene_dir(cfg, made['name']) / 'review.html'}")


VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".webm"}


@identity_app.command("profile")
def identity_profile(person: str = typer.Option(..., help="Alias: the folder name under cast/."),
                     leave_out: list[str] = typer.Option([], "--leave-out",
                                                         help="Photo file names to keep out, such as bench frames.")):
    """Build a person's face profiles from cast/ALIAS/photos: the person in most photos, one face per photo, one
    profile per scorer (SFace, and AuraFace once downloaded)."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["face-detect", "face-id"])
    try:
        profiles = identity.build_profiles(cfg, person, face_tools(cfg), leave_out=leave_out)
    except ValueError as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    for profile in profiles.values():
        identity.save_profile(cfg, profile)
    first = next(iter(profiles.values()))
    typer.echo(f"PASS  profile of {person}: {len(first.sources)} photos, digest {identity.digest(profiles)}")
    for name, p in profiles.items():
        typer.echo(f"      {name}: genuine pairs {len(p.genuine)}: median {p.genuine_median:.3f}, p5 "
                   f"{p.genuine_p5:.3f}, min {float(p.genuine.min()):.3f}")


@identity_app.command("score")
def identity_score(files: list[Path] = typer.Argument(..., help="Images or clips to score."),
                   person: str = typer.Option(..., help="Alias whose profiles to score against.")):
    """Score how much each face looks like the person, as z against their genuine pairs (0 is the middle of their
    real photo pairs). Files made from a model whose license forbids face recognition, or of unknown origin, are
    refused."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["face-detect", "face-id"])
    try:
        profiles = identity.load_profiles(cfg, person)
    except FileNotFoundError as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    tools, refused = face_tools(cfg), False
    for path in files:
        problems = provenance.problems(cfg, path)
        if problems:
            for problem in problems:
                typer.echo(f"REFUSED  {problem}")
            refused = True
            continue
        if path.suffix.lower() in VIDEO_TYPES:
            pinned = [] if provenance.is_camera_original(cfg, path) else provenance.params(path).get("pinned_frames", [0])
            s = identity.score_clip(profiles, tools, path, pinned).summary
            if s.median is None:
                typer.echo(f"{path.name}  no scorable face (coverage {s.coverage:.0%})")
                continue
            cosines = ", ".join(f"{k} {v:.3f}" for k, v in sorted(s.by_scorer.items()))
            typer.echo(f"{path.name}  score {s.median:+.2f} ({cosines}), mid-clip min {s.mid_min:+.2f}, dips {s.dips} "
                       f"({s.dip_seconds:.2f} s), coverage {s.coverage:.0%}, yaw range {s.yaw_range:.2f}")
        else:
            frame = identity.score_image(profiles, tools, path)
            if frame.sim is None:
                typer.echo(f"{path.name}  no face found")
                continue
            cosines = ", ".join(f"{k} {v:.3f}" for k, v in sorted(frame.sims.items()))
            typer.echo(f"{path.name}  score {frame.sim:+.2f} ({cosines}), face {frame.px:.0f} px, yaw {frame.yaw:.2f}")
    if refused:
        raise typer.Exit(1)


@bench_app.command("identity")
def bench_identity(person: str = typer.Option(..., help="Alias whose real photos start the clips."),
                   variant: str = typer.Option("ltx-int8-2stage", help=f"One of: {', '.join(bench.VARIANTS)}."),
                   name: str = typer.Option(None, help="Run name; default the time and the variant."),
                   count: int = typer.Option(6, min=1, max=12, help="How many start photos."),
                   keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """Render every bench cell with one variant and score each clip against the person's profile."""
    cfg = _state["cfg"]
    if variant not in bench.VARIANTS:
        typer.echo(f"FAIL  unknown variant {variant}; known: {', '.join(bench.VARIANTS)}")
        raise typer.Exit(1)
    _models_or_exit(cfg, ["face-detect", "face-id", *bench.VARIANTS[variant]["models"]])
    run_name = scene.clean_name(name or f"{datetime.now():%Y%m%d-%H%M}-{variant}")
    run_dir = cfg.data_root / "bench" / run_name
    if (run_dir / "results.json").exists():
        typer.echo(f"FAIL  bench run {run_name} exists in {run_dir}; pick another name")
        raise typer.Exit(1)
    try:
        with _comfy_session(cfg, keep_up):
            results = bench.run(cfg, alias=person, variant=variant, tools=face_tools(cfg),
                                run_dir=private_dir(run_dir), echo=typer.echo, count=count)
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    for motion in dict.fromkeys(c["motion"] for c in results["cells"]):
        cells = [c for c in results["cells"] if c["motion"] == motion and c["median"] is not None]
        if not cells:
            typer.echo(f"{motion:<7} no scorable face")
            continue
        mean = sum(c["median"] for c in cells) / len(cells)
        low = sum(c["mid_min"] for c in cells) / len(cells)
        cover = sum(c["coverage"] for c in cells) / len(cells)
        typer.echo(f"{motion:<7} median {mean:.3f} min {low:.3f} dips {sum(c['dips'] for c in cells)} "
                   f"coverage {cover:.0%} ({len(cells)} cells)")
    typer.echo(f"PASS  {len(results['cells'])} clips; review: {run_dir / 'review.html'}")


@bench_app.command("compare")
def bench_compare(runs: list[str] = typer.Argument(..., help="Run names under bench/; the first is the baseline."),
                  person: str = typer.Option(..., help="Alias the runs were scored against.")):
    """Each run against the first, cell by cell, by the round rule; writes a side-by-side page."""
    cfg = _state["cfg"]
    loaded = []
    for name in runs:
        run_dir = cfg.data_root / "bench" / scene.clean_name(name)
        if not (run_dir / "results.json").is_file():
            typer.echo(f"FAIL  no bench run {name} in {run_dir.parent}")
            raise typer.Exit(1)
        loaded.append((run_dir.name, json.loads((run_dir / "results.json").read_text(encoding="utf-8"))))
    try:
        report = bench.compare([data for _, data in loaded], delta=bench.DELTA)
    except ValueError as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    typer.echo(f"scores are z against {person}'s genuine pairs; delta {report['delta']:.2f}; per motion: median "
               f"(drift from the start frame)")
    for row in report["variants"]:
        motions = "  ".join(f"{m} {v['median']:+.2f} ({v['drift'] or 0:+.2f})" for m, v in row["motions"].items())
        dips = sum(v["dips"] for v in row["motions"].values())
        paired = (f"  gain {row['gain']:+.2f} wins {row['photo_wins']}/{row['photos']}"
                  if row.get("gain") is not None else "")
        typer.echo(f"{row['variant']:<22} {motions}  dips {dips}  {row['render_s'] or 0:.0f} s/clip{paired}  "
                   f"{row['verdict']}")
    page = cfg.data_root / "bench" / f"compare-{datetime.now():%Y%m%d-%H%M%S}.html"
    write_private(page, bench.compare_page(loaded, report))
    typer.echo(f"PASS  page: {page}")


@bench_app.command("rescore")
def bench_rescore(runs: list[str] = typer.Argument(..., help="Run names under bench/."),
                  person: str = typer.Option(..., help="Alias to score against.")):
    """Score finished runs again from their clips against the person's current profiles."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["face-detect", "face-id"])
    tools = face_tools(cfg)
    for name in runs:
        run_dir = cfg.data_root / "bench" / scene.clean_name(name)
        if not (run_dir / "results.json").is_file():
            typer.echo(f"FAIL  no bench run {name} in {run_dir.parent}")
            raise typer.Exit(1)
        try:
            bench.rescore(cfg, alias=person, tools=tools, run_dir=run_dir, echo=typer.echo)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            typer.echo(f"FAIL  {error}")
            raise typer.Exit(1)
        typer.echo(f"PASS  rescored {run_dir.name}")


@bench_app.command("keyframes")
def bench_keyframes(person: str = typer.Option(..., help="Alias whose photos are the references."),
                    engine: str = typer.Option(..., help=f"One of: {', '.join(keyframe_bench.ENGINES)}."),
                    name: str = typer.Option(None, help="Run name; default the time and the engine."),
                    limit: int = typer.Option(None, min=1, help="Only the first N shot prompts."),
                    keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """Render the bake-off's shot prompts with one engine from the person's best photos. Apache engines' stills are
    face-scored; FLUX 3 Klein's are judged by eye."""
    cfg = _state["cfg"]
    if engine not in keyframe_bench.ENGINES:
        typer.echo(f"FAIL  unknown engine {engine}; known: {', '.join(keyframe_bench.ENGINES)}")
        raise typer.Exit(1)
    _models_or_exit(cfg, ["face-detect", "face-id", *keyframe_bench.ENGINES[engine]["models"]])
    run_name = scene.clean_name(name or f"{datetime.now():%Y%m%d-%H%M}-kf-{engine}")
    run_dir = cfg.data_root / "bench" / run_name
    if (run_dir / "results.json").exists():
        typer.echo(f"FAIL  bench run {run_name} exists in {run_dir}; pick another name")
        raise typer.Exit(1)
    try:
        with _comfy_session(cfg, keep_up):
            results = keyframe_bench.run(cfg, alias=person, engine=engine, tools=face_tools(cfg),
                                         run_dir=private_dir(run_dir), echo=typer.echo, limit=limit)
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    scores = [c["score"] for c in results["cells"] if c["score"] is not None]
    summary = f"mean score {sum(scores) / len(scores):+.2f} over {len(scores)} stills" if scores else "judged by eye"
    typer.echo(f"PASS  {len(results['cells'])} stills, {summary}; review: {run_dir / 'review.html'}")


STRIP_TIMES = (1.0, 2.5, 4.0)


def _run_results(cfg: Config, name: str) -> tuple[Path, dict]:
    run_dir = cfg.data_root / "bench" / scene.clean_name(name)
    if not (run_dir / "results.json").is_file():
        typer.echo(f"FAIL  no bench run {name} in {run_dir.parent}")
        raise typer.Exit(1)
    return run_dir, json.loads((run_dir / "results.json").read_text(encoding="utf-8"))


def _blind_dir(cfg: Config, name: str | None) -> Path:
    out = cfg.data_root / "bench" / "blind" / scene.clean_name(name or f"{datetime.now():%Y%m%d-%H%M%S}")
    if (out / "key.json").exists():
        typer.echo(f"FAIL  blind set {out.name} exists; pick another name")
        raise typer.Exit(1)
    return out


@bench_app.command("blind")
def bench_blind(runs: list[str] = typer.Argument(..., help="Two run names; clips are paired by cell."),
                name: str = typer.Option(None, help="Name of the blind set; default the time."),
                seed: int = typer.Option(None, help="Shuffle seed; random when omitted.")):
    """A blind page of paired clips as frame strips at matched times, no labels or scores."""
    cfg = _state["cfg"]
    if len(runs) != 2:
        typer.echo("FAIL  give two runs")
        raise typer.Exit(1)
    loaded = [(_run_results(cfg, r), r) for r in runs]
    key = lambda c: (c["start"], c["motion"], c["seed"])  # noqa: E731
    cells = [{key(c): c for c in data["cells"] if c.get("median") is not None} for (_, data), _ in loaded]
    common = [k for k in cells[0] if k in cells[1]]
    out = _blind_dir(cfg, name)
    groups = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in common:
            group = []
            for ((run_dir, data), run_name), by_key in zip(loaded, cells):
                cell = by_key[k]
                frames = []
                for t in STRIP_TIMES:
                    frame = Path(tmp) / f"{run_name}-{cell['clip']}-{t}.png"
                    ffmpeg.run(cfg, "-y", "-v", "error", "-ss", str(t), "-i", str(run_dir / "clips" / cell["clip"]),
                               "-frames:v", "1", "-vf", "scale=640:-2", str(frame)).check_returncode()
                    frames.append(frame)
                group.append({"id": f"{run_name}/{cell['start']}", "run": data["variant"], "images": frames,
                              "score": cell["median"]})
            groups.append(group)
        blind.make(groups, out, seed if seed is not None else secrets.randbelow(2**31))
    typer.echo(f"PASS  {len(groups)} pairs; open {out / 'page.html'}, then: uv run studio bench pick {out.name} 1A 2B ...")


@bench_app.command("blind-keyframes")
def bench_blind_keyframes(runs: list[str] = typer.Argument(..., help="Keyframe runs, one per engine."),
                          name: str = typer.Option(None, help="Name of the blind set; default the time."),
                          seed: int = typer.Option(None, help="Shuffle seed; random when omitted.")):
    """A blind page of each shot's stills across engines, no labels or scores."""
    cfg = _state["cfg"]
    loaded = [(_run_results(cfg, r), r) for r in runs]
    by_shot = [{(c["shot"], c["seed"]): c for c in data["cells"]} for (_, data), _ in loaded]
    shots = [k for k in by_shot[0] if all(k in other for other in by_shot[1:])]
    groups = [[{"id": f"{run_name}/shot{k[0]:02d}", "run": data["engine"],
                "images": [run_dir / "stills" / cells[k]["file"]], "score": cells[k].get("score")}
               for ((run_dir, data), run_name), cells in zip(loaded, by_shot)] for k in shots]
    out = _blind_dir(cfg, name)
    blind.make(groups, out, seed if seed is not None else secrets.randbelow(2**31))
    typer.echo(f"PASS  {len(groups)} shots; open {out / 'page.html'}, then: uv run studio bench pick {out.name} 1A 2C ...")


@bench_app.command("pick")
def bench_pick(name: str = typer.Argument(..., help="The blind set."),
               picks: list[str] = typer.Argument(..., help="Group number and letter, like 1A 2B, or 3= for no pick.")):
    """Record picks on a blind page; shows wins per variant or engine and agreement with the scores."""
    cfg = _state["cfg"]
    out = cfg.data_root / "bench" / "blind" / scene.clean_name(name)
    if not (out / "key.json").is_file():
        typer.echo(f"FAIL  no blind set {name}")
        raise typer.Exit(1)
    try:
        report = blind.pick(out, picks, delta=bench.DELTA)
    except ValueError as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    typer.echo("wins: " + ", ".join(f"{run} {count}" for run, count in report["wins"].items()))
    scored = report["agree"] + report["disagree"]
    typer.echo(f"scores agree {report['agree']} of {scored} picks where they differ by more than {bench.DELTA:.2f}; "
               f"no preference {report['ties']}")


@cast_app.command("ingest")
def cast_ingest(person: str = typer.Option(..., help="Alias: the folder name under cast/."),
                per_bin: int = typer.Option(5, min=1, help="Images kept per head-angle bin."),
                fps: float = typer.Option(2.0, help="Frames read per second of video."),
                trigger: str = typer.Option(None, help="Caption trigger word; default zq plus the alias."),
                replace: bool = typer.Option(False, "--replace", help="Build the dataset again.")):
    """Turn cast/ALIAS/{photos,video} into a training set binned by head angle, near-copies dropped, each image
    checked against the person's profile and traced back to its camera file."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, ["face-detect", "face-id"])
    try:
        report = cast.ingest(cfg, person, face_tools(cfg), echo=typer.echo, per_bin=per_bin, fps=fps,
                             trigger=trigger, replace=replace)
    except (ValueError, FileNotFoundError) as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    if report["missing"]:
        typer.echo(f"missing angles: {', '.join(report['missing'])}; film those to fill them")
    typer.echo(f"PASS  {report['kept']} images in {report['dataset']}, trigger word {report['trigger']}")


@scene_app.command("compare")
def scene_compare(names: list[str] = typer.Argument(..., help="Scene names, side by side in this order.")):
    """A local page with the scenes' stills and clips side by side."""
    cfg = _state["cfg"]
    try:
        page = scene.compare_page(cfg, names)
    except scene.SceneError as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    out = cfg.projects_dir / "scenes" / f"compare-{'-'.join(scene.clean_name(n) for n in names)}.html"
    write_private(out, page)
    typer.echo(f"PASS  page: {out}")


FILM_ERRORS = (*SCENE_ERRORS, film.FilmError, assemble.AssemblyError, sound.SoundError)
SOUND_FREE_GIB = 24  # PrismAudio with its JAX and torch encoders, with room to spare


@film_app.command("make")
def film_make(pictures: list[Path] = typer.Argument(..., help="A folder of pictures, or picture files: photos of the "
                                                              "person and pictures of places or things."),
              prompt: str = typer.Option(..., help="The film in your words: place, look, what happens."),
              name: str = typer.Option(..., help="Film name; its folder is projects/films/NAME."),
              shots: int = typer.Option(4, min=3, max=6, help="How many shots Claude plans."),
              seconds: int = typer.Option(5, min=2, max=10, help="Shot length in whole seconds; Wan 2.2 is trained "
                                                                 "on 5."),
              portrait: bool = typer.Option(False, "--portrait", help="9:16 instead of 16:9."),
              candidates: int = typer.Option(3, min=1, max=6, help="Candidates of the set picture, the character "
                                                                   "sheet and each shot's still; the film uses the "
                                                                   "first of each."),
              takes: int = typer.Option(4, min=1, max=8, help="Clips made from each shot's picked still (the first "
                                                              "until you pick); the film uses the first."),
              events: list[Path] = typer.Option(None, "--event", help="A picture of what happens (for example a "
                                                                      "real blast over the sea): a shot's event is "
                                                                      "painted from it into the event's box, never "
                                                                      "into the set. Repeat for more."),
              keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """References from your pictures and a shot plan with a continuity bible from Claude (text only); then the look
    (set pictures, character sheets and a picture of each view), each shot's stills drawn from it with events painted
    in, and takes of each shot's picked still, joined into one silent MP4 from the picks."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, film.MODEL_SETS)
    try:
        made = film.make(cfg, name=name, prompt=prompt, pictures=pictures, shots=shots, seconds=seconds,
                         portrait=portrait, claude=Claude(cfg), tools=face_tools(cfg), echo=typer.echo,
                         gpu=lambda: _comfy_session(cfg, keep_up), candidates=candidates, takes=takes,
                         events=events or [])
    except FILM_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    folder = film.film_dir(cfg, made["name"])
    typer.echo(f'PASS  "{made["title"]}", {len(made["shots"])} shots; film: {folder / made["film"]}')
    typer.echo(f"      review: {folder / 'review.html'}")
    typer.echo(f"      sound: uv run studio film sound --name {made['name']} (once the sound worker is set up)")


def _choice(text: str) -> tuple[int, tuple[int, int | None]]:
    """SHOT=STILL or SHOT=STILL.TAKE"""
    shot, _, rest = text.partition("=")
    still, _, take = rest.partition(".")
    if not (shot.isdigit() and still.isdigit() and (take.isdigit() or not take)):
        raise ValueError(f"{text}: use SHOT=STILL or SHOT=STILL.TAKE, for example 2=3 or 2=3.1")
    return int(shot), (int(still), int(take) if take else None)


def _report_cut(data: dict, cfg) -> None:
    if data["film"]:
        typer.echo(f"PASS  {film.film_dir(cfg, data['name']) / data['film']}")
        typer.echo(f"      sound: uv run studio film sound --name {data['name']}")


@film_app.command("pick")
def film_pick(choices: list[str] = typer.Argument(..., help="SHOT=STILL or SHOT=STILL.TAKE, for example 2=3 or 2=3.1; "
                                                            "a still without a take uses its steadiest take."),
              name: str = typer.Option(..., help="Film name.")):
    """Choose which still and take each shot uses, then cut the film again from the picks."""
    cfg = _state["cfg"]
    try:
        made = film.pick(cfg, name=name, choices=dict(_choice(c) for c in choices), echo=typer.echo)
    except (ValueError, *FILM_ERRORS) as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    _report_cut(made, cfg)


@film_app.command("takes")
def film_takes(name: str = typer.Option(..., help="Film name."),
               count: int = typer.Option(4, min=1, max=8, help="Takes each shot's picked still should have."),
               keep_up: bool = typer.Option(False, "--keep-up", help="Leave ComfyUI running afterwards.")):
    """Make takes of every shot's picked still until each has COUNT, then cut the film from the picks."""
    cfg = _state["cfg"]
    _models_or_exit(cfg, bench.VARIANTS[film.VIDEO]["models"])
    try:
        made = film.takes(cfg, name=name, count=count, echo=typer.echo, gpu=lambda: _comfy_session(cfg, keep_up))
    except FILM_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    _report_cut(made, cfg)


@film_app.command("sound")
def film_sound(name: str = typer.Option(..., help="Film name."),
               seed: int = typer.Option(None, help="Seed of shot 1's Foley track; shot N uses seed + N - 1. Random "
                                                   "when omitted."),
               bed: Path = typer.Option(None, help="A recording of the place's ambience to lay under the whole film "
                                                   "instead of a generated bed; it is looped to the film's length.")):
    """The film's sound in layers: one ambience bed under every cut, PrismAudio Foley for the shots with their own
    sound, every generated track checked for speech, and the film mixed to -16 LUFS into film-sound.mp4."""
    cfg = _state["cfg"]
    worker = sound.PrismAudio()
    try:
        film.load(cfg, name)
        if bed is not None and not bed.is_file():
            typer.echo(f"FAIL  no recording at {bed}")
            raise typer.Exit(1)
        problems = worker.problems(cfg)
        if problems:
            for problem in problems:
                typer.echo(f"FAIL  {problem}")
            raise typer.Exit(1)
        # The worker needs the GPU: our ComfyUI goes down first, and nothing else may hold it.
        if runtime.is_running():
            runtime.down(cfg)
        verdict = run_preflight(cfg.gpu, min_free_gib=SOUND_FREE_GIB)
        if not verdict.ok:
            typer.echo("FAIL  " + "; ".join(verdict.problems))
            raise typer.Exit(1)
        made = film.add_sound(cfg, name=name, worker=worker, echo=typer.echo,
                              seed=None if seed is None else seed - 1, bed_file=bed)
    except FILM_ERRORS as error:
        typer.echo(f"FAIL  {error}")
        raise typer.Exit(1)
    typer.echo(f"PASS  {film.film_dir(cfg, made['name']) / made['sound']['film']}")
