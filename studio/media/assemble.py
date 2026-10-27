"""A film from its shots: clips joined with hard cuts, then optionally one sound track per shot, fitted to the shot,
joined and normalized in two passes to the studio's loudness."""

import json
import re
import subprocess
from pathlib import Path

from studio.config import Config
from studio.media import ffmpeg
from studio.media.smoke import TARGET_LUFS

TRUE_PEAK = -1.5
LRA = 11
QUIET_LUFS = -70.0  # below this a track is silence and is left as it is


class AssemblyError(RuntimeError):
    pass


def _check(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise AssemblyError(f"{what} failed: {lines[-1] if lines else 'no message'}")


def frame_count(cfg: Config, path: Path) -> int:
    out = subprocess.run([str(ffmpeg.binary(cfg, "ffprobe")), "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def silent_film(cfg: Config, clips: list[Path], out: Path, *, seconds: int, fps: int) -> None:
    """Each clip cut to its first `seconds` (Wan's 81 frames become 80), joined in order, H.264 at the clips' rate."""
    frames = seconds * fps
    for clip in clips:
        if frame_count(cfg, clip) < frames:
            raise AssemblyError(f"{clip.name} has fewer than {frames} frames")
    cuts = ";".join(f"[{i}:v]trim=end_frame={frames},setpts=PTS-STARTPTS[v{i}]" for i in range(len(clips)))
    graph = cuts + ";" + "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[v]"
    inputs = [arg for clip in clips for arg in ("-i", str(clip))]
    _check(ffmpeg.run(cfg, "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]", "-r", str(fps),
                      "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                      "-movflags", "+faststart", str(out), timeout=3600), "joining the clips")


def parse_loudnorm(log: str) -> dict:
    """The measured values from loudnorm's JSON report, the last {...} block in ffmpeg's log."""
    block = re.findall(r"\{[^{}]*\}", log)
    if not block:
        raise AssemblyError("no loudnorm report in the ffmpeg log")
    return {key: float(value) for key, value in json.loads(block[-1]).items()
            if key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}


def _tracks_graph(count: int, seconds: int) -> str:
    """Each track resampled to 48 kHz stereo, cut or padded with silence to exactly `seconds`, then joined."""
    fitted = ";".join(f"[{i + 1}:a]aresample=48000:resampler=soxr,aformat=channel_layouts=stereo,atrim=end={seconds},"
                      f"apad=whole_dur={seconds},asetpts=PTS-STARTPTS[a{i}]" for i in range(count))
    return fitted + ";" + "".join(f"[a{i}]" for i in range(count)) + f"concat=n={count}:v=0:a=1[mix]"


def with_sound(cfg: Config, film: Path, tracks: list[Path], out: Path, *, seconds: int) -> None:
    """The silent film's picture with one track per shot, at TARGET_LUFS and at most TRUE_PEAK, AAC 48 kHz."""
    inputs = ["-i", str(film), *[arg for track in tracks for arg in ("-i", str(track))]]
    joined = _tracks_graph(len(tracks), seconds)
    target = f"I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA={LRA}"
    first = ffmpeg.run(cfg, "-nostats", *inputs, "-filter_complex",
                       f"{joined};[mix]loudnorm={target}:print_format=json[out]", "-map", "[out]", "-f", "null", "-",
                       timeout=1800)
    _check(first, "measuring the sound")
    m = parse_loudnorm(first.stderr)
    if m["input_i"] < QUIET_LUFS:
        level = "anull"
    else:
        level = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                 f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
                 "linear=true")
    # loudnorm works at 192 kHz inside, so the chain ends back at 48 kHz.
    _check(ffmpeg.run(cfg, "-y", "-v", "error", *inputs, "-filter_complex",
                      f"{joined};[mix]{level},aresample=48000:resampler=soxr[out]", "-map", "0:v", "-map", "[out]",
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
                      str(out), timeout=1800), "mixing the sound")
