"""A film from its shots: clips joined with hard cuts, then optionally one sound track per shot. Each track is made
HANDLE seconds longer at both ends; the mix brings every shot to one level, keeps stereo from cancelling in mono,
crossfades at each cut without moving any sound, and keeps the film's true peak under TRUE_PEAK after AAC."""

import subprocess
import wave
from pathlib import Path

import numpy as np

from studio.config import Config
from studio.media import ffmpeg
from studio.media.ffmpeg import parse_ebur128_summary
from studio.media.smoke import TARGET_LUFS, measure

TRUE_PEAK = -1.5
LIMIT = -2.0  # dBTP before AAC, which can add a few tenths
HANDLE = 0.5  # seconds of sound made past each end of a shot
CROSSFADE = 0.5  # seconds, centred on each cut
EDGE_FADE = 0.25  # seconds, at the film's start and end
FLIP_BELOW = -0.5  # correlation under which the right channel is inverted rather than narrowed
QUIET_LUFS = -70.0  # below this a track is silence and is left as it is
LIMITER = (f"aresample=192000:resampler=soxr,alimiter=limit={10 ** (LIMIT / 20):.4f}:attack=1:release=50:level=0:"
           "latency=1,aresample=48000:resampler=soxr")  # 4x oversampled, so it holds true peaks


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


def padded_clip(cfg: Config, clip: Path, out: Path, *, seconds: int, fps: int) -> Path:
    """The shot as the film shows it (the clip's first `seconds`) with its first and last frames held for HANDLE
    seconds: the picture the sound worker makes the handles from."""
    graph = (f"trim=end_frame={seconds * fps},setpts=PTS-STARTPTS,"
             f"tpad=start_mode=clone:start_duration={HANDLE}:stop_mode=clone:stop_duration={HANDLE}")
    _check(ffmpeg.run(cfg, "-y", "-v", "error", "-i", str(clip), "-vf", graph, "-r", str(fps), "-c:v", "libx264",
                      "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(out), timeout=600),
           "padding the clip")
    return out


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """(sample rate, float samples (n, 2)) of a 16-bit WAV; mono is copied to both channels."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise AssemblyError(f"{path.name} is not 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
    samples = samples.reshape(-1, channels)
    return rate, samples if channels == 2 else np.repeat(samples[:, :1], 2, axis=1)


def correlation(samples: np.ndarray) -> float:
    """Left-right correlation from -1 (cancels in mono) to 1 (mono); silence counts as 1."""
    left, right = samples[:, 0] - samples[:, 0].mean(), samples[:, 1] - samples[:, 1].mean()
    norm = np.sqrt((left ** 2).sum() * (right ** 2).sum())
    return float((left * right).sum() / norm) if norm > 0 else 1.0


def stereo_fix(samples: np.ndarray) -> tuple[str, float]:
    """("kept", 1.0), ("narrowed", side gain) or ("inverted", 1.0): what stops the track cancelling in mono.

    Narrowing turns the side (L-R) down and leaves the mid (L+R) alone; at 0.9 of the gain where the correlation
    reaches 0 a little positive correlation is left. Past FLIP_BELOW that would leave almost nothing, so the right
    channel is inverted instead."""
    rho = correlation(samples)
    if rho >= 0:
        return "kept", 1.0
    if rho < FLIP_BELOW:
        return "inverted", 1.0
    mid, side = samples.sum(axis=1) / 2, (samples[:, 0] - samples[:, 1]) / 2
    return "narrowed", round(0.9 * float(np.sqrt(mid.var() / side.var())), 3)


def apply_stereo(samples: np.ndarray, action: str, width: float) -> np.ndarray:
    if action == "inverted":
        return samples * np.array([1.0, -1.0])
    a, b = (1 + width) / 2, (1 - width) / 2
    return samples @ np.array([[a, b], [b, a]])


def _stereo_filter(action: str, width: float) -> str:
    if action == "inverted":
        return "pan=stereo|c0=c0|c1=-1*c1"
    if action == "narrowed":
        a, b = (1 + width) / 2, (1 - width) / 2
        return f"pan=stereo|c0={a:.4f}*c0+{b:.4f}*c1|c1={b:.4f}*c0+{a:.4f}*c1"
    return "anull"


def _shot_sound(cfg: Config, track: Path, seconds: int) -> dict:
    """The stereo fix and the gain that bring the shot's own part of `track` (between the handles) to TARGET_LUFS."""
    rate, samples = read_wav(track)
    core = samples[int(HANDLE * rate):int((HANDLE + seconds) * rate)]
    action, width = stereo_fix(core)
    stereo = _stereo_filter(action, width)
    made, _ = measure(cfg, "-i", str(track), chain=f"atrim={HANDLE}:{HANDLE + seconds},{stereo}")
    return {"lufs_made": made, "gain_db": round(TARGET_LUFS - made, 2) if made > QUIET_LUFS else 0.0,
            "stereo": action, "width": width, "correlation_before": round(correlation(core), 3),
            "correlation_after": round(correlation(apply_stereo(core, action, width)), 3), "filter": stereo}


def _mix_graph(shots: list[dict], seconds: int) -> str:
    """Each track fitted to seconds + 2 HANDLE, fixed and leveled, then cut so that neighbours overlap by CROSSFADE
    centred on their cut: the film starts and ends on the shots' own sound and lasts exactly len(shots) * seconds."""
    count, whole, half = len(shots), seconds + 2 * HANDLE, CROSSFADE / 2
    parts = []
    for i, shot in enumerate(shots):
        start, end = HANDLE - (half if i > 0 else 0), HANDLE + seconds + (half if i < count - 1 else 0)
        parts.append(f"[{i + 1}:a]aresample=48000:resampler=soxr,aformat=channel_layouts=stereo,atrim=end={whole},"
                     f"apad=whole_dur={whole},{shot['filter']},volume={shot['gain_db']}dB,atrim={start}:{end},"
                     f"asetpts=PTS-STARTPTS[s{i}]")
    joined = "s0"
    for i in range(1, count):
        parts.append(f"[{joined}][s{i}]acrossfade=d={CROSSFADE}:c1=qsin:c2=qsin[x{i}]")
        joined = f"x{i}"
    length = count * seconds
    parts.append(f"[{joined}]afade=t=in:d={EDGE_FADE},afade=t=out:st={length - EDGE_FADE}:d={EDGE_FADE}[mix]")
    return ";".join(parts)


def under_peak(cfg: Config, render, gain_db: float, out: Path) -> tuple[tuple[float, float | None], float]:
    """render(gain_db) writes `out`. Its true peak after AAC is measured; when over TRUE_PEAK the gain is lowered by
    the excess plus 0.2 dB and it is written once more. Returns ((LUFS, true peak), the gain used)."""
    render(gain_db)
    lufs, peak = measure(cfg, "-i", str(out))
    if peak is not None and peak > TRUE_PEAK:
        gain_db -= peak - TRUE_PEAK + 0.2
        render(gain_db)
        lufs, peak = measure(cfg, "-i", str(out))
        if peak is not None and peak > TRUE_PEAK:
            raise AssemblyError(f"true peak {peak:.1f} dBTP is still over {TRUE_PEAK} after a second encode")
    return (lufs, peak), gain_db


def with_sound(cfg: Config, film: Path, tracks: list[Path], out: Path, *, seconds: int) -> dict:
    """The silent film's picture with one track per shot (each HANDLE seconds longer at both ends), every shot at one
    level, crossfaded at the cuts, the film at TARGET_LUFS with its true peak under TRUE_PEAK, AAC 48 kHz.

    Returns the film's LUFS and true peak as encoded, and per shot its level as made and in the film, its gain and
    what was done to its stereo."""
    shots = [_shot_sound(cfg, track, seconds) for track in tracks]
    inputs = ["-i", str(film), *[arg for track in tracks for arg in ("-i", str(track))]]
    mix = _mix_graph(shots, seconds)
    first = ffmpeg.run(cfg, "-nostats", *inputs, "-filter_complex", f"{mix};[mix]ebur128=peak=true[m]",
                       "-map", "[m]", "-f", "null", "-", timeout=1800)
    _check(first, "measuring the sound")
    joined, _ = parse_ebur128_summary(first.stderr)

    def render(gain_db: float) -> None:
        _check(ffmpeg.run(cfg, "-y", "-v", "error", *inputs, "-filter_complex",
                          f"{mix};[mix]volume={gain_db:.2f}dB,{LIMITER}[out]", "-map", "0:v", "-map", "[out]",
                          "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
                          str(out), timeout=1800), "mixing the sound")

    (lufs, peak), gain = under_peak(cfg, render, TARGET_LUFS - joined if joined > QUIET_LUFS else 0.0, out)
    for i, shot in enumerate(shots):
        shot["lufs"], _ = measure(cfg, "-i", str(out), chain=f"atrim={i * seconds}:{(i + 1) * seconds}")
        del shot["filter"]
    return {"lufs": lufs, "true_peak": peak, "gain_db": round(gain, 2), "shots": shots}
