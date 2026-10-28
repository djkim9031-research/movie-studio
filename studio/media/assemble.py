"""A film from its shots: clips joined with hard cuts, then its sound in layers. One ambience bed runs under every
cut; each shot that has its own sound gets a Foley track made FOLEY_HANDLE seconds longer at both ends and placed on
the shot with short fades; tracks are kept from cancelling in mono; the film is brought to TARGET_LUFS as one
programme, with its true peak under TRUE_PEAK after AAC."""

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
HANDLE = 0.5  # seconds a padded clip holds its first and last frames by default
FOLEY_HANDLE = 1.5  # seconds of sound made past each end of a Foley shot: PrismAudio hears 8 s for a 5 s shot
FOLEY_IN, FOLEY_OUT = 0.1, 0.3  # a Foley track's fade in before its shot and its tail into the next
BED_LUFS = -26.0  # the bed, and each Foley track six LU over it, before the film's one gain
FOLEY_LUFS = -20.0
BED_SEGMENT, BED_CROSSFADE = 10, 1.0  # seconds: a generated bed is made in segments joined by crossfades
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


def padded_clip(cfg: Config, clip: Path, out: Path, *, seconds: int, fps: int, handle: float = HANDLE) -> Path:
    """The shot as the film shows it (the clip's first `seconds`) with its first and last frames held for `handle`
    seconds: the picture the sound worker makes the handles from."""
    graph = (f"trim=end_frame={seconds * fps},setpts=PTS-STARTPTS,"
             f"tpad=start_mode=clone:start_duration={handle}:stop_mode=clone:stop_duration={handle}")
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


def still_clip(cfg: Config, picture: Path, out: Path, *, seconds: int, fps: int) -> Path:
    """`picture` held for `seconds`: the picture a bed is made from."""
    _check(ffmpeg.run(cfg, "-y", "-v", "error", "-loop", "1", "-i", str(picture), "-t", str(seconds), "-r", str(fps),
                      "-vf", "scale=1280:-2,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                      str(out), timeout=600), "making the bed's picture")
    return out


def join_bed(cfg: Config, segments: list[Path], out: Path) -> Path:
    """Bed segments joined by BED_CROSSFADE crossfades, as 16-bit 44.1 kHz stereo."""
    inputs = [arg for seg in segments for arg in ("-i", str(seg))]
    if len(segments) == 1:
        graph = "[0:a]anull[bed]"
    else:
        parts, joined = [], "0:a"
        for i in range(1, len(segments)):
            parts.append(f"[{joined}][{i}:a]acrossfade=d={BED_CROSSFADE}:c1=qsin:c2=qsin[j{i}]")
            joined = f"j{i}"
        graph = ";".join(parts) + f";[{joined}]anull[bed]"
    _check(ffmpeg.run(cfg, "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[bed]", "-ar", "44100",
                      "-ac", "2", "-c:a", "pcm_s16le", str(out), timeout=600), "joining the bed")
    return out


def bed_from_file(cfg: Config, source: Path, out: Path, *, seconds: float) -> Path:
    """A recording looped to at least `seconds`, as 16-bit 44.1 kHz stereo."""
    _check(ffmpeg.run(cfg, "-y", "-v", "error", "-stream_loop", "-1", "-i", str(source), "-t", str(seconds),
                      "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out), timeout=600),
           "reading the bed recording")
    return out


def _layer(cfg: Config, track: Path, start: float, end: float, target: float) -> dict:
    """The stereo fix and the gain that bring the part of `track` between start and end s to `target` LUFS."""
    rate, samples = read_wav(track)
    part = samples[int(start * rate):int(end * rate)]
    action, width = stereo_fix(part)
    stereo = _stereo_filter(action, width)
    made, _ = measure(cfg, "-i", str(track), chain=f"atrim={start}:{end},{stereo}")
    return {"lufs_made": made, "gain_db": round(target - made, 2) if made > QUIET_LUFS else 0.0, "stereo": action,
            "width": width, "correlation_before": round(correlation(part), 3),
            "correlation_after": round(correlation(apply_stereo(part, action, width)), 3), "filter": stereo}


def _layers_graph(bed: dict, foley: list[tuple[int, dict]], shots: int, seconds: int) -> str:
    """The bed fitted to the film; each Foley track's own part placed on its shot, with FOLEY_IN before it and a
    FOLEY_OUT tail after it; all summed; the film's ends faded."""
    length = shots * seconds
    fit = "aresample=48000:resampler=soxr,aformat=channel_layouts=stereo"
    parts = [f"[1:a]{fit},atrim=end={length},apad=whole_dur={length},{bed['filter']},volume={bed['gain_db']}dB,"
             f"asetpts=PTS-STARTPTS[bed]"]
    for i, (n, info) in enumerate(foley, start=2):
        pre = min(FOLEY_IN, (n - 1) * seconds)
        start, end = FOLEY_HANDLE - pre, FOLEY_HANDLE + seconds + FOLEY_OUT
        parts.append(f"[{i}:a]{fit},{info['filter']},volume={info['gain_db']}dB,atrim={start}:{end},"
                     f"asetpts=PTS-STARTPTS,afade=t=in:d={max(pre, 0.01)},"
                     f"afade=t=out:st={end - start - FOLEY_OUT}:d={FOLEY_OUT},"
                     f"adelay={round(((n - 1) * seconds - pre) * 1000)}:all=1[f{n}]")
    labels = "[bed]" + "".join(f"[f{n}]" for n, _ in foley)
    parts.append(f"{labels}amix=inputs={1 + len(foley)}:normalize=0:duration=first,atrim=end={length},"
                 f"afade=t=in:d={EDGE_FADE},afade=t=out:st={length - EDGE_FADE}:d={EDGE_FADE}[mix]")
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


def with_layers(cfg: Config, film: Path, bed: Path, foley: dict[int, Path], out: Path, *, shots: int,
                seconds: int) -> dict:
    """The silent film's picture with `bed` under every cut and each `foley` track (by shot number, FOLEY_HANDLE
    seconds longer at both ends) on its shot; the bed at BED_LUFS and each Foley at FOLEY_LUFS before one gain brings
    the whole film to TARGET_LUFS, true peak under TRUE_PEAK, AAC 48 kHz.

    Returns the film's LUFS and true peak as encoded, the bed's level and stereo fix, and per shot whether it has
    Foley, its level in the film, and its Foley track's level and stereo fix."""
    order = sorted(foley)
    bed_layer = _layer(cfg, bed, 0, shots * seconds, BED_LUFS)
    layers = {n: _layer(cfg, foley[n], FOLEY_HANDLE, FOLEY_HANDLE + seconds, FOLEY_LUFS) for n in order}
    inputs = ["-i", str(film), "-i", str(bed), *[arg for n in order for arg in ("-i", str(foley[n]))]]
    mix = _layers_graph(bed_layer, [(n, layers[n]) for n in order], shots, seconds)
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
    report = []
    for n in range(1, shots + 1):
        level, _ = measure(cfg, "-i", str(out), chain=f"atrim={(n - 1) * seconds}:{n * seconds}")
        layer = {k: v for k, v in layers.get(n, {}).items() if k != "filter"}
        report.append({"foley": n in layers, "lufs": level, **layer})
    del bed_layer["filter"]
    return {"lufs": lufs, "true_peak": peak, "gain_db": round(gain, 2), "bed": bed_layer, "shots": report}
