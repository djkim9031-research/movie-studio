import subprocess
import wave

import numpy as np
import pytest

from studio.config import load_config
from studio.media import assemble, ffmpeg
from studio.media.smoke import measure

CFG = load_config()
needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(CFG).exists(), reason="run: uv run studio fetch-ffmpeg")


def clip(path, color, frames=81, fps=16):
    ffmpeg.run(CFG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:size=320x176:rate={fps}",
               "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)).check_returncode()
    return path


def tone(path, seconds, frequency, gain_db):
    ffmpeg.run(CFG, "-y", "-v", "error", "-f", "lavfi", "-i",
               f"sine=frequency={frequency}:sample_rate=44100:duration={seconds}",
               "-af", f"volume={gain_db}dB,aformat=channel_layouts=stereo", str(path)).check_returncode()
    return path


def stream(path, kind, entry):
    out = subprocess.run([str(ffmpeg.binary(CFG, "ffprobe")), "-v", "error", "-select_streams", kind, "-count_frames",
                          "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def handled_tone(path, core_hz, handle_hz, seconds, gain_db=-20):
    """A shot's track as the worker makes it: the shot's tone with HANDLE seconds of another tone at each end."""
    h = assemble.HANDLE
    ffmpeg.run(CFG, "-y", "-v", "error",
               "-f", "lavfi", "-i", f"sine=frequency={handle_hz}:sample_rate=44100:duration={h}",
               "-f", "lavfi", "-i", f"sine=frequency={core_hz}:sample_rate=44100:duration={seconds}",
               "-f", "lavfi", "-i", f"sine=frequency={handle_hz}:sample_rate=44100:duration={h}",
               "-filter_complex", f"[0:a][1:a][2:a]concat=n=3:v=0:a=1,volume={gain_db}dB,"
                                  "aformat=channel_layouts=stereo", str(path)).check_returncode()
    return path


def write_wav(path, samples, rate=44100):
    """samples: float (n, 2) in [-1, 1], written as 16-bit stereo."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def decoded(path, rate=48000):
    out = subprocess.run([str(ffmpeg.binary(CFG)), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2",
                          "-ar", str(rate), "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype="<f4").reshape(-1, 2)


def tones(samples, start, end, candidates, rate=48000):
    """The candidate frequencies present between start and end s, within 26 dB of the strongest."""
    part = samples[int(start * rate):int(end * rate)].mean(axis=1)
    spectrum = np.abs(np.fft.rfft(part * np.hanning(len(part))))
    freqs = np.fft.rfftfreq(len(part), 1 / rate)
    level = {f: spectrum[(freqs > f - 8) & (freqs < f + 8)].max() for f in candidates}
    top = max(level.values())
    return {f for f, v in level.items() if v > top * 10 ** (-26 / 20)}


def color_at(path, seconds):
    out = subprocess.run([str(ffmpeg.binary(CFG)), "-v", "error", "-ss", str(seconds), "-i", str(path),
                          "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True)
    return "rgb"[max(range(3), key=lambda i: out.stdout[i])]


@needs_ffmpeg
def test_silent_film_cuts_each_clip_to_whole_seconds_and_keeps_the_shot_order(tmp_path):
    clips = [clip(tmp_path / f"shot{n}.mp4", color) for n, color in enumerate(("red", "green", "blue"), 1)]
    out = tmp_path / "film.mp4"
    assemble.silent_film(CFG, clips, out, seconds=5, fps=16)
    assert stream(out, "v:0", "nb_read_frames") == "240"
    assert ffmpeg.probe(CFG, out)["fps"] == 16 and ffmpeg.probe(CFG, out)["audio"] is None
    assert [color_at(out, t) for t in (2.5, 7.5, 12.5)] == ["r", "g", "b"]


@needs_ffmpeg
def test_silent_film_refuses_a_clip_shorter_than_the_shot(tmp_path):
    clips = [clip(tmp_path / "shot1.mp4", "red"), clip(tmp_path / "shot2.mp4", "green", frames=60)]
    with pytest.raises(assemble.AssemblyError, match="shot2.mp4"):
        assemble.silent_film(CFG, clips, tmp_path / "film.mp4", seconds=5, fps=16)


@needs_ffmpeg
def test_padded_clip_holds_the_shots_first_and_last_frames_for_the_handles(tmp_path):
    # Red then green for the shot's 80 frames, and a blue 81st frame that the film drops.
    src = tmp_path / "shot.mp4"
    ffmpeg.run(CFG, "-y", "-v", "error",
               *[arg for c, d in (("red", 2.5), ("green", 2.5), ("blue", 0.0625))
                 for arg in ("-f", "lavfi", "-i", f"color=c={c}:size=320x176:rate=16:duration={d}")],
               "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0", "-frames:v", "81", "-pix_fmt", "yuv420p",
               str(src)).check_returncode()
    padded = assemble.padded_clip(CFG, src, tmp_path / "padded.mp4", seconds=5, fps=16)
    assert stream(padded, "v:0", "nb_read_frames") == str(80 + 2 * int(assemble.HANDLE * 16))
    assert [color_at(padded, t) for t in (0.1, 0.6, 3.3, 5.9)] == ["r", "r", "g", "g"]


def test_stereo_fix_leaves_a_track_alone_unless_it_cancels_in_mono():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, 44100)) * 0.1
    assert assemble.stereo_fix(np.stack([x, 0.5 * x + 0.5 * y], axis=1)) == ("kept", 1.0)
    assert assemble.stereo_fix(np.zeros((44100, 2))) == ("kept", 1.0)
    # Partly out of phase: the side is turned down until left and right no longer cancel; the mid is untouched.
    partly = np.stack([x + 0.3 * y, -0.4 * x + y], axis=1)
    action, width = assemble.stereo_fix(partly)
    narrowed = assemble.apply_stereo(partly, action, width)
    assert action == "narrowed" and 0 < width < 1 and assemble.correlation(partly) < 0
    assert assemble.correlation(narrowed) > 0 and np.allclose(narrowed.sum(axis=1), partly.sum(axis=1))
    # Mostly out of phase: narrowing would leave almost nothing in mono, so the right channel is inverted.
    flipped = np.stack([x, -0.9 * x + 0.1 * y], axis=1)
    action, width = assemble.stereo_fix(flipped)
    assert action == "inverted" and assemble.correlation(assemble.apply_stereo(flipped, action, width)) > 0.5


@needs_ffmpeg
def test_with_sound_crossfades_at_each_cut_without_moving_any_sound(tmp_path):
    film = tmp_path / "film.mp4"
    assemble.silent_film(CFG, [clip(tmp_path / "a.mp4", "red"), clip(tmp_path / "b.mp4", "green")], film,
                         seconds=5, fps=16)
    # Shot sounds 300 and 500 Hz; their handles 900 and 1100 Hz may only be heard inside the crossfade.
    tracks = [handled_tone(tmp_path / "a.wav", 300, 900, 5), handled_tone(tmp_path / "b.wav", 500, 1100, 5)]
    out = tmp_path / "film-sound.mp4"
    assemble.with_sound(CFG, film, tracks, out, seconds=5)
    assert abs(ffmpeg.probe(CFG, out)["duration_s"] - 10) < 0.05 and stream(out, "v:0", "nb_read_frames") == "160"
    pcm, all4 = decoded(out), {300, 500, 900, 1100}
    half = assemble.CROSSFADE / 2
    assert tones(pcm, 0.3, 0.45, all4) == {300}  # no head handle before the first shot
    assert tones(pcm, 1.0, 5 - half - 0.05, all4) == {300}
    assert {300, 500} <= tones(pcm, 4.85, 5.15, all4)
    assert tones(pcm, 5 + half + 0.05, 9.5, all4) == {500}


@needs_ffmpeg
def test_with_sound_brings_every_shot_to_one_level_and_keeps_the_film_under_the_peak(tmp_path):
    film = tmp_path / "film.mp4"
    assemble.silent_film(CFG, [clip(tmp_path / f"{c}.mp4", c) for c in ("red", "green", "blue")], film,
                         seconds=5, fps=16)
    rng = np.random.default_rng(1)
    n = int((5 + 2 * assemble.HANDLE) * 44100)
    x, y = rng.standard_normal((2, n)) * 0.05
    # Loud and short, quiet and long (both fitted), and one that cancels in mono.
    tracks = [handled_tone(tmp_path / "loud.wav", 440, 440, 5.3, gain_db=-6),
              handled_tone(tmp_path / "quiet.wav", 660, 660, 4.5, gain_db=-38),
              write_wav(tmp_path / "phase.wav", np.stack([x + 0.3 * y, -0.4 * x + y], axis=1))]
    out = tmp_path / "film-sound.mp4"
    report = assemble.with_sound(CFG, film, tracks, out, seconds=5)
    assert ffmpeg.probe(CFG, out)["audio"] == "aac"
    assert stream(out, "a:0", "sample_rate") == "48000" and stream(out, "a:0", "channels") == "2"
    lufs, peak = measure(CFG, "-i", str(out))
    assert abs(lufs - assemble.TARGET_LUFS) <= 0.5 and peak <= assemble.TRUE_PEAK
    assert (report["lufs"], report["true_peak"]) == (lufs, peak)
    shots = [measure(CFG, "-i", str(out), chain=f"atrim={5 * i}:{5 * i + 5}")[0] for i in range(3)]
    assert max(shots) - min(shots) <= 1.0
    assert [s["stereo"] for s in report["shots"]] == ["kept", "kept", "narrowed"]
    assert report["shots"][2]["correlation_after"] > 0 > report["shots"][2]["correlation_before"]


def test_under_peak_encodes_once_more_with_the_gain_lowered_by_the_excess(tmp_path, monkeypatch):
    gains, measured = [], [(-15.9, -1.2), (-16.2, -1.7)]
    monkeypatch.setattr("studio.media.assemble.measure", lambda cfg, *args, **kw: measured.pop(0))
    assert assemble.under_peak(CFG, gains.append, 3.0, tmp_path / "out.mp4") == ((-16.2, -1.7), gains[1])
    assert gains[0] == 3.0 and gains[1] <= 3.0 - 0.3


def test_under_peak_fails_when_the_second_encode_is_still_over(tmp_path, monkeypatch):
    measured = [(-15.9, -1.2), (-16.0, -1.3)]
    monkeypatch.setattr("studio.media.assemble.measure", lambda cfg, *args, **kw: measured.pop(0))
    with pytest.raises(assemble.AssemblyError, match="true peak"):
        assemble.under_peak(CFG, lambda gain: None, 3.0, tmp_path / "out.mp4")
