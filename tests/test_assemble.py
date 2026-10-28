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


def handled_tone(path, core_hz, handle_hz, seconds, gain_db=-20, h=assemble.HANDLE):
    """A shot's track as the worker makes it: the shot's tone with `h` seconds of another tone at each end."""
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


def level(samples, start, end, frequency, rate=48000):
    """The level of one frequency between start and end s, in dB."""
    part = samples[int(start * rate):int(end * rate)].mean(axis=1)
    spectrum = np.abs(np.fft.rfft(part * np.hanning(len(part))))
    freqs = np.fft.rfftfreq(len(part), 1 / rate)
    return 20 * np.log10(spectrum[(freqs > frequency - 8) & (freqs < frequency + 8)].max())


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
def test_with_layers_runs_one_bed_under_every_cut_and_puts_each_foley_on_its_own_shot(tmp_path):
    film = tmp_path / "film.mp4"
    assemble.silent_film(CFG, [clip(tmp_path / f"{c}.mp4", c) for c in ("red", "green", "blue")], film,
                         seconds=5, fps=16)
    bed = tone(tmp_path / "bed.wav", 20, 300, -30)  # longer than the film, cut to it
    # Shot 2's own sound at 900 Hz; its handles at 1500 Hz may only be heard inside the short fades.
    foley = handled_tone(tmp_path / "shot2.wav", 900, 1500, 5, h=assemble.FOLEY_HANDLE)
    out = tmp_path / "film-sound.mp4"
    report = assemble.with_layers(CFG, film, bed, {2: foley}, out, shots=3, seconds=5)
    assert abs(ffmpeg.probe(CFG, out)["duration_s"] - 15) < 0.05 and stream(out, "v:0", "nb_read_frames") == "240"
    pcm, three = decoded(out), {300, 900, 1500}
    assert tones(pcm, 0.5, 4.8, three) == {300}
    assert tones(pcm, 5.3, 9.6, three) == {300, 900}
    assert tones(pcm, 10.5, 14.5, three) == {300}
    for cut in (5, 10):  # the bed's tone does not dip at a cut
        assert abs(level(pcm, cut - 0.1, cut + 0.1, 300) - level(pcm, cut - 2.6, cut - 2.4, 300)) < 1.5
    assert report["shots"][1]["foley"] and not report["shots"][0]["foley"] and report["bed"]["stereo"] == "kept"


@needs_ffmpeg
def test_with_layers_levels_the_film_as_one_programme_and_keeps_stereo_safe(tmp_path):
    film = tmp_path / "film.mp4"
    assemble.silent_film(CFG, [clip(tmp_path / f"{c}.mp4", c) for c in ("red", "green", "blue")], film,
                         seconds=5, fps=16)
    rng = np.random.default_rng(1)
    n = int((5 + 2 * assemble.FOLEY_HANDLE) * 44100)
    x, y = rng.standard_normal((2, n)) * 0.05
    bed = tone(tmp_path / "bed.wav", 20, 300, -45)
    tracks = {1: write_wav(tmp_path / "phase.wav", np.stack([x + 0.3 * y, -0.4 * x + y], axis=1)),
              3: handled_tone(tmp_path / "loud.wav", 660, 660, 5, gain_db=-4, h=assemble.FOLEY_HANDLE)}
    out = tmp_path / "film-sound.mp4"
    report = assemble.with_layers(CFG, film, bed, tracks, out, shots=3, seconds=5)
    assert ffmpeg.probe(CFG, out)["audio"] == "aac"
    assert stream(out, "a:0", "sample_rate") == "48000" and stream(out, "a:0", "channels") == "2"
    lufs, peak = measure(CFG, "-i", str(out))
    assert abs(lufs - assemble.TARGET_LUFS) <= 0.5 and peak <= assemble.TRUE_PEAK
    assert (report["lufs"], report["true_peak"]) == (lufs, peak)
    shots = [s["lufs"] for s in report["shots"]]
    assert shots[1] < min(shots[0], shots[2]) - 3  # the bed-only shot stays under the shots with their own sound
    assert report["shots"][0]["stereo"] == "narrowed"
    assert report["shots"][0]["correlation_after"] > 0 > report["shots"][0]["correlation_before"]


@needs_ffmpeg
def test_a_short_recording_is_looped_with_crossfades_into_a_bed_as_long_as_the_film(tmp_path):
    # Noise, like a real room: a loop joins the recording's end to its start, which do not repeat each other.
    noise = np.random.default_rng(2).standard_normal((3 * 44100, 2)) * 0.05
    recording = write_wav(tmp_path / "rec.wav", noise)
    bed = assemble.bed_from_file(CFG, recording, tmp_path / "bed.wav", seconds=15)
    assert ffmpeg.probe(CFG, bed)["duration_s"] >= 15
    pcm = decoded(bed)
    rms = [20 * np.log10(np.sqrt((pcm[int(t * 48000):int((t + 0.2) * 48000)] ** 2).mean()))
           for t in np.arange(0.2, 14.6, 0.1)]
    assert max(rms) - min(rms) < 1.5  # no gap or dip where the loop joins
    with pytest.raises(assemble.AssemblyError, match="shorter than 2 s"):
        assemble.bed_from_file(CFG, tone(tmp_path / "short.wav", 1.5, 300, -20), tmp_path / "b.wav", seconds=15)


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
