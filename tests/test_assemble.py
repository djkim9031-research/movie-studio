import subprocess

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
def test_with_sound_fits_each_track_to_its_shot_at_the_studio_loudness(tmp_path):
    film = tmp_path / "film.mp4"
    assemble.silent_film(CFG, [clip(tmp_path / "a.mp4", "red"), clip(tmp_path / "b.mp4", "green")], film,
                         seconds=5, fps=16)
    # One track a little long, one short and quiet: both end up exactly 5 s.
    tracks = [tone(tmp_path / "a.wav", 5.3, 440, -20), tone(tmp_path / "b.wav", 4.0, 660, -35)]
    out = tmp_path / "film-sound.mp4"
    assemble.with_sound(CFG, film, tracks, out, seconds=5)
    assert ffmpeg.probe(CFG, out)["audio"] == "aac"
    assert stream(out, "a:0", "sample_rate") == "48000" and stream(out, "a:0", "channels") == "2"
    assert stream(out, "v:0", "nb_read_frames") == "160"
    lufs, peak = measure(CFG, "-i", str(out))
    assert abs(lufs - assemble.TARGET_LUFS) <= 0.5 and peak <= assemble.TRUE_PEAK + 0.5


def test_loudnorm_values_are_read_from_the_last_json_block():
    log = 'noise {"a": 1}\n[Parsed_loudnorm_1 @ 0x1]\n{\n\t"input_i" : "-31.20",\n\t"input_tp" : "-12.00",\n' \
          '\t"input_lra" : "1.50",\n\t"input_thresh" : "-41.40",\n\t"target_offset" : "0.10"\n}\n'
    assert assemble.parse_loudnorm(log) == {"input_i": -31.2, "input_tp": -12.0, "input_lra": 1.5,
                                            "input_thresh": -41.4, "target_offset": 0.1}
