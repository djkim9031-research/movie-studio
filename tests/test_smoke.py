import json
import subprocess

import pytest

from studio.config import load_config
from studio.media import ffmpeg
from studio.media.smoke import TARGET_LUFS, smoke_mux

CFG = load_config()
needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(CFG).exists(), reason="run: uv run studio fetch-ffmpeg")


@needs_ffmpeg
def test_smoke_mux_lands_on_the_target_loudness_with_aac_48k(tmp_path):
    out = tmp_path / "smoke.mp4"
    integrated, true_peak = smoke_mux(CFG, out)
    assert abs(integrated - TARGET_LUFS) <= 0.5
    assert true_peak is not None and true_peak < -1.0
    probe = subprocess.run([str(ffmpeg.binary(CFG, "ffprobe")), "-v", "error", "-show_streams", "-of", "json",
                            str(out)], capture_output=True, text=True, check=True)
    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert streams["video"]["codec_name"] == "h264"
    assert streams["audio"]["codec_name"] == "aac" and streams["audio"]["sample_rate"] == "48000"


@needs_ffmpeg
def test_probe_reports_duration_size_rate_and_audio(tmp_path):
    out = tmp_path / "smoke.mp4"
    smoke_mux(CFG, out)
    info = ffmpeg.probe(CFG, out)
    assert abs(info["duration_s"] - 10.0) < 0.1
    assert (info["width"], info["height"], info["fps"]) == (1280, 720, 24.0)
    assert info["audio"] == "aac"
