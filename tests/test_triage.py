import subprocess

import numpy as np
import pytest

from studio.config import load_config
from studio.media import ffmpeg, triage

CFG = load_config()
needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(CFG).exists(), reason="run: uv run studio fetch-ffmpeg")


def encode(frames: np.ndarray, out, fps=16):
    h, w = frames.shape[1:3]
    subprocess.run([str(ffmpeg.binary(CFG)), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt",
                    "yuv420p", str(out)], input=frames.tobytes(), check=True)
    return out


@needs_ffmpeg
def test_triage_measures_motion_frozen_ends_and_color_drift(tmp_path):
    fps, frames = 16, []
    for i in range(80):  # 0.5 s frozen, 4 s of a square crossing the frame while the picture warms, 0.5 s frozen
        t = min(max(i - 8, 0), 63)
        frame = np.full((180, 320, 3), 90, dtype=np.uint8)
        frame[..., 0] = 90 + t  # red rises: the picture drifts warm
        x = 10 + 3 * t
        frame[40:140, x:x + 100] = 230
        frames.append(frame)
    clip = encode(np.stack(frames), tmp_path / "take.mp4", fps)
    found = triage.clip(CFG, clip, fps)
    assert found["still_head_s"] == pytest.approx(0.5, abs=0.13)
    assert found["still_tail_s"] == pytest.approx(0.5, abs=0.13)
    assert found["motion"] > 0.5
    assert found["color_drift"] > 10

    still = encode(np.full((80, 180, 320, 3), 90, dtype=np.uint8), tmp_path / "still.mp4", fps)
    frozen = triage.clip(CFG, still, fps)
    assert frozen["motion"] < 0.05 and frozen["color_drift"] < 1 and frozen["still_head_s"] == 5.0
