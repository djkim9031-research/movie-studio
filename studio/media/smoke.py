"""10 s test pattern with a tone at the studio's loudness target: proves picture, sound and the meter work."""

from pathlib import Path

from studio.config import Config
from studio.fs import private_dir
from studio.media import ffmpeg
from studio.media.ffmpeg import parse_ebur128_summary

TARGET_LUFS = -16.0
TONE = "sine=frequency=440:sample_rate=48000:duration=10,aformat=channel_layouts=stereo"
PATTERN = "testsrc2=size=1280x720:rate=24:duration=10"


def measure(cfg: Config, *inputs: str, chain: str = "") -> tuple[float, float | None]:
    """(integrated LUFS, true peak dBFS) of the input, after the filter `chain` if one is given."""
    meter = f"{chain},ebur128=peak=true" if chain else "ebur128=peak=true"
    result = ffmpeg.run(cfg, "-nostats", *inputs, "-af", meter, "-f", "null", "-")
    result.check_returncode()
    return parse_ebur128_summary(result.stderr)


def smoke_mux(cfg: Config, out: Path) -> tuple[float, float | None]:
    """Writes `out` with the tone gain set from a first measurement; returns its (LUFS, true peak dBFS)."""
    raw, _ = measure(cfg, "-f", "lavfi", "-i", TONE)
    gain_db = TARGET_LUFS - raw
    private_dir(out.parent)
    ffmpeg.run(cfg, "-y", "-v", "error",
               "-f", "lavfi", "-i", PATTERN,
               "-f", "lavfi", "-i", f"{TONE},volume={gain_db:.2f}dB",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", str(out)).check_returncode()
    return measure(cfg, "-i", str(out))
