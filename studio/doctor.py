"""Environment checks run by `studio doctor`; each returns pass or fail with one line of detail."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from studio.config import Config
from studio.gpu.preflight import run_preflight
from studio.media import ffmpeg

MIN_DRIVER = 580
ENCODERS = ("libx264", "aac")
NVENC = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
FILTERS = ("xfade", "loudnorm", "sidechaincompress", "ebur128", "subtitles")
# Frozen and deterministic: well over the 1,024-token prefix the cache check needs, never edited per call.
CACHE_PREFIX = "\n".join(f"Reference line {i:03d}: scenes run in order, shots run in order, frames run in order."
                         for i in range(160))


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def exit_code(checks: list[Check]) -> int:
    return 0 if all(c.ok for c in checks) else 1


def check_disk(path: Path, min_free_gb: float) -> Check:
    free_gb = shutil.disk_usage(path).free / 1e9
    return Check(f"disk {path}", free_gb >= min_free_gb, f"{free_gb:.0f} GB free, need {min_free_gb:.0f}")


def check_driver(version: str, minimum: int = MIN_DRIVER) -> Check:
    major = int(version.split(".")[0])
    return Check("nvidia driver", major >= minimum, f"{version}, need {minimum}+")


def check_gpu(cfg: Config) -> list[Check]:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=30).stdout.strip().splitlines()[0]
    verdict = run_preflight(cfg.gpu.agent_service, cfg.gpu.min_free_gib)
    detail = "; ".join(verdict.problems) or f"no compute processes, {verdict.memory.free_gib:.1f} GiB free"
    return [check_driver(driver), Check("gpu idle", verdict.ok, detail)]


def check_ffmpeg(cfg: Config) -> list[Check]:
    if not ffmpeg.binary(cfg).exists():
        return [Check("ffmpeg", False, f"missing {ffmpeg.binary(cfg)}; run: uv run studio fetch-ffmpeg")]
    checks = [Check("ffmpeg", "n8.1" in ffmpeg.version(cfg), ffmpeg.version(cfg))]
    found_encoders, found_filters = ffmpeg.encoders(cfg), ffmpeg.filters(cfg)
    missing = [e for e in ENCODERS if e not in found_encoders]
    checks.append(Check("ffmpeg encoders", not missing, "missing " + ", ".join(missing) if missing
                        else ", ".join(ENCODERS)))
    missing = [f for f in FILTERS if f not in found_filters]
    checks.append(Check("ffmpeg filters", not missing, "missing " + ", ".join(missing) if missing
                        else ", ".join(FILTERS)))
    for encoder in NVENC:
        ok, error = ffmpeg.try_encode(cfg, encoder)
        checks.append(Check(f"encode {encoder}", ok, "1 s test pattern" if ok else error))
    return checks


def check_claude_cache(claude) -> list[Check]:
    """Two identical calls; the second must read the cached prefix the first wrote."""
    request = {"max_tokens": 256,
               "system": [{"type": "text", "text": CACHE_PREFIX, "cache_control": {"type": "ephemeral"}}],
               "messages": [{"role": "user", "content": "Reply with the single word ok."}]}
    first = claude.create("doctor", film="_doctor", **request)
    second = claude.create("doctor", film="_doctor", **request)
    written = first.usage.cache_creation_input_tokens or 0
    read = second.usage.cache_read_input_tokens or 0
    return [Check("claude auth", True, f"{first.model}, request {getattr(first, '_request_id', None)}"),
            Check("claude cache", read > 0, f"first call wrote {written} tokens, second read {read}")]


def run_doctor(cfg: Config, with_claude: bool = False) -> list[Check]:
    checks = check_gpu(cfg)
    checks.append(check_disk(Path.home(), cfg.disk.home_min_free_gb))
    checks.append(check_disk(cfg.disk.docker_root, cfg.disk.docker_min_free_gb))
    checks += check_ffmpeg(cfg)
    if with_claude:
        from studio.claude.client import Claude
        checks += check_claude_cache(Claude(cfg))
    return checks
