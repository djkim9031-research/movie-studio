"""Pinned static FFmpeg (BtbN linux64-gpl n8.1) and probes of what it can do.

n8.1 uses NVENC SDK 13.0 (driver 570+); n9.0 and master builds need driver 610+ for NVENC.
"""

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from studio.config import Config
from studio.fs import private_dir, sha256_file

RELEASES = "https://github.com/BtbN/FFmpeg-Builds/releases/download"
BINARIES = ("ffmpeg", "ffprobe")
DOWNLOAD_TIMEOUT_S = 60


class ChecksumMismatch(RuntimeError):
    pass


def parse_encoders(text: str) -> set[str]:
    """Names from `ffmpeg -encoders`: the second column of every row after the dashed rule."""
    names, past_rule = set(), False
    for line in text.splitlines():
        if line.strip().startswith("------"):
            past_rule = True
            continue
        parts = line.split()
        if past_rule and len(parts) >= 2:
            names.add(parts[1])
    return names


def parse_filters(text: str) -> set[str]:
    """Names from `ffmpeg -filters`: rows whose third column is an io spec such as AA->A."""
    return {parts[1] for parts in map(str.split, text.splitlines()) if len(parts) >= 3 and "->" in parts[2]}


def parse_ebur128_summary(text: str) -> tuple[float, float | None]:
    """(integrated LUFS, true peak dBFS or None) from the last ebur128 summary in ffmpeg's log."""
    summary = text[text.rfind("Summary:"):]
    integrated = re.search(r"^\s*I:\s*(-?[\d.]+|-inf) LUFS", summary, re.M)
    if not integrated:
        raise ValueError("no integrated loudness in ebur128 output")
    peak = re.search(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-inf) dBFS", summary)
    return float(integrated.group(1)), float(peak.group(1)) if peak else None


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumMismatch(f"{path}: sha256 {actual} differs from the pinned {expected}")


def extract_binaries(tarball: Path, dest: Path) -> list[Path]:
    """Copies <top>/bin/ffmpeg and <top>/bin/ffprobe out of the release; nothing else is extracted.

    Each binary is written to a temp file and renamed over the old one: a running ffmpeg keeps its file, a crash
    leaves the old pair intact, and a symlink planted at the destination is replaced rather than written through.
    """
    dest = private_dir(dest)
    with tarfile.open(tarball, "r:*") as tar:
        found = {}
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if (member.isfile() and not path.is_absolute() and ".." not in path.parts and len(path.parts) == 3
                    and path.parts[1] == "bin" and path.name in BINARIES):
                found[path.name] = member
        missing = sorted(set(BINARIES) - found.keys())
        if missing:
            raise RuntimeError(f"{tarball} has no {', '.join(missing)}")
        staged = {}
        try:
            for name, member in found.items():
                fd, temp = tempfile.mkstemp(dir=dest, prefix=f".{name}-")
                staged[name] = Path(temp)
                with os.fdopen(fd, "wb") as out, tar.extractfile(member) as src:
                    shutil.copyfileobj(src, out)
                os.chmod(temp, 0o755)
            for name, temp in staged.items():
                os.replace(temp, dest / name)
        finally:
            for temp in staged.values():
                temp.unlink(missing_ok=True)
    return [dest / name for name in BINARIES]


def install(cfg: Config) -> list[Path]:
    pin = cfg.ffmpeg
    if pin is None:
        raise RuntimeError("no ffmpeg pin in config/studio.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / pin.asset
        url = f"{RELEASES}/{pin.release_tag}/{pin.asset}"
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, open(tarball, "wb") as out:
            shutil.copyfileobj(response, out)
        verify_sha256(tarball, pin.sha256)
        return extract_binaries(tarball, cfg.bin_dir)


def binary(cfg: Config, name: str = "ffmpeg") -> Path:
    return cfg.bin_dir / name


def run(cfg: Config, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary(cfg)), "-hide_banner", *args], capture_output=True, text=True,
                          timeout=timeout)


def version(cfg: Config) -> str:
    return run(cfg, "-version").stdout.splitlines()[0]


def encoders(cfg: Config) -> set[str]:
    return parse_encoders(run(cfg, "-encoders").stdout)


def filters(cfg: Config) -> set[str]:
    return parse_filters(run(cfg, "-filters").stdout)


def try_encode(cfg: Config, encoder: str) -> tuple[bool, str]:
    """Encodes 1 s of a test pattern with `encoder` and discards it; (ok, last error line)."""
    result = run(cfg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24:duration=1",
                 "-c:v", encoder, "-f", "null", "-")
    lines = result.stderr.strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else ""
