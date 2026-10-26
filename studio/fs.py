"""Everything under ~/studio-data belongs to its owner alone.

A folder others can write lets them swap the ffmpeg that studio runs, edit the spend ledger, or plant files among
the models; this machine's default umask leaves new folders group-writable.
"""

import hashlib
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_dir(path: Path) -> Path:
    """Creates the missing levels of `path` with mode 0700; existing folders are left as they are."""
    path = Path(path)
    missing = []
    level = path
    while not level.exists():
        missing.append(level)
        level = level.parent
    for folder in reversed(missing):
        folder.mkdir(mode=0o700)
        os.chmod(folder, 0o700)
    return path


def append_private(path: Path, text: str) -> None:
    """Appends to `path`, creating it with mode 0600."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(text)
