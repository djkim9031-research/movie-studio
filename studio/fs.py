"""Everything under ~/studio-data belongs to its owner alone.

A folder others can write lets them swap the ffmpeg that studio runs, edit the spend ledger, or plant files among
the models; this machine's default umask leaves new folders group-writable.
"""

import hashlib
import os
import shutil
import stat
from pathlib import Path

_DIR_NOFOLLOW = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


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


def copy_into(src: Path, folder: Path, name: str) -> Path:
    """Copies `src` to folder/name in a folder the container can also write: neither the folder itself nor an entry
    already at `name` is followed; that entry is removed and a new file created."""
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"bad file name: {name!r}")
    dir_fd = os.open(folder, _DIR_NOFOLLOW)
    try:
        try:
            os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
        with os.fdopen(fd, "wb") as out, open(src, "rb") as data:
            shutil.copyfileobj(data, out)
    finally:
        os.close(dir_fd)
    return Path(folder) / name


def safe_parts(parts: list[str]) -> bool:
    """Plain names only: no empty part, no "." or "..", no slash."""
    return bool(parts) and all(part and part not in (".", "..") and "/" not in part for part in parts)


def copy_out(base: Path, parts: list[str], dest: Path) -> Path:
    """Copies base/parts[0]/.../parts[-1] to `dest`. `base` is ours; the parts may come from the container, so each
    is opened without following links and the last must be a regular file."""
    if not safe_parts(parts):
        raise ValueError(f"unsafe path: {parts}")
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIR_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(file_fd, "rb") as data:
        if not stat.S_ISREG(os.fstat(data.fileno()).st_mode):
            raise OSError(f"not a regular file: {'/'.join(parts)}")
        with open(dest, "wb") as out:
            shutil.copyfileobj(data, out)
    return Path(dest)


def remove_in(base: Path, parts: list[str]) -> bool:
    """Deletes base/parts[0]/.../parts[-1] without following links on the way; a link at the end is removed itself.
    False when the entry is already gone."""
    if not safe_parts(parts):
        raise ValueError(f"unsafe path: {parts}")
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIR_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        os.unlink(parts[-1], dir_fd=fd)
        return True
    except FileNotFoundError:
        return False
    finally:
        os.close(fd)
