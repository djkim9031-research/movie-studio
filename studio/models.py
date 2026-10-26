"""Pinned model weights: downloaded at a fixed commit into ComfyUI's model folders, then checked by size and sha256."""

import shutil

from studio.config import Config
from studio.fs import private_dir, sha256_file
from studio.secrets import read_secret

# Passed on every download so an HF_ENDPOINT in the environment cannot send the token elsewhere.
HF_URL = "https://huggingface.co"


class DiskFloor(RuntimeError):
    """A download would leave less free disk than config disk.home_min_free_gb."""


def hf_hub_download(**kwargs) -> str:
    # Imported on use: only `studio models fetch` needs the Hugging Face client.
    from huggingface_hub import hf_hub_download as download
    return download(**kwargs)


def present(cfg: Config, name: str) -> list[str]:
    """Like verify, by name and size only: a quick check before a run; verify proves the content."""
    problems = []
    for f in cfg.models[name].files:
        path = cfg.model_root(name) / f.path
        if not path.exists():
            problems.append(f"{f.path}: missing")
        elif path.stat().st_size != f.size:
            problems.append(f"{f.path}: size {path.stat().st_size}, expected {f.size}")
    return problems


def verify(cfg: Config, name: str) -> list[str]:
    """One line per file that is missing, the wrong size, or has the wrong sha256; empty when all match."""
    problems = []
    for f in cfg.models[name].files:
        path = cfg.model_root(name) / f.path
        if not path.exists():
            problems.append(f"{f.path}: missing")
            continue
        size = path.stat().st_size
        if size != f.size:
            problems.append(f"{f.path}: size {size}, expected {f.size}")
            continue
        digest = sha256_file(path)
        if digest != f.sha256:
            problems.append(f"{f.path}: sha256 {digest}, expected {f.sha256}")
    return problems


def fetch(cfg: Config, name: str) -> list[str]:
    pinned = cfg.models[name]
    root = private_dir(cfg.model_root(name))
    needed = sum(f.size for f in pinned.files
                 if not ((root / f.path).exists() and (root / f.path).stat().st_size == f.size))
    left = shutil.disk_usage(root).free - needed
    if left < cfg.disk.home_min_free_gb * 1e9:
        raise DiskFloor(f"{name} needs {needed / 1e9:.0f} GB, which would leave {left / 1e9:.0f} GB free, under the "
                        f"{cfg.disk.home_min_free_gb:.0f} GB floor")
    token = read_secret(cfg.hf_env, "HF_TOKEN")
    for f in pinned.files:
        hf_hub_download(repo_id=pinned.repo, filename=f.path, revision=pinned.revision, local_dir=root,
                        token=token, endpoint=HF_URL)
    return verify(cfg, name)
