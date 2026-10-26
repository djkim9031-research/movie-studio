"""Pinned model weights: downloaded at a fixed commit into ComfyUI's model folders, then checked by size and sha256."""

from studio.config import Config
from studio.fs import private_dir, sha256_file
from studio.secrets import read_secret

# Passed on every download so an HF_ENDPOINT in the environment cannot send the token elsewhere.
HF_URL = "https://huggingface.co"


def hf_hub_download(**kwargs) -> str:
    # Imported on use: only `studio models fetch` needs the Hugging Face client.
    from huggingface_hub import hf_hub_download as download
    return download(**kwargs)


def verify(cfg: Config, name: str) -> list[str]:
    """One line per file that is missing, the wrong size, or has the wrong sha256; empty when all match."""
    problems = []
    for f in cfg.models[name].files:
        path = cfg.models_dir / f.path
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
    token = read_secret(cfg.hf_env, "HF_TOKEN")
    private_dir(cfg.models_dir)
    for f in pinned.files:
        hf_hub_download(repo_id=pinned.repo, filename=f.path, revision=pinned.revision, local_dir=cfg.models_dir,
                        token=token, endpoint=HF_URL)
    return verify(cfg, name)
