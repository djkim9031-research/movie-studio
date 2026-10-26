"""The ComfyUI container: built from docker/comfyui, started only for GPU work, reachable only from this machine."""

import os
import subprocess
import time
from pathlib import Path

from studio.config import Config
from studio.fs import private_dir
from studio.gpu.comfy import ComfyClient
from studio.gpu.preflight import parse_compute_apps, run_preflight

CONTAINER = "studio-comfyui"
IO_SUBDIRS = ("input", "output", "temp", "user", "home")
DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker" / "comfyui"


def run_args(cfg: Config, uid: int, gid: int) -> list[str]:
    c = cfg.comfyui
    return ["docker", "run", "--detach", "--rm", "--name", CONTAINER,
            "--gpus", "all", "--user", f"{uid}:{gid}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--shm-size", f"{c.shm_gb}g",
            "--env", "HOME=/io/home", "--env", "HF_HUB_OFFLINE=1", "--env", "USER=studio",
            "--volume", f"{cfg.models_dir}:/models:ro",
            "--volume", f"{cfg.comfy_io_dir}:/io",
            "--publish", f"127.0.0.1:{c.port}:8188",
            c.image]


def parse_top_pids(text: str) -> set[int]:
    """PIDs from `docker top <container> -eo pid`: host PIDs, as nvidia-smi reports them."""
    return {int(token) for token in text.split() if token.isdigit()}


def container_pids() -> set[int]:
    result = subprocess.run(["docker", "top", CONTAINER, "-eo", "pid"], capture_output=True, text=True)
    return parse_top_pids(result.stdout) if result.returncode == 0 else set()


def is_running() -> bool:
    result = subprocess.run(["docker", "ps", "--quiet", "--filter", f"name=^{CONTAINER}$"],
                            capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def client(cfg: Config) -> ComfyClient:
    return ComfyClient(f"http://127.0.0.1:{cfg.comfyui.port}")


def build(cfg: Config) -> None:
    c = cfg.comfyui
    missing = [name for name in ("base_image", "commit", "torch") if not getattr(c, name)]
    if missing:
        raise RuntimeError(f"pin comfyui.{', comfyui.'.join(missing)} in config/studio.yaml before building")
    subprocess.run(["docker", "build", "--tag", c.image,
                    "--build-arg", f"BASE_IMAGE={c.base_image}",
                    "--build-arg", f"COMFY_COMMIT={c.commit}",
                    "--build-arg", f"TORCH_VERSION={c.torch}",
                    str(DOCKER_DIR)], check=True)


def up(cfg: Config, ready_timeout_s: float = 180) -> None:
    if is_running():
        return
    verdict = run_preflight(cfg.gpu.agent_service, cfg.gpu.min_free_gib)
    if not verdict.ok:
        raise RuntimeError("GPU not free: " + "; ".join(verdict.problems))
    for name in IO_SUBDIRS:
        private_dir(cfg.comfy_io_dir / name)
    subprocess.run(run_args(cfg, os.getuid(), os.getgid()), check=True, capture_output=True, text=True)
    client(cfg).wait_ready(ready_timeout_s)


def down(cfg: Config, timeout_s: float = 60) -> None:
    """Stops the container and waits until none of its processes hold the GPU."""
    pids = container_pids()
    subprocess.run(["docker", "stop", CONTAINER], capture_output=True, text=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        if not pids & {p.pid for p in parse_compute_apps(apps)}:
            return
        time.sleep(1)
    raise TimeoutError(f"container processes still on the GPU after {timeout_s:.0f} s")
