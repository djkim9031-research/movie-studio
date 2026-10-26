"""The ComfyUI container: built from docker/comfyui, run only for GPU work, on an internal network with no way out.

The host reaches ComfyUI at the container's address on that network; nothing is published on a host port, and the
container has no route to the internet and no DNS. `up` proves that from inside before anything is rendered.
"""

import hashlib
import json
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
EGRESS_PROBE = """
import json, socket
def tcp(host, port):
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False
def dns(name):
    try:
        socket.getaddrinfo(name, 443)
        return True
    except OSError:
        return False
print(json.dumps({"1.1.1.1:443": tcp("1.1.1.1", 443), "8.8.8.8:53": tcp("8.8.8.8", 53),
                  "dns pypi.org": dns("pypi.org")}))
"""


def image_tag(cfg: Config, docker_dir: Path = DOCKER_DIR) -> str:
    """Changes whenever a build input changes, so an image built from older pins is never run under newer ones."""
    c = cfg.comfyui
    digest = hashlib.sha256()
    for pin in (c.base_image or "", c.commit or "", c.torch or ""):
        digest.update(pin.encode() + b"\0")
    for path in sorted(p for p in Path(docker_dir).iterdir() if p.is_file()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return f"{c.image_repo}:{(c.commit or 'unpinned')[:12]}-{digest.hexdigest()[:10]}"


def run_args(cfg: Config, image: str, uid: int, gid: int) -> list[str]:
    c = cfg.comfyui
    return ["docker", "run", "--detach", "--init", "--pull", "never", "--name", CONTAINER,
            "--gpus", "all", "--user", f"{uid}:{gid}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--network", c.network, "--ip", c.ip, "--dns", "127.0.0.1",
            "--shm-size", f"{c.shm_gb}g",
            "--env", "HOME=/io/home", "--env", "HF_HUB_OFFLINE=1", "--env", "USER=studio",
            "--mount", f"type=bind,src={cfg.models_dir},dst=/models,readonly",
            "--mount", f"type=bind,src={cfg.comfy_io_dir},dst=/io",
            image]


def network_create_args(cfg: Config) -> list[str]:
    c = cfg.comfyui
    return ["docker", "network", "create", "--driver", "bridge", "--internal", "--subnet", c.subnet,
            "--gateway", c.gateway, c.network]


def network_problems(inspect: dict, cfg: Config) -> list[str]:
    c = cfg.comfyui
    problems = []
    if not inspect.get("Internal"):
        problems.append(f"network {c.network} is not internal, so the container could reach the internet")
    subnets = [entry.get("Subnet") for entry in (inspect.get("IPAM") or {}).get("Config") or []]
    if c.subnet not in subnets:
        problems.append(f"network {c.network} has subnet {subnets}, expected {c.subnet}")
    return problems


def egress_problems(result: dict) -> list[str]:
    return [f"the container reached {target}" for target, reached in result.items() if reached]


def _docker(*args: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, input=stdin)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {(result.stderr or result.stdout).strip()}")
    return result


def parse_top_pids(text: str) -> set[int]:
    """PIDs from `docker top <container> -eo pid`: host PIDs, as nvidia-smi reports them."""
    return {int(token) for token in text.split() if token.isdigit()}


def container_pids() -> set[int]:
    result = _docker("top", CONTAINER, "-eo", "pid", check=False)
    return parse_top_pids(result.stdout) if result.returncode == 0 else set()


def is_running() -> bool:
    return bool(_docker("ps", "--quiet", "--filter", f"name=^{CONTAINER}$").stdout.strip())


def client(cfg: Config) -> ComfyClient:
    return ComfyClient(cfg.comfyui.base_url)


def ensure_network(cfg: Config) -> None:
    found = _docker("network", "inspect", cfg.comfyui.network, check=False)
    if found.returncode != 0:
        _docker(*network_create_args(cfg)[1:])
        return
    problems = network_problems(json.loads(found.stdout)[0], cfg)
    if problems:
        raise RuntimeError("; ".join(problems))


def build(cfg: Config) -> str:
    c = cfg.comfyui
    missing = [name for name in ("base_image", "commit", "torch") if not getattr(c, name)]
    if missing:
        raise RuntimeError(f"pin comfyui.{', comfyui.'.join(missing)} in config/studio.yaml before building")
    tag = image_tag(cfg)
    subprocess.run(["docker", "build", "--tag", tag,
                    "--build-arg", f"BASE_IMAGE={c.base_image}",
                    "--build-arg", f"COMFY_COMMIT={c.commit}",
                    "--build-arg", f"TORCH_VERSION={c.torch}",
                    str(DOCKER_DIR)], check=True)
    return tag


def check_egress() -> list[str]:
    probe = _docker("exec", "-i", CONTAINER, "python", "-", stdin=EGRESS_PROBE)
    return egress_problems(json.loads(probe.stdout))


def up(cfg: Config, ready_timeout_s: float = 180) -> None:
    if is_running():
        client(cfg).wait_ready(ready_timeout_s)
        return
    verdict = run_preflight(cfg.gpu.agent_service, cfg.gpu.min_free_gib)
    if not verdict.ok:
        raise RuntimeError("GPU not free: " + "; ".join(verdict.problems))
    for name in IO_SUBDIRS:
        private_dir(cfg.comfy_io_dir / name)
    ensure_network(cfg)
    _docker("rm", "--force", CONTAINER, check=False)
    _docker(*run_args(cfg, image_tag(cfg), os.getuid(), os.getgid())[1:])
    try:
        client(cfg).wait_ready(ready_timeout_s)
        problems = check_egress()
        if problems:
            raise RuntimeError("network isolation failed: " + "; ".join(problems))
    except BaseException:
        down(cfg)
        raise


def down(cfg: Config, timeout_s: float = 60) -> None:
    """Stops the container and waits until none of its processes hold the GPU; a stopped container keeps its logs
    until the next `up`."""
    pids = container_pids()
    if is_running():
        _docker("stop", CONTAINER)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if apps.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed, so the GPU cannot be confirmed free: {apps.stderr.strip()}")
        if not pids & {p.pid for p in parse_compute_apps(apps.stdout)}:
            return
        time.sleep(1)
    raise TimeoutError(f"container processes still on the GPU after {timeout_s:.0f} s")
