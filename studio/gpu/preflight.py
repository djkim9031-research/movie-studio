"""Refuse to start GPU work unless this GPU is ours.

Only compute processes count: the display runs on this GPU, and its graphics processes are not listed by
--query-compute-apps. Their memory still shows in memory.used, which the free-memory check accounts for. A display
program that opens a CUDA context anyway (NoMachine's screen encoder, for NVENC) is named in gpu.display_processes and
reported instead of refused.
"""

import os
import subprocess
from dataclasses import dataclass, field

from studio.config import GpuConfig

MIB_PER_GIB = 1024


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    used_mib: int | None

    @property
    def held(self) -> str:
        return f"{self.used_mib} MiB" if self.used_mib is not None else "unknown memory"


@dataclass(frozen=True)
class GpuMemory:
    total_mib: int
    used_mib: int

    @property
    def free_gib(self) -> float:
        return (self.total_mib - self.used_mib) / MIB_PER_GIB


@dataclass
class Verdict:
    ok: bool
    problems: list[str] = field(default_factory=list)
    memory: GpuMemory | None = None
    display: list[GpuProcess] = field(default_factory=list)


def _int_or_none(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def parse_compute_apps(text: str) -> list[GpuProcess]:
    """Rows of `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`."""
    processes = []
    for line in text.splitlines():
        if line.count(",") < 2:
            continue
        pid, rest = line.split(",", 1)
        name, used = rest.rsplit(",", 1)
        pid_value = _int_or_none(pid)
        if pid_value is not None:
            processes.append(GpuProcess(pid_value, name.strip(), _int_or_none(used)))
    return processes


def parse_gpu_memory(text: str) -> GpuMemory:
    """First row of `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits`."""
    total, used = (int(part) for part in text.strip().splitlines()[0].split(","))
    return GpuMemory(total_mib=total, used_mib=used)


def evaluate(processes: list[GpuProcess], memory: GpuMemory, *, agent_active: bool, agent_service: str,
             min_free_gib: float, own_pids: frozenset[int] | set[int] = frozenset(),
             display: frozenset[str] | set[str] = frozenset()) -> Verdict:
    problems, shown = [], []
    if agent_active:
        problems.append(f"{agent_service} is running; stop it first: sudo systemctl stop {agent_service}")
    for p in processes:
        if p.pid in own_pids:
            continue
        if p.name in display:
            shown.append(p)
        else:
            problems.append(f"pid {p.pid} ({p.name}) holds {p.held} on the GPU")
    if memory.free_gib < min_free_gib:
        problems.append(f"{memory.free_gib:.1f} GiB free, need {min_free_gib:.0f}")
    return Verdict(ok=not problems, problems=problems, memory=memory, display=shown)


def _nvidia_smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30).stdout


def service_is_active(name: str) -> bool:
    result = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() == "active"


def run_preflight(gpu: GpuConfig, min_free_gib: float | None = None,
                  own_pids: frozenset[int] | set[int] = frozenset()) -> Verdict:
    """The GPU checked against `gpu`'s agent service, display processes and free-memory floor; `min_free_gib`, when
    given, replaces the floor."""
    processes = parse_compute_apps(
        _nvidia_smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"))
    memory = parse_gpu_memory(_nvidia_smi("--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"))
    return evaluate(processes, memory, agent_active=service_is_active(gpu.agent_service),
                    agent_service=gpu.agent_service,
                    min_free_gib=gpu.min_free_gib if min_free_gib is None else min_free_gib,
                    own_pids={os.getpid(), *own_pids}, display=set(gpu.display_processes))
