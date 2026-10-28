import re
import sys
from pathlib import Path

import pytest

from studio.config import Config, load_config
from studio.gpu import runtime
from studio.gpu.preflight import Verdict
from studio.gpu.runtime import (egress_problems, network_create_args, network_problems, parse_top_pids,
                                run_args)
from studio.gpu.vram import VramSampler


DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker" / "comfyui"


def make_config(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path)})


def test_run_args_give_the_gpu_and_keep_the_container_locked_down(tmp_path):
    cfg = make_config(tmp_path)
    args = run_args(cfg, image="studio-comfyui:abc", uid=1000, gid=1000)
    joined = " ".join(args)
    assert args[:3] == ["docker", "run", "--detach"]
    assert "--gpus all" in joined and "--user 1000:1000" in joined and "--init" in joined
    assert "--network studio-net" in joined and "--ip 192.168.131.2" in joined and "--dns 127.0.0.1" in joined
    assert f"--mount type=bind,src={tmp_path}/models/comfy,dst=/models,readonly" in joined
    assert f"--mount type=bind,src={tmp_path}/comfy-io,dst=/io" in joined
    assert "--cap-drop ALL" in joined and "--security-opt no-new-privileges" in joined
    assert "--pull never" in joined and "--env HF_HUB_OFFLINE=1" in joined
    assert "--publish" not in args and "-p" not in args and "--privileged" not in args
    assert args[-1] == "studio-comfyui:abc"


def test_network_is_created_internal_on_its_own_subnet(tmp_path):
    assert network_create_args(make_config(tmp_path)) == [
        "docker", "network", "create", "--driver", "bridge", "--internal", "--subnet", "192.168.131.0/29",
        "--gateway", "192.168.131.1", "studio-net"]


def test_an_existing_network_must_be_internal_with_the_pinned_subnet(tmp_path):
    cfg = make_config(tmp_path)
    good = {"Internal": True, "IPAM": {"Config": [{"Subnet": "192.168.131.0/29"}]}}
    assert network_problems(good, cfg) == []
    assert "not internal" in network_problems({**good, "Internal": False}, cfg)[0]
    assert "subnet" in network_problems({"Internal": True, "IPAM": {"Config": [{"Subnet": "10.0.0.0/24"}]}},
                                        cfg)[0]


def test_egress_problems_name_every_target_the_container_reached():
    assert egress_problems({"1.1.1.1:443": False, "8.8.8.8:53": False, "dns pypi.org": False}) == []
    problems = egress_problems({"1.1.1.1:443": True, "8.8.8.8:53": False, "dns pypi.org": True})
    assert problems == ["the container reached 1.1.1.1:443", "the container reached dns pypi.org"]


def test_parse_top_pids_reads_the_pid_column():
    assert parse_top_pids("PID\n101\n202\n") == {101, 202}
    assert parse_top_pids("") == set()


def test_vram_sampler_reports_the_absolute_peak():
    script = "import time\nfor v in (1200, 5000, 30000, 8000):\n    print(v, flush=True)\n    time.sleep(0.02)\n"
    with VramSampler(command=[sys.executable, "-c", script]) as sampler:
        sampler.wait_for_samples(4, timeout=5)
    assert sampler.peak_mib == 30000


def test_vram_sampler_without_samples_reports_no_peak_instead_of_failing():
    with VramSampler(command=[sys.executable, "-c", "print('NVML error')"]) as sampler:
        sampler.wait_for_samples(1, timeout=0.5)
    assert sampler.peak_mib is None


def test_vram_sampler_exit_never_raises_even_if_the_sampler_ignores_sigterm():
    script = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint(1, flush=True)\ntime.sleep(60)\n"
    sampler = VramSampler(command=[sys.executable, "-c", script], stop_timeout_s=0.5)
    with sampler:
        sampler.wait_for_samples(1, timeout=5)
    assert sampler.process.poll() is not None


def test_every_package_in_the_image_comes_from_the_hashed_lock():
    instructions = re.sub(r"\\\n", " ", (DOCKER_DIR / "Dockerfile").read_text()).splitlines()
    installs = [line for line in instructions if "pip install" in line]
    assert installs and all("--require-hashes" in line and "--no-deps" in line and "requirements.lock" in line
                            for line in installs)


def test_the_lock_pins_every_package_with_hashes_and_the_configured_torch():
    pins = re.split(r"\n(?=\S)", (DOCKER_DIR / "requirements.lock").read_text().strip())
    pins = [pin for pin in pins if not pin.startswith("#")]
    assert len(pins) > 50 and all(re.match(r"^[A-Za-z0-9._-]+==\S+ ", pin) and "--hash=sha256:" in pin for pin in pins)
    assert any(pin.startswith(f"torch=={load_config().comfyui.torch}+cu130 ") for pin in pins)


def test_up_refuses_a_busy_gpu_before_any_docker_call(tmp_path, monkeypatch):
    cfg, seen = make_config(tmp_path), []
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: False)
    monkeypatch.setattr("studio.gpu.runtime.run_preflight", lambda gpu: seen.append(gpu) or Verdict(
        ok=False, problems=["pid 4242 (python3) holds 9000 MiB on the GPU"]))
    monkeypatch.setattr("studio.gpu.runtime._docker", lambda *a, **k: pytest.fail("docker ran on a busy GPU"))
    with pytest.raises(runtime.GpuBusy, match=r"GPU not free: pid 4242 \(python3\) holds 9000 MiB on the GPU"):
        runtime.up(cfg)
    assert seen == [cfg.gpu]
