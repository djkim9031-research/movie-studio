import sys

from studio.config import Config
from studio.gpu.runtime import parse_top_pids, run_args
from studio.gpu.vram import VramSampler


def make_config(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path),
                                  "comfyui": {"image": "studio-comfyui:test", "port": 8188}})


def test_run_args_give_the_gpu_and_keep_the_container_locked_down(tmp_path):
    args = run_args(make_config(tmp_path), uid=1000, gid=1000)
    joined = " ".join(args)
    assert args[:3] == ["docker", "run", "--detach"]
    assert "--gpus all" in joined and "--user 1000:1000" in joined
    assert f"--volume {tmp_path}/models/comfy:/models:ro" in joined
    assert f"--volume {tmp_path}/comfy-io:/io" in joined
    assert "--publish 127.0.0.1:8188:8188" in joined
    assert "--cap-drop ALL" in joined and "--security-opt no-new-privileges" in joined
    assert "--env HF_HUB_OFFLINE=1" in joined and "--env HOME=/io/home" in joined
    assert args[-1] == "studio-comfyui:test"


def test_parse_top_pids_reads_the_pid_column():
    assert parse_top_pids("PID\n101\n202\n") == {101, 202}
    assert parse_top_pids("") == set()


def test_vram_sampler_reports_the_first_sample_as_baseline_and_the_peak():
    script = "import time\nfor v in (1200, 5000, 30000, 8000):\n    print(v, flush=True)\n    time.sleep(0.02)\n"
    with VramSampler(command=[sys.executable, "-c", script]) as sampler:
        sampler.wait_for_samples(4, timeout=5)
    assert (sampler.baseline_mib, sampler.peak_mib) == (1200, 30000)
    assert sampler.peak_over_baseline_gib == (30000 - 1200) / 1024
