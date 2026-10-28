from studio.config import GpuConfig
from studio.gpu.preflight import (GpuMemory, GpuProcess, evaluate, parse_compute_apps, parse_gpu_memory,
                                  run_preflight)

IDLE = GpuMemory(total_mib=73415, used_mib=1271)
NOMACHINE = "/usr/NX/bin/nxnode.bin"


def check(processes=(), memory=IDLE, agent_active=False, own_pids=frozenset(), display=frozenset()):
    return evaluate(list(processes), memory, agent_active=agent_active, agent_service="agent-vllm",
                    min_free_gib=60, own_pids=own_pids, display=display)


def test_parse_compute_apps_reads_pid_name_memory():
    text = "509258, /usr/bin/python3, 2048\n77, VLLM::EngineCore, 56644\n"
    assert parse_compute_apps(text) == [GpuProcess(509258, "/usr/bin/python3", 2048),
                                        GpuProcess(77, "VLLM::EngineCore", 56644)]


def test_parse_compute_apps_empty_output_means_no_processes():
    assert parse_compute_apps("") == []


def test_parse_compute_apps_keeps_unknown_memory_as_none():
    assert parse_compute_apps("12, python, [N/A]\n") == [GpuProcess(12, "python", None)]


def test_parse_gpu_memory():
    assert parse_gpu_memory("73415, 1271\n") == IDLE


def test_idle_gpu_passes():
    verdict = check()
    assert verdict.ok and verdict.problems == []


def test_foreign_compute_process_is_refused_and_named():
    verdict = check(processes=[GpuProcess(4242, "python3", 9000)])
    assert not verdict.ok
    assert "4242" in verdict.problems[0] and "python3" in verdict.problems[0]


def test_own_process_is_not_refused():
    assert check(processes=[GpuProcess(4242, "python3", 9000)], own_pids={4242}).ok


def test_active_agent_service_asks_user_to_stop_it():
    verdict = check(agent_active=True)
    assert not verdict.ok
    assert "sudo systemctl stop agent-vllm" in verdict.problems[0]


def test_too_little_free_memory_is_refused():
    verdict = check(memory=GpuMemory(total_mib=73415, used_mib=20000))
    assert not verdict.ok
    assert "GiB free" in verdict.problems[0]


def test_a_display_process_is_let_through_and_reported():
    encoder = GpuProcess(26315, NOMACHINE, 406)
    verdict = check(processes=[encoder], display={NOMACHINE})
    assert verdict.ok and verdict.display == [encoder]


def test_a_display_process_is_matched_by_its_whole_name_only():
    verdict = check(processes=[GpuProcess(7, "nxnode.bin", 406), GpuProcess(8, "/tmp" + NOMACHINE, 10)],
                    display={NOMACHINE})
    assert not verdict.ok and len(verdict.problems) == 2 and verdict.display == []


def test_run_preflight_takes_the_agent_service_and_the_display_from_the_gpu_config(monkeypatch):
    smi = {"--query-compute-apps=pid,process_name,used_memory": f"26315, {NOMACHINE}, 406\n4242, python3, 9000\n",
           "--query-gpu=memory.total,memory.used": "73415, 10705\n"}
    asked = []
    monkeypatch.setattr("studio.gpu.preflight._nvidia_smi", lambda query, fmt: smi[query])
    monkeypatch.setattr("studio.gpu.preflight.service_is_active", lambda name: asked.append(name) or False)
    verdict = run_preflight(GpuConfig())
    assert asked == ["agent-vllm"]
    assert verdict.problems == ["pid 4242 (python3) holds 9000 MiB on the GPU"]
    assert verdict.display == [GpuProcess(26315, NOMACHINE, 406)]
