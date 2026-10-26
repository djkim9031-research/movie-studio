from studio.gpu.preflight import GpuMemory, GpuProcess, evaluate, parse_compute_apps, parse_gpu_memory

IDLE = GpuMemory(total_mib=73415, used_mib=1271)


def check(processes=(), memory=IDLE, agent_active=False, own_pids=frozenset()):
    return evaluate(list(processes), memory, agent_active=agent_active, agent_service="agent-vllm",
                    min_free_gib=60, own_pids=own_pids)


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
