import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from studio.gpu.comfy import ComfyClient, ComfyUnavailable, RenderError, WorkflowError, cached_nodes, output_files

DONE = {"status": {"status_str": "success", "completed": True, "messages": [
            ["execution_start", {}], ["execution_cached", {"nodes": ["398:364"]}], ["execution_success", {}]]},
        "outputs": {"75": {"images": [{"filename": "clip_00001_.mp4", "subfolder": "studio", "type": "output"}]}}}


class FakeComfy:
    """Enough of ComfyUI's HTTP API to drive the client: /prompt, /history/{id}, /free, /system_stats."""

    def __init__(self, entry=DONE, polls_before_done=2, history_code=200, prompt_reply=None, prompt_raw=None):
        self.entry, self.polls_left, self.history_code = entry, polls_before_done, history_code
        self.prompt_reply, self.prompt_raw = prompt_reply, prompt_raw
        self.queued, self.frees = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, raw=None):
                data = raw if raw is not None else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/system_stats":
                    return self.reply(200, {"system": {"comfyui_version": "test"}})
                prompt_id = self.path.rsplit("/", 1)[-1]
                if fake.history_code != 200:
                    return self.reply(fake.history_code, {"error": "boom"})
                if prompt_id not in fake.queued or fake.polls_left > 0:
                    fake.polls_left -= 1
                    return self.reply(200, {})
                return self.reply(200, {prompt_id: fake.entry})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/free":
                    fake.frees.append(body)
                    return self.reply(200, {})
                if fake.prompt_raw is not None:
                    return self.reply(500, None, raw=fake.prompt_raw)
                if "bad" in body["prompt"]:
                    return self.reply(400, {"error": {"message": "Prompt outputs failed validation"},
                                            "node_errors": {"7": {"errors": [{"message": "Value not in list"}],
                                                                  "class_type": "LoadImage"}}})
                fake.queued.append(body["prompt_id"])
                reply = fake.prompt_reply or {"prompt_id": body["prompt_id"], "number": 1, "node_errors": {}}
                return self.reply(200, reply)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def comfy():
    servers = []

    def start(**kwargs):
        servers.append(FakeComfy(**kwargs))
        return servers[-1]

    yield start
    for server in servers:
        server.server.shutdown()


def test_run_submits_our_prompt_id_and_waits_for_its_history(comfy):
    fake = comfy()
    entry = ComfyClient(fake.url, poll_s=0.01).run({"1": {"class_type": "Noop", "inputs": {}}}, timeout_s=5)
    assert fake.queued and output_files(entry) == [("studio", "clip_00001_.mp4", "output")]


def test_cached_nodes_lists_what_comfyui_reused():
    assert cached_nodes(DONE) == {"398:364"}


def test_validation_errors_name_the_node(comfy):
    with pytest.raises(WorkflowError, match="7 LoadImage: Value not in list"):
        ComfyClient(comfy().url, poll_s=0.01).run({"bad": {}}, timeout_s=5)


def test_node_errors_in_a_200_reply_are_refused(comfy):
    fake = comfy(prompt_reply={"prompt_id": "x", "node_errors": {"3": {"errors": [{"message": "bad input"}],
                                                                       "class_type": "VAELoader"}}})
    with pytest.raises(WorkflowError, match="3 VAELoader: bad input"):
        ComfyClient(fake.url, poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_a_reply_queued_under_another_id_is_refused(comfy):
    fake = comfy(prompt_reply={"prompt_id": "someone-else", "node_errors": {}})
    with pytest.raises(WorkflowError, match="someone-else"):
        ComfyClient(fake.url, poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_execution_errors_carry_the_exception_message(comfy):
    failed = {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]]}, "outputs": {}}
    with pytest.raises(RenderError, match="KSampler: CUDA out of memory"):
        ComfyClient(comfy(entry=failed).url, poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_an_interrupted_run_says_so(comfy):
    stopped = {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_interrupted", {"node_type": "SamplerCustomAdvanced"}]]}, "outputs": {}}
    with pytest.raises(RenderError, match="interrupted"):
        ComfyClient(comfy(entry=stopped).url, poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_a_failing_history_endpoint_fails_fast(comfy):
    with pytest.raises(RenderError, match="HTTP 500"):
        ComfyClient(comfy(history_code=500).url, poll_s=0.01).run({"1": {}}, timeout_s=60)


def test_a_reply_that_is_not_json_is_reported(comfy):
    with pytest.raises(RenderError, match="not JSON"):
        ComfyClient(comfy(prompt_raw=b"500 Internal Server Error").url, poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_a_comfyui_that_is_not_there_is_reported_as_unavailable():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(ComfyUnavailable):
        ComfyClient(f"http://127.0.0.1:{port}", poll_s=0.01).run({"1": {}}, timeout_s=5)


def test_proxy_settings_do_not_reroute_requests_to_comfyui(comfy, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    entry = ComfyClient(comfy().url, poll_s=0.01).run({"1": {}}, timeout_s=5)
    assert output_files(entry)


def test_a_render_that_never_finishes_times_out(comfy):
    with pytest.raises(TimeoutError):
        ComfyClient(comfy(polls_before_done=10**9).url, poll_s=0.01).run({"1": {}}, timeout_s=0.2)


def test_free_asks_comfyui_to_unload_models_and_free_memory(comfy):
    fake = comfy()
    ComfyClient(fake.url).free()
    assert fake.frees == [{"unload_models": True, "free_memory": True}]
