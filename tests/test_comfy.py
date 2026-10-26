import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from studio.gpu.comfy import ComfyClient, RenderError, WorkflowError, output_files


class FakeComfy:
    """Just enough of ComfyUI's HTTP API: /prompt, /history/{id}, /free, /system_stats."""

    def __init__(self, outcome="success", polls_before_done=2):
        self.outcome, self.polls_left = outcome, polls_before_done
        self.prompts, self.frees = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/system_stats":
                    return self.reply(200, {"system": {"comfyui_version": "test"}})
                prompt_id = self.path.rsplit("/", 1)[-1]
                if fake.polls_left > 0:
                    fake.polls_left -= 1
                    return self.reply(200, {})
                return self.reply(200, {prompt_id: fake.history_entry()})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/free":
                    fake.frees.append(body)
                    return self.reply(200, {})
                if "bad" in body["prompt"]:
                    return self.reply(400, {"error": {"message": "Prompt outputs failed validation"},
                                            "node_errors": {"7": {"errors": [{"message": "Value not in list"}],
                                                                  "class_type": "LoadImage"}}})
                fake.prompts.append(body)
                return self.reply(200, {"prompt_id": body["prompt_id"], "number": 1, "node_errors": {}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def history_entry(self):
        if self.outcome == "error":
            return {"status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]]},
                "outputs": {}}
        return {"status": {"status_str": "success", "completed": True, "messages": []},
                "outputs": {"12": {"images": [{"filename": "clip_00001_.mp4", "subfolder": "studio",
                                               "type": "output"}], "animated": [True]}}}


@pytest.fixture
def fake():
    server = FakeComfy()
    yield server
    server.server.shutdown()


def test_run_submits_our_prompt_id_and_waits_for_success(fake):
    client = ComfyClient(fake.url, poll_s=0.01)
    entry = client.run({"1": {"class_type": "Noop", "inputs": {}}}, timeout_s=5)
    assert fake.prompts[0]["prompt_id"] and fake.prompts[0]["client_id"]
    assert output_files(entry) == [("studio", "clip_00001_.mp4", "output")]


def test_validation_errors_name_the_node(fake):
    with pytest.raises(WorkflowError, match="7 LoadImage: Value not in list"):
        ComfyClient(fake.url, poll_s=0.01).run({"bad": {}}, timeout_s=5)


def test_execution_errors_carry_the_exception_message():
    server = FakeComfy(outcome="error")
    try:
        with pytest.raises(RenderError, match="KSampler: CUDA out of memory"):
            ComfyClient(server.url, poll_s=0.01).run({"1": {}}, timeout_s=5)
    finally:
        server.server.shutdown()


def test_a_render_that_never_finishes_times_out():
    server = FakeComfy(polls_before_done=10**9)
    try:
        with pytest.raises(TimeoutError):
            ComfyClient(server.url, poll_s=0.01).run({"1": {}}, timeout_s=0.2)
    finally:
        server.server.shutdown()


def test_free_asks_comfyui_to_unload_models_and_free_memory(fake):
    ComfyClient(fake.url).free()
    assert fake.frees == [{"unload_models": True, "free_memory": True}]
