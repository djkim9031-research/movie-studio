"""Client for ComfyUI's HTTP API: queue a workflow under our own prompt id, poll its history, free GPU memory.

Polling /history/{prompt_id} needs no websocket and cannot miss an event: the entry appears once the run ends.
Requests never go through a proxy; ComfyUI is on this machine.
"""

import json
import time
import urllib.error
import urllib.request
import uuid


class WorkflowError(RuntimeError):
    """ComfyUI refused the workflow before running it."""


class RenderError(RuntimeError):
    """The workflow ran and failed, or ComfyUI answered in a way the client cannot use."""


class ComfyUnavailable(RenderError):
    """ComfyUI did not answer at all."""


def output_files(entry: dict) -> list[tuple[str, str, str]]:
    """(subfolder, filename, type) of every file a finished run reported, in node order."""
    files = []
    for node_output in entry.get("outputs", {}).values():
        for items in node_output.values():
            if isinstance(items, list):
                files += [(item.get("subfolder", ""), item["filename"], item.get("type", "output"))
                          for item in items if isinstance(item, dict) and "filename" in item]
    return files


def cached_nodes(entry: dict) -> set[str]:
    """Nodes ComfyUI took from an earlier run instead of executing."""
    nodes = set()
    for kind, data in entry.get("status", {}).get("messages", []):
        if kind == "execution_cached":
            nodes.update(str(node) for node in data.get("nodes", []))
    return nodes


def _node_errors(body: dict) -> str:
    parts = [f"{node_id} {err.get('class_type', '?')}: {e.get('message')}"
             for node_id, err in (body.get("node_errors") or {}).items() for e in err.get("errors", [])]
    return "; ".join(parts) or (body.get("error") or {}).get("message", "workflow refused")


def _execution_error(status: dict) -> str:
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            return f"{data.get('node_type')}: {str(data.get('exception_message', '')).strip()}"
        if kind == "execution_interrupted":
            return f"interrupted at {data.get('node_type')}"
    return "execution failed"


class ComfyClient:
    def __init__(self, base_url: str, poll_s: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.poll_s = poll_s
        self.client_id = str(uuid.uuid4())
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                code, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, raw = error.code, error.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            raise ComfyUnavailable(f"ComfyUI is not answering at {self.base_url}: "
                                   f"{getattr(error, 'reason', error)}") from None
        try:
            return code, json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise RenderError(f"{path} answered HTTP {code} with a body that is not JSON: {raw[:120]!r}") from None

    def wait_ready(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                code, body = self._request("/system_stats", timeout=5)
                if code == 200:
                    return body
            except ComfyUnavailable:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"ComfyUI not answering at {self.base_url} after {timeout_s:.0f} s")
            time.sleep(1)

    def submit(self, workflow: dict) -> str:
        prompt_id = str(uuid.uuid4())
        code, body = self._request("/prompt", {"prompt": workflow, "client_id": self.client_id,
                                               "prompt_id": prompt_id})
        if code != 200 or body.get("node_errors"):
            raise WorkflowError(_node_errors(body))
        if body.get("prompt_id") != prompt_id:
            raise WorkflowError(f"ComfyUI queued the workflow as {body.get('prompt_id')}, not {prompt_id}")
        return prompt_id

    def wait(self, prompt_id: str, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code, body = self._request(f"/history/{prompt_id}")
            if code != 200:
                raise RenderError(f"/history answered HTTP {code}")
            entry = body.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RenderError(_execution_error(status))
                if status.get("completed") or status.get("status_str") == "success":
                    return entry
            time.sleep(self.poll_s)
        raise TimeoutError(f"prompt {prompt_id} still running after {timeout_s:.0f} s")

    def run(self, workflow: dict, timeout_s: float) -> dict:
        return self.wait(self.submit(workflow), timeout_s)

    def free(self) -> None:
        """Asks ComfyUI to unload models; it only sets a flag, so callers wait for GPU memory to drop."""
        self._request("/free", {"unload_models": True, "free_memory": True})
