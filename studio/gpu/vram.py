"""GPU memory sampled every 100 ms by nvidia-smi while a render runs."""

import subprocess
import threading
import time


class VramSampler:
    def __init__(self, command: list[str] | None = None, interval_ms: int = 100, stop_timeout_s: float = 5.0):
        self.command = command or ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                   "-lms", str(interval_ms)]
        self.stop_timeout_s = stop_timeout_s
        self.samples: list[int] = []

    def __enter__(self):
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self) -> None:
        for line in self.process.stdout:
            value = line.strip().split(",")[0]
            if value.isdigit():
                self.samples.append(int(value))

    def wait_for_samples(self, count: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while len(self.samples) < count and time.monotonic() < deadline:
            time.sleep(0.01)

    def __exit__(self, *exc) -> bool:
        # Never raises: an error here must not replace the render's own error.
        self.process.terminate()
        try:
            self.process.wait(timeout=self.stop_timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=self.stop_timeout_s)
        return False

    @property
    def peak_mib(self) -> int | None:
        return max(self.samples) if self.samples else None
