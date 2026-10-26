"""GPU memory sampled every 100 ms by nvidia-smi while a render runs; the first sample is the baseline."""

import subprocess
import threading
import time

MIB_PER_GIB = 1024


class VramSampler:
    def __init__(self, command: list[str] | None = None, interval_ms: int = 100):
        self.command = command or ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                   "-lms", str(interval_ms)]
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

    def __exit__(self, *exc) -> None:
        self.process.terminate()
        self.process.wait(timeout=5)
        self.thread.join(timeout=5)

    @property
    def baseline_mib(self) -> int:
        return self.samples[0]

    @property
    def peak_mib(self) -> int:
        return max(self.samples)

    @property
    def peak_over_baseline_gib(self) -> float:
        return (self.peak_mib - self.baseline_mib) / MIB_PER_GIB
