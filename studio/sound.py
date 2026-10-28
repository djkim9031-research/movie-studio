"""PrismAudio, the sound worker: a track for a finished clip from its sound prompt, made in the worker's own Python
environment (torch and JAX for this GPU) from pinned source and weights, offline. It runs while ComfyUI is down."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from studio import models
from studio.config import Config
from studio.fs import write_private

RUNNER = Path(__file__).resolve().parent.parent / "workers" / "prismaudio" / "run.py"
SPEECH = RUNNER.with_name("speech.py")
SITE = Path(".venv") / "lib" / "python3.11" / "site-packages"
ERROR = "prismaudio: error: "  # the runner's own error line
SETS = ["prismaudio", "videoprism-lvt", "t5gemma"]
# Paths under models/audio.
TOKENIZER = "videoprism/cc_en.32000.sentencepiece.model"
T5GEMMA = "t5gemma-l-l-ul2-it"


class SoundError(RuntimeError):
    pass


@dataclass
class Track:
    path: Path
    elapsed_s: float
    peak_gib: float | None


class PrismAudio:
    models = SETS

    @staticmethod
    def home(cfg: Config) -> Path:
        return cfg.data_root / "workers" / "prismaudio"

    def python(self, cfg: Config) -> Path:
        return self.home(cfg) / ".venv" / "bin" / "python"

    def detector(self, cfg: Config) -> Path:
        """Silero VAD in the worker environment: it hears speech-like sound in a generated track."""
        return self.home(cfg) / SITE / "silero_vad"

    @staticmethod
    def audio_root(cfg: Config) -> Path:
        return cfg.data_root / "models" / "audio"

    def problems(self, cfg: Config) -> list[str]:
        found = []
        for name in SETS:
            if name not in cfg.models:
                found.append(f"model set {name} is not in the config")
                continue
            missing = models.present(cfg, name)
            if missing:
                found.append(f"model set {name}: {missing[0]}; run: uv run studio models fetch {name}")
        if not (self.audio_root(cfg) / TOKENIZER).is_file():
            found.append(f"no VideoPrism text tokenizer at {self.audio_root(cfg) / TOKENIZER}")
        if not self.python(cfg).exists() or not (self.home(cfg) / "src" / "predict.py").is_file():
            found.append(f"no worker environment in {self.home(cfg)}; see the README, Sound")
        elif not self.detector(cfg).is_dir():
            found.append(f"no speech detector in {self.home(cfg)}; rerun workers/prismaudio/setup.sh")
        return found

    def render(self, cfg: Config, *, clip: Path, prompt: str, seconds: int, seed: int, dest: Path) -> Track:
        root = self.audio_root(cfg)
        job = {"source": str(self.home(cfg) / "src"), "clip": str(clip), "prompt": prompt, "seconds": seconds,
               "seed": seed, "out": str(dest), "device": "cuda",
               "weights": {"prismaudio": str(root / "prismaudio" / "prismaudio.ckpt"),
                           "vae": str(root / "prismaudio" / "vae.ckpt"),
                           "synchformer": str(root / "prismaudio" / "synchformer_state_dict.pth"),
                           "videoprism": str(root / "videoprism" / "flax_lvt_large_f8r288_repeated.npz"),
                           "videoprism_tokenizer": str(root / TOKENIZER),
                           "t5gemma": str(root / T5GEMMA)}}
        started = time.monotonic()
        result = subprocess.run([str(self.python(cfg)), "-I", str(RUNNER)], input=json.dumps(job),
                                capture_output=True, text=True, timeout=3600, env=_env(), cwd=self.home(cfg))
        if result.returncode != 0:
            log = dest.with_suffix(".log")
            write_private(log, result.stderr)
            # Upstream code can print after the runner's error line.
            lines = result.stderr.strip().splitlines()
            errors = [line.removeprefix(ERROR) for line in lines if line.startswith(ERROR)]
            message = errors[-1] if errors else lines[-1] if lines else "no message"
            raise SoundError(f"PrismAudio failed on {clip.name}: {message}; log: {log}")
        if not dest.is_file():
            raise SoundError(f"PrismAudio wrote no track for {clip.name}")
        reply = json.loads(result.stdout.strip().splitlines()[-1])
        return Track(path=dest, elapsed_s=time.monotonic() - started, peak_gib=reply.get("peak_gib"))

    def speech(self, cfg: Config, wav: Path) -> float:
        """The share of `wav` that Silero VAD hears as speech, 0 to 1. On CPU, a second or two."""
        result = subprocess.run([str(self.python(cfg)), "-I", str(SPEECH)], input=json.dumps({"wav": str(wav)}),
                                capture_output=True, text=True, timeout=600, env=_env(), cwd=self.home(cfg))
        if result.returncode != 0:
            lines = result.stderr.strip().splitlines()
            raise SoundError(f"the speech check failed on {wav.name}: {lines[-1] if lines else 'no message'}")
        return float(json.loads(result.stdout.strip().splitlines()[-1])["speech"])


def _env() -> dict:
    """A clean environment: no tokens, no network fetches, and JAX takes GPU memory only as it needs it."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "TF_CPP_MIN_LOG_LEVEL": "2"}
