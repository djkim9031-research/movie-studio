"""Where a generated file came from: a sidecar beside it with its sha256, the model sets that made it, its inputs and
its settings. Face recognition runs only on files whose whole lineage allows it: camera originals, and files made by
model sets whose license permits it (config models.*.biometric_outputs)."""

import json
from pathlib import Path

from studio.config import Config
from studio.fs import sha256_file, write_private

SUFFIX = ".prov.json"
CAPTURE_FOLDERS = ("photos", "video", "bench")


def sidecar(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + SUFFIX)


def is_camera_original(cfg: Config, path: Path) -> bool:
    """Files the user put in a person's capture folders: ~/studio-data/cast/ALIAS/{photos,video,bench}."""
    try:
        parts = Path(path).resolve().relative_to((cfg.data_root / "cast").resolve()).parts
    except ValueError:
        return False
    return len(parts) >= 3 and parts[1] in CAPTURE_FOLDERS


def write(cfg: Config, path: Path, *, kind: str, models: list[str], inputs: list[Path], params: dict) -> dict:
    record = {"file": Path(path).name, "sha256": sha256_file(path), "kind": kind,
              "models": [{"set": name, "revision": cfg.models[name].revision if name in cfg.models else None}
                         for name in models],
              "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in inputs],
              "params": params}
    write_private(sidecar(path), json.dumps(record, indent=1))
    return record


def problems(cfg: Config, path: Path, _seen: frozenset = frozenset()) -> list[str]:
    """Reasons `path` must not be face-scored; empty when its whole lineage allows it."""
    path = Path(path)
    if is_camera_original(cfg, path):
        return []
    if path.resolve() in _seen:
        return [f"{path.name}: its provenance loops back on itself"]
    if not sidecar(path).is_file():
        return [f"{path.name}: unknown origin (no provenance)"]
    record = json.loads(sidecar(path).read_text(encoding="utf-8"))
    if sha256_file(path) != record.get("sha256"):
        return [f"{path.name}: changed since its provenance was written"]
    found = []
    for model in record.get("models", []):
        status = cfg.models[model["set"]].biometric_outputs if model["set"] in cfg.models else "unknown"
        if status == "forbidden":
            found.append(f"{path.name}: made by {model['set']}, whose license forbids face recognition")
        elif status != "allowed":
            found.append(f"{path.name}: made by {model['set']}, whose license for face recognition is unknown")
    for item in record.get("inputs", []):
        source = Path(item["path"])
        if not source.exists():
            found.append(f"{path.name}: input {source.name} is gone")
        elif sha256_file(source) != item["sha256"]:
            found.append(f"{path.name}: input {source.name} changed since")
        else:
            found += problems(cfg, source, _seen | {path.resolve()})
    return found


def params(path: Path) -> dict:
    """The settings recorded with a generated file; empty without a sidecar."""
    side = sidecar(path)
    return json.loads(side.read_text(encoding="utf-8")).get("params", {}) if side.is_file() else {}
