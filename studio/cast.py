"""A person's capture turned into a training set: photos and video frames that show them, binned by head angle, the
best few per bin with near-copies dropped, each a square crop with a short caption and provenance back to the camera
file. Membership is checked against the person's saved profile (studio identity profile)."""

import bisect
import shutil
import tempfile
from pathlib import Path

from studio import identity, provenance, refs
from studio.config import Config
from studio.fs import private_dir, write_private
from studio.media import ffmpeg

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv"}
# Bin edges on the yaw proxy (nose offset in eye distances); the sign is not trusted until calibrated on a real turn.
EDGES = [-0.6, -0.35, -0.15, 0.15, 0.35, 0.6]
WORDS = ["frontal", "slightly turned", "three-quarter view", "near profile"]


def angle_bin(yaw: float) -> int:
    return bisect.bisect_right(EDGES, yaw)


def angle_words(yaw: float) -> str:
    size = abs(yaw)
    return WORDS[0] if size < 0.15 else WORDS[1] if size < 0.35 else WORDS[2] if size < 0.6 else WORDS[3]


def _frames(cfg: Config, video: Path, out: Path, fps: float) -> list[Path]:
    private_dir(out)
    ffmpeg.run(cfg, "-y", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", str(out / "%05d.png"),
               timeout=1800).check_returncode()
    return sorted(out.glob("*.png"))


def _the_person(faces: list, centroid) -> tuple:
    """The face most like the person, if it is the same person at all; the rest are others."""
    scored = [(float(f.embedding @ centroid), f) for f in faces if f.embedding is not None]
    if not scored:
        return None, faces
    sim, face = max(scored, key=lambda item: item[0])
    return (face if sim >= refs.SAME_PERSON else None), [f for f in faces if f is not face]


def ingest(cfg: Config, alias: str, tools, *, echo, per_bin: int = 5, fps: float = 2.0, trigger: str | None = None,
           replace: bool = False) -> dict:
    root = cfg.data_root / "cast" / alias
    consent = root / "consent.txt"
    if not consent.is_file() or not consent.read_text(encoding="utf-8").strip():
        raise ValueError(f"write {consent} first: who agreed, when, and what the pictures are for")
    centroid = identity.load_profiles(cfg, alias, scorers=("sface",))["sface"].centroid
    dataset = root / "dataset" / "images"
    if dataset.exists() and any(dataset.iterdir()):
        if not replace:
            raise ValueError(f"{dataset} exists; pass --replace to build it again")
        shutil.rmtree(dataset)
    trigger = trigger or f"zq{alias}"
    photos, _ = refs.gather([root / "photos"], limit=2000)
    videos = sorted(p for p in (root / "video").glob("*") if p.suffix.lower() in VIDEO_TYPES) \
        if (root / "video").is_dir() else []
    by_bin: dict[int, list] = {}
    work = private_dir(root / ".work")
    try:
        sources = [(photo, photo, None) for photo in photos]
        for video in videos:
            frames = _frames(cfg, video, Path(tempfile.mkdtemp(dir=work)), fps)
            sources += [(frame, video, i) for i, frame in enumerate(frames)]
            echo(f"cast  {video.name}: {len(frames)} frames")
        candidates = {}
        for n, (image, source, frame) in enumerate(sources):
            faces = tools.faces(refs.load(image))
            face, others = _the_person(faces, centroid)
            if face is None:
                continue
            key = f"{n}"
            candidates[key] = (image, source, frame, face, others)
            by_bin.setdefault(angle_bin(face.yaw), []).append((key, face))
        dataset = private_dir(dataset)
        kept, counts = 0, {w: 0 for w in WORDS}
        for b in sorted(by_bin):
            for key, face in refs.best_faces(by_bin[b], limit=per_bin):
                image, source, frame, face, others = candidates[key]
                kept += 1
                crop = refs.save_png(refs.square_crop(refs.load(image), face, others), dataset / f"{kept:03d}.png")
                words = angle_words(face.yaw)
                write_private(crop.with_suffix(".txt"), f"photo of {trigger} person, {words}, close-up\n")
                provenance.write(cfg, crop, kind="dataset", models=[], inputs=[source],
                                 params={"frame": frame, "fps": fps if frame is not None else None,
                                         "yaw": round(face.yaw, 3)})
                counts[words] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    missing = [w for w, c in counts.items() if c == 0]
    echo(f"cast  {kept} images: " + ", ".join(f"{w} {c}" for w, c in counts.items()))
    return {"kept": kept, "bins": counts, "missing": missing, "dataset": str(dataset), "trigger": trigger}
