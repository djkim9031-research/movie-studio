"""How much a face looks like one person: a profile from their real photos, and frame-by-frame scores of images and
clips. Callers check studio.provenance first: only files whose whole lineage allows face recognition are scored."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from studio import refs
from studio.config import Config
from studio.fs import private_dir, write_private

MIN_FACE_PX = 100
# Nose offset from the eye midpoint, in eye distances; beyond this a face is too turned to score. Calibrated to
# degrees on a real head-turn clip once the user has filmed one.
MAX_YAW = 0.5
PIN_MARGIN = 2
SMOOTH_S = 0.3
DIP_S = 0.25
MAX_PHOTOS = 500


@dataclass
class Profile:
    alias: str
    scorer: str
    embeddings: np.ndarray
    sources: list[str]
    genuine: np.ndarray  # similarities of every pair of the person's real faces
    yaws: list[float]

    @property
    def centroid(self) -> np.ndarray:
        mean = self.embeddings.mean(axis=0)
        return mean / np.linalg.norm(mean)

    @property
    def digest(self) -> str:
        data = self.scorer.encode() + np.ascontiguousarray(self.embeddings, dtype=np.float32).tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    @property
    def genuine_p5(self) -> float:
        return float(np.percentile(self.genuine, 5))

    @property
    def genuine_median(self) -> float:
        return float(np.median(self.genuine))

    @property
    def genuine_sd(self) -> float:
        return float(np.std(self.genuine))

    def without(self, source: str) -> "Profile":
        """The profile without one photo, so that photo can be scored as unseen."""
        keep = [i for i, s in enumerate(self.sources) if s != source]
        embeddings = self.embeddings[keep]
        sims = embeddings @ embeddings.T
        return Profile(alias=self.alias, scorer=self.scorer, embeddings=embeddings,
                       sources=[self.sources[i] for i in keep],
                       genuine=np.sort(sims[np.triu_indices(len(keep), 1)]), yaws=[self.yaws[i] for i in keep])


@dataclass
class Frame:
    t: float
    sim: float | None
    yaw: float | None
    px: float | None


@dataclass
class Summary:
    median: float | None
    mid_min: float | None
    dips: int
    dip_seconds: float
    coverage: float
    yaw_range: float
    scored: int


@dataclass
class ClipScore:
    frames: list[Frame]
    summary: Summary


def build_profile(cfg: Config, alias: str, tools, leave_out=()) -> Profile:
    """The person in most of cast/ALIAS/photos; `leave_out` names photos kept out, such as bench start frames."""
    pictures, _ = refs.gather([cfg.data_root / "cast" / alias / "photos"], limit=MAX_PHOTOS)
    pictures = [p for p in pictures if p.name not in set(leave_out)]
    faces = {p.name: tools.faces(refs.load(p)) for p in pictures}
    subject = refs.find_subject(faces)
    if len(subject) < 2:
        raise ValueError(f"{len(subject)} usable photos of {alias}; a profile needs at least 2")
    embeddings = np.stack([f.embedding for _, f in subject])
    sims = embeddings @ embeddings.T
    genuine = np.sort(sims[np.triu_indices(len(subject), 1)])
    return Profile(alias=alias, scorer="sface", embeddings=embeddings, sources=[name for name, _ in subject],
                   genuine=genuine, yaws=[round(f.yaw, 3) for _, f in subject])


def _profile_dir(cfg: Config, alias: str) -> Path:
    return cfg.data_root / "cast" / alias / "profile"


def save_profile(cfg: Config, profile: Profile) -> Path:
    """Face embeddings are biometric data: owner-only files in the person's folder."""
    folder = private_dir(_profile_dir(cfg, profile.alias))
    arrays = io.BytesIO()
    np.savez(arrays, embeddings=profile.embeddings, genuine=profile.genuine, yaws=np.array(profile.yaws))
    write_private(folder / f"{profile.scorer}.npz", arrays.getvalue())
    meta = {"alias": profile.alias, "scorer": profile.scorer, "digest": profile.digest, "sources": profile.sources,
            "genuine": {"pairs": len(profile.genuine), "p5": round(profile.genuine_p5, 4),
                        "median": round(profile.genuine_median, 4)}}
    write_private(folder / f"{profile.scorer}.json", (json.dumps(meta, indent=1) + "\n").encode())
    return folder / f"{profile.scorer}.npz"


def load_profile(cfg: Config, alias: str, scorer: str = "sface") -> Profile:
    folder = _profile_dir(cfg, alias)
    if not (folder / f"{scorer}.npz").is_file():
        raise FileNotFoundError(f"no {scorer} profile of {alias}; run: uv run studio identity profile --person {alias}")
    meta = json.loads((folder / f"{scorer}.json").read_text(encoding="utf-8"))
    with np.load(folder / f"{scorer}.npz") as arrays:
        return Profile(alias=alias, scorer=scorer, embeddings=arrays["embeddings"], sources=meta["sources"],
                       genuine=arrays["genuine"], yaws=arrays["yaws"].tolist())


def percentile(profile: Profile, sim: float) -> float:
    """Share of the person's genuine pairs scoring at or below `sim`, in percent."""
    return float(np.mean(profile.genuine <= sim) * 100)


def pinned_mask(n: int, pinned: list[int], margin: int = PIN_MARGIN) -> list[bool]:
    """Frames copied from a conditioning image (index -1 is the last frame), and `margin` frames on each side."""
    mask = [False] * n
    for index in pinned:
        centre = index % n
        for i in range(max(centre - margin, 0), min(centre + margin + 1, n)):
            mask[i] = True
    return mask


def rolling_median(times: list[float], values: list[float], window_s: float) -> list[float]:
    t, v = np.asarray(times), np.asarray(values)
    return [float(np.median(v[np.abs(t - ti) <= window_s / 2])) for ti in t]


def summarize(frames: list[Frame], pinned: list[int], genuine_p5: float, min_px: float = MIN_FACE_PX,
              max_yaw: float = MAX_YAW) -> Summary:
    mask = pinned_mask(len(frames), pinned)
    candidates = [f for f, pin in zip(frames, mask) if not pin]
    scored = [f for f in candidates if f.sim is not None and f.px >= min_px and abs(f.yaw) <= max_yaw]
    coverage = len(scored) / len(candidates) if candidates else 0.0
    if not scored:
        return Summary(None, None, 0, 0.0, coverage, 0.0, 0)
    times = [f.t for f in scored]
    smooth = rolling_median(times, [f.sim for f in scored], SMOOTH_S)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    dips, dip_seconds, start = 0, 0.0, None
    for i, value in enumerate(smooth + [np.inf]):
        if value < genuine_p5 and start is None:
            start = i
        elif value >= genuine_p5 and start is not None:
            length = times[i - 1] - times[start] + step
            if length >= DIP_S:
                dips, dip_seconds = dips + 1, dip_seconds + length
            start = None
    yaws = [f.yaw for f in scored]
    return Summary(median=float(np.median(smooth)), mid_min=float(min(smooth)), dips=dips,
                   dip_seconds=round(dip_seconds, 3), coverage=coverage, yaw_range=float(max(yaws) - min(yaws)),
                   scored=len(scored))


def _frame(profile: Profile, tools, bgr: np.ndarray, t: float) -> Frame:
    faces = tools.faces(bgr)
    if not faces or faces[0].embedding is None:
        return Frame(t, None, None, None)
    face = faces[0]
    return Frame(t, float(face.embedding @ profile.centroid), face.yaw, face.box[3])


def score_image(profile: Profile, tools, path: Path) -> Frame:
    return _frame(profile, tools, refs.load(Path(path)), 0.0)


def score_clip(profile: Profile, tools, path: Path, pinned: list[int]) -> ClipScore:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot read {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frames, index = [], 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(_frame(profile, tools, bgr, index / fps))
            index += 1
    finally:
        capture.release()
    return ClipScore(frames=frames, summary=summarize(frames, pinned, profile.genuine_p5))
