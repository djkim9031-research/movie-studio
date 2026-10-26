"""Reference pictures for a scene: the person in most of the pictures as square face crops, other pictures whole.

Only the input pictures come here. Generated images never do: the FLUX license (4.a(iv)) forbids biometric
processing of the model's outputs.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, features

from studio.fs import private_dir

# SFace cosine similarity at or above which two faces are taken as the same person (OpenCV's published threshold).
SAME_PERSON = 0.363
# Pairs this similar are near-copies of one photo and add nothing as a second reference.
NEAR_COPY = 0.9
MAX_REFERENCES = 10
MAX_PERSON = 6
MAX_PICTURES = 60
PICTURE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"} | (
    {".avif"} if features.check("avif") else set())
HEIC_TYPES = {".heic", ".heif"}
CROP_SIDE = 1024
EXTRA_LONG_SIDE = 2048


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, width, height in picture pixels
    landmarks: np.ndarray  # right eye, left eye, nose tip, right and left mouth corners
    score: float
    embedding: np.ndarray | None  # SFace
    sharpness: float
    extra: dict = field(default_factory=dict)  # other recognizers' embeddings by name

    @property
    def yaw(self) -> float:
        """Nose offset from the eye midpoint in eye distances; 0 is frontal."""
        right_eye, left_eye, nose = self.landmarks[0], self.landmarks[1], self.landmarks[2]
        distance = max(float(np.linalg.norm(left_eye - right_eye)), 1e-6)
        return float((nose[0] - (right_eye[0] + left_eye[0]) / 2) / distance)


@dataclass
class Reference:
    kind: str  # person or extra
    file: Path  # what the image model gets
    square: Path  # what the prompt upsampler gets: its image batch crops everything to the first image's square
    source: str
    face_px: int = 0
    sharpness: float = 0.0


@dataclass
class Report:
    used: list[Reference] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def people(self) -> int:
        return sum(1 for r in self.used if r.kind == "person")


# ArcFace's five-point layout of a 112 px face crop: eyes, nose tip, mouth corners, image left to right.
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)


def align_112(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), ARCFACE_112, method=cv2.LMEDS)
    return cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0)


class AuraFace:
    """AuraFace v1 (Apache-2.0, ArcFace-style): 512 numbers per face from a 112 px crop aligned on the landmarks."""

    def __init__(self, model: Path):
        self.net = cv2.dnn.readNetFromONNX(str(model))

    def embed(self, bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(align_112(bgr, landmarks), 1 / 127.5, (112, 112), (127.5, 127.5, 127.5),
                                     swapRB=True)
        self.net.setInput(blob)
        vector = self.net.forward().flatten()
        return vector / np.linalg.norm(vector)


def add_extra(bgr: np.ndarray, faces: list, recognizers: dict) -> list:
    for face in faces:
        for name, recognizer in recognizers.items():
            face.extra[name] = recognizer.embed(bgr, face.landmarks)
    return faces


class FaceTools:
    """YuNet finds faces and five landmarks; SFace turns each face into 128 numbers, and any `extra` recognizers
    (by name, such as AuraFace) add their own embeddings."""

    def __init__(self, detector: Path, recognizer: Path, min_score: float = 0.7, extra: dict | None = None):
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        self.detector = str(detector)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.min_score = min_score
        self.extra = extra or {}

    def faces(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        detector = cv2.FaceDetectorYN.create(self.detector, "", (small.shape[1], small.shape[0]), self.min_score, 0.3,
                                             5000)
        _, rows = detector.detect(small)
        faces = []
        for row in [] if rows is None else rows:
            row = row.astype(np.float64)
            row[:14] /= scale
            feature = self.recognizer.feature(self.recognizer.alignCrop(bgr, row.astype(np.float32))).flatten()
            faces.append(Face(box=tuple(float(v) for v in row[:4]), landmarks=row[4:14].reshape(5, 2),
                              score=float(row[14]), embedding=feature / np.linalg.norm(feature),
                              sharpness=sharpness(bgr, row[:4])))
        return add_extra(bgr, sorted(faces, key=lambda f: -f.box[2] * f.box[3]), self.extra)


def sharpness(bgr: np.ndarray, box) -> float:
    """Variance of the Laplacian of the face at 256 px; higher is sharper."""
    x, y, w, h = (int(round(v)) for v in box)
    H, W = bgr.shape[:2]
    region = bgr[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
    if region.size == 0:
        return 0.0
    gray = cv2.resize(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_subject(faces: dict[str, list[Face]]) -> list[tuple[str, Face]]:
    """The identity seen in the most pictures, one face per picture, by average-linkage grouping of embeddings.
    Two faces of one picture are never merged."""
    items = [(name, f) for name, fs in faces.items() for f in fs if f.embedding is not None]
    if not items:
        return []
    clusters = [[i] for i in range(len(items))]
    sims = np.stack([f.embedding for _, f in items]) @ np.stack([f.embedding for _, f in items]).T
    while True:
        best, pair = SAME_PERSON, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if {items[i][0] for i in clusters[a]} & {items[i][0] for i in clusters[b]}:
                    continue
                average = float(sims[np.ix_(clusters[a], clusters[b])].mean())
                if average >= best:
                    best, pair = average, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] += clusters.pop(b)

    def weight(cluster):
        return (len({items[i][0] for i in cluster}), sum(items[i][1].box[2] * items[i][1].box[3] for i in cluster))

    subject = max(clusters, key=weight)
    order = list(faces)
    return sorted(((items[i][0], items[i][1]) for i in subject), key=lambda item: order.index(item[0]))


def best_faces(candidates: list[tuple[str, Face]], limit: int = MAX_PERSON) -> list[tuple[str, Face]]:
    """Sharpest, largest and most frontal first; near-copies of a chosen face are skipped."""
    if not candidates:
        return []
    top_sharp = max(f.sharpness for _, f in candidates) or 1.0
    top_size = max(f.box[3] for _, f in candidates) or 1.0

    def score(face: Face) -> float:
        frontal = 1 - min(abs(face.yaw) / 0.5, 1.0)
        return 0.5 * face.sharpness / top_sharp + 0.3 * face.box[3] / top_size + 0.2 * frontal

    chosen = []
    for name, face in sorted(candidates, key=lambda item: -score(item[1])):
        if any(float(face.embedding @ other.embedding) > NEAR_COPY for _, other in chosen):
            continue
        chosen.append((name, face))
        if len(chosen) == limit:
            break
    return chosen


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def crop_window(shape, face: Face, others: list[Face]) -> tuple[int, int, int, int]:
    """A square around the face, head and shoulders, shrunk step by step to leave out other faces."""
    H, W = shape[:2]
    x, y, w, h = face.box
    cx, cy = x + w / 2, y + h / 2
    window = None
    for factor in (2.4, 2.2, 2.0, 1.8, 1.6):
        side = int(min(h * factor, W, H))
        x0 = int(min(max(cx - side / 2, 0), W - side))
        y0 = int(min(max(cy - side * 0.45, 0), H - side))
        window = (x0, y0, x0 + side, y0 + side)
        if not any(_overlaps(window, (o.box[0], o.box[1], o.box[0] + o.box[2], o.box[1] + o.box[3])) for o in others):
            break
    return window


def face_frame(shape, face: Face, aspect: float = 16 / 9, face_frac: float = 0.35,
               eyes_at: float = 0.4) -> tuple[int, int, int, int]:
    """A window of `aspect` around the face: the face `face_frac` of its height when the picture allows (otherwise
    as wide as the picture), the eyes `eyes_at` of the way down, the window inside the picture."""
    H, W = shape[:2]
    x, y, w, h = face.box
    height = min(h / face_frac, H, W / aspect)
    width = height * aspect
    eyes_y = float((face.landmarks[0][1] + face.landmarks[1][1]) / 2)
    x0 = min(max(x + w / 2 - width / 2, 0), W - width)
    y0 = min(max(eyes_y - eyes_at * height, 0), H - height)
    return round(x0), round(y0), round(x0 + width), round(y0 + height)


def square_crop(bgr: np.ndarray, face: Face, others: list[Face]) -> np.ndarray:
    x0, y0, x1, y1 = crop_window(bgr.shape, face, others)
    crop = bgr[y0:y1, x0:x1].copy()
    for o in others:
        ox, oy, ow, oh = o.box
        bx0, by0 = max(int(ox - 0.25 * ow) - x0, 0), max(int(oy - 0.35 * oh) - y0, 0)
        bx1, by1 = min(int(ox + 1.25 * ow) - x0, crop.shape[1]), min(int(oy + 1.25 * oh) - y0, crop.shape[0])
        if bx1 > bx0 and by1 > by0:
            region = crop[by0:by1, bx0:bx1]
            k = max(51, (max(region.shape[:2]) // 2) * 2 + 1)
            crop[by0:by1, bx0:bx1] = cv2.GaussianBlur(region, (k, k), 0)
    method = cv2.INTER_AREA if crop.shape[0] >= CROP_SIDE else cv2.INTER_LANCZOS4
    return cv2.resize(crop, (CROP_SIDE, CROP_SIDE), interpolation=method)


def extra_versions(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The picture at its own aspect (long side at most 2048) and a black-padded 1024 square of it."""
    h, w = bgr.shape[:2]
    scale = min(1.0, EXTRA_LONG_SIDE / max(h, w))
    full = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    fit = CROP_SIDE / max(h, w)
    small = cv2.resize(bgr, (max(1, round(w * fit)), max(1, round(h * fit))),
                       interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_LANCZOS4)
    square = np.zeros((CROP_SIDE, CROP_SIDE, 3), dtype=bgr.dtype)
    top, left = (CROP_SIDE - small.shape[0]) // 2, (CROP_SIDE - small.shape[1]) // 2
    square[top:top + small.shape[0], left:left + small.shape[1]] = small
    return full, square


def load(path: Path) -> np.ndarray:
    """BGR pixels, turned upright as the camera recorded."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(rgb)[:, :, ::-1].copy()


def save_png(bgr: np.ndarray, path: Path) -> Path:
    """PNG without the source's metadata (camera, place, time)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        Image.fromarray(bgr[:, :, ::-1]).save(f, format="PNG")
    return path


def gather(paths: list[Path], limit: int = MAX_PICTURES) -> tuple[list[Path], list[tuple[str, str]]]:
    """Pictures from files and folders: every picture directly inside a folder, in name order, each picture once.
    Hidden files, subfolders and other files are left out; HEIC files are reported."""
    pictures, skipped, seen = [], [], set()

    def take(path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in HEIC_TYPES:
            skipped.append((path.name, "HEIC is not supported yet; export it as JPEG"))
        elif suffix in PICTURE_TYPES and path.resolve() not in seen:
            seen.add(path.resolve())
            pictures.append(path)

    for path in map(Path, paths):
        if path.is_dir():
            for child in sorted(path.iterdir(), key=lambda c: c.name.lower()):
                if child.is_file() and not child.name.startswith("."):
                    take(child)
        elif path.is_file():
            if path.suffix.lower() not in PICTURE_TYPES | HEIC_TYPES:
                skipped.append((path.name, "not a picture"))
            take(path)
        else:
            raise ValueError(f"no picture or folder at {path}")
    if len(pictures) > limit:
        raise ValueError(f"{len(pictures)} pictures given; at most {limit} are read")
    return pictures, skipped


def prepare(pictures: list[Path], out: Path, tools) -> Report:
    """Faces are found picture by picture; only the pictures used are read again, so a folder of photos never sits
    in memory at once."""
    if not pictures:
        raise ValueError("no pictures given")
    out = private_dir(out)
    keys = [f"{i}:{Path(p).name}" for i, p in enumerate(pictures)]
    paths = dict(zip(keys, map(Path, pictures)))
    faces = {key: tools.faces(load(paths[key])) for key in keys}
    subject = find_subject(faces)
    chosen = best_faces(subject)
    chosen_keys = {key for key, _ in chosen}
    subject_keys = {key for key, _ in subject}
    report = Report()
    for n, (key, face) in enumerate(sorted(chosen, key=lambda item: keys.index(item[0])), start=1):
        others = [f for f in faces[key] if f is not face]
        file = save_png(square_crop(load(paths[key]), face, others), out / f"person{n}.png")
        report.used.append(Reference("person", file, file, key.split(":", 1)[1], int(face.box[3]),
                                     round(face.sharpness, 1)))
    extras = 0
    for key in keys:
        name = key.split(":", 1)[1]
        if key in chosen_keys:
            continue
        if faces[key] and key in subject_keys:
            report.skipped.append((name, f"not among the {MAX_PERSON} best faces"))
        elif faces[key]:
            report.skipped.append((name, "only other people"))
        elif len(report.used) >= MAX_REFERENCES:
            report.skipped.append((name, f"over the {MAX_REFERENCES}-reference limit"))
        else:
            extras += 1
            full, square = extra_versions(load(paths[key]))
            report.used.append(Reference("extra", save_png(full, out / f"extra{extras}.png"),
                                         save_png(square, out / f"extra{extras}_square.png"), name))
    return report
