"""Stills from FLUX 3 Image Klein with reference pictures, built as BFL's multi-image workflow
(flux3_klein_bf16_mi2i): the prompt upsampler sees the references and writes the full caption, then the distilled
model draws in 16 steps. Reference copies and results are removed from ComfyUI's folders after each still."""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from studio.config import Config
from studio.fs import copy_into, copy_out, private_dir, remove_in, safe_parts
from studio.gpu import runtime
from studio.gpu.comfy import RenderError, cached_nodes
from studio.refs import MAX_REFERENCES, Reference

PROMPTS = Path(__file__).resolve().parent.parent / "config" / "prompts"
REQUEST = "Reference images follow in order (the first image, the second image, ...).\nUser request: {}\nOutput the caption."
LANDSCAPE = (1920, 1088)
PORTRAIT = (1088, 1920)
CAPTION_NODE = "34"
SAVE_NODE = "50"
# Finding what an edit added: a quarter-scale change, smooth colour drift (a Gaussian of DRIFT quarter pixels) taken
# out, weighted 0 at CHANGE_LO and 1 at CHANGE_HI levels, searched within the box grown by its own size on each side.
SCALE, DRIFT, CHANGE_LO, CHANGE_HI = 4, 24, 6.0, 14.0


def _upsampler(request: str, images: dict) -> dict:
    """BFL's prompt upsampler: it sees the references (`images`, a batch of squares) and writes the caption."""
    return {
        "30": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors", "type": "flux3", "device": "default"}},
        "31": {"class_type": "BatchImagesNode", "inputs": images},
        "33": {"class_type": "TextGenerate", "inputs": {
            "clip": ["30", 0], "image": ["31", 0], "prompt": REQUEST.format(request), "max_length": 3584,
            "sampling_mode": "on", "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0, "sampling_mode.repetition_penalty": 1.0,
            "sampling_mode.seed": 0, "sampling_mode.presence_penalty": 0.0, "thinking": False,
            "use_default_template": True, "mtp": "off",
            "system_prompt": (PROMPTS / "flux3_upsampler_i2i.txt").read_text(encoding="utf-8")}},
        CAPTION_NODE: {"class_type": "PreviewAny", "inputs": {"source": ["33", 0]}},
    }


class _Loads:
    """One LoadImage node per file name, numbered from 100."""

    def __init__(self, g: dict):
        self.g, self.ids = g, {}

    def __call__(self, name: str) -> str:
        if name not in self.ids:
            self.ids[name] = str(100 + len(self.ids))
            self.g[self.ids[name]] = {"class_type": "LoadImage", "inputs": {"image": name}}
        return self.ids[name]


def upsample_graph(squares: list[str], request: str) -> dict:
    """The upsampler alone: the caption it would draw from, for checking before anything is drawn."""
    if not 1 <= len(squares) <= MAX_REFERENCES:
        raise ValueError(f"FLUX 3 Klein takes 1 to {MAX_REFERENCES} references, got {len(squares)}")
    g: dict = {}
    load = _Loads(g)
    g.update(_upsampler(request, {f"images.image{i}": [load(name), 0] for i, name in enumerate(squares)}))
    return g


def _model(text, seed: int, latent: list, prefix: str, image: list | None = None) -> dict:
    """The distilled model drawing `text` (a string, or the upsampler's output) into `latent` in 16 steps; the saved
    image is the decode, or `image` when given."""
    return {
        "40": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux3-image-klein-bf16.safetensors",
                                                       "weight_dtype": "default"}},
        "41": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "flux3",
                                                       "device": "default"}},
        "42": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "43": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["40", 0], "max_shift": math.log(16),
                                                              "base_shift": math.log(16), "width": 1024,
                                                              "height": 1024}},
        "44": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["41", 0], "text": text}},
        "45": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["41", 0], "text": ""}},
        "48": {"class_type": "KSampler", "inputs": {"model": ["43", 0], "positive": ["46", 0], "negative": ["46", 1],
                                                    "latent_image": latent, "seed": seed, "steps": 16, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "49": {"class_type": "VAEDecode", "inputs": {"samples": ["48", 0], "vae": ["42", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": image or ["49", 0], "filename_prefix": prefix}},
    }


def _references(load: "_Loads", names: list[str]) -> dict:
    return {"class_type": "ReferenceMultiLatent", "inputs": {
        "positive": ["44", 0], "negative": ["45", 0], "vae": ["42", 0], "resolution": "resize",
        "resolution.megapixels": 1.0, **{f"images.image{i}": [load(name), 0] for i, name in enumerate(names)}}}


def graph(files: list[str], squares: list[str], request: str, *, seed: int, size: tuple[int, int],
          prefix: str, caption: str | None = None) -> dict:
    """`files` go to the model at their own aspect; `squares` (same order) to the upsampler's image batch. With a
    `caption` the model draws from it directly and the upsampler is left out."""
    if not 1 <= len(files) <= MAX_REFERENCES or len(squares) != len(files):
        raise ValueError(f"FLUX 3 Klein takes 1 to {MAX_REFERENCES} references, got {len(files)}")
    g = _model(caption if caption is not None else ["33", 0], seed, ["47", 0], prefix)
    g["47"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1],
                                                                  "batch_size": 1}}
    load = _Loads(g)
    if caption is None:
        g.update(_upsampler(request, {f"images.image{i}": [load(name), 0] for i, name in enumerate(squares)}))
    g["46"] = _references(load, files)
    return g


def blend_event(base: np.ndarray, edit: np.ndarray, box: list[float]) -> tuple[np.ndarray, int]:
    """`base` with what `edit` (the same frame redrawn with an event added) changed near `box` ([left, top, right,
    bottom] fractions) carried over through a soft mask, and the number of pixels it carried over. Away from the box,
    and wherever the edit only drifted in colour, the base stays exactly as it was; a sharp edge near the box that the
    edit moved by a pixel or two comes over with the event. Float RGB arrays (h, w, 3)."""
    h, w = base.shape[:2]
    sw, sh = w // SCALE, h // SCALE

    def small(a):
        return cv2.GaussianBlur(cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 1.0)

    d = small(edit) - small(base)
    drift = cv2.GaussianBlur(d, (0, 0), DRIFT)
    for _ in range(2):  # the drift again from the pixels that only drifted, so the event does not dim itself
        calm = (np.abs(d - drift).max(axis=2) < CHANGE_LO).astype(np.float32)
        drift = cv2.GaussianBlur(d * calm[..., None], (0, 0), DRIFT) / np.maximum(
            cv2.GaussianBlur(calm, (0, 0), DRIFT), 1e-3)[..., None]
    x0, y0, x1, y1 = box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh
    near = np.zeros((sh, sw), np.float32)
    near[int(max(0, 2 * y0 - y1)):int(min(sh, 2 * y1 - y0)), int(max(0, 2 * x0 - x1)):int(min(sw, 2 * x1 - x0))] = 1
    change = np.abs(d - drift).max(axis=2) * near
    weight = np.clip((change - CHANGE_LO) / (CHANGE_HI - CHANGE_LO), 0, 1)
    weight = cv2.GaussianBlur(cv2.dilate(weight, np.ones((5, 5), np.uint8)), (0, 0), 1.5) * near
    mask = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    shift = cv2.resize(drift, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.where(mask > 0, base * (1 - mask) + (edit - shift) * mask, base)
    return out, int((mask > 0.5).sum())


@dataclass
class Still:
    path: Path
    caption: str
    elapsed_s: float
    graph: dict
    edit: Path | None = None  # for a painted still: the redrawn frame its event came from
    event_px: int = 0  # for a painted still: pixels carried over from the edit


def _collect(cfg: Config, entry: dict, dest: Path) -> None:
    """The saved image copied out to `dest`, and every output removed from ComfyUI's folder."""
    images = entry.get("outputs", {}).get(SAVE_NODE, {}).get("images", [])
    parts = [["output", *Path(i.get("subfolder", "")).parts, i["filename"]] for i in images]
    try:
        if SAVE_NODE in cached_nodes(entry):
            raise RenderError("ComfyUI took this image from an earlier run, so nothing was drawn; use another seed")
        if not parts:
            raise RenderError("the run finished without an image")
        try:
            copy_out(cfg.comfy_io_dir, parts[0], dest)
        except (ValueError, OSError) as error:
            raise RenderError(f"unsafe output path from ComfyUI: {'/'.join(parts[0])} ({error})") from None
    finally:
        for p in parts:
            if safe_parts(p):
                remove_in(cfg.comfy_io_dir, p)


def upsample(cfg: Config, references: list[Reference], request: str, *, name: str, timeout_s: float = 900) -> str:
    """The caption BFL's upsampler writes for `request` from the references, without drawing."""
    input_dir = private_dir(cfg.comfy_io_dir / "input")
    copied = []
    try:
        try:
            for i, ref in enumerate(references, start=1):
                copy_into(ref.square, input_dir, f"{name}_up{i}.png")
                copied.append(f"{name}_up{i}.png")
        except (ValueError, OSError) as error:
            raise RenderError(f"unsafe input folder: {error}") from None
        entry = runtime.client(cfg).run(upsample_graph(copied, request), timeout_s=timeout_s)
    finally:
        for file_name in copied:
            remove_in(cfg.comfy_io_dir, ["input", file_name])
    text = entry.get("outputs", {}).get(CAPTION_NODE, {}).get("text") or [""]
    caption = text[0] if isinstance(text, list) else str(text)
    if not caption.strip():
        raise RenderError("the prompt upsampler wrote no caption")
    return caption


def render_still(cfg: Config, references: list[Reference], request: str, *, seed: int, size: tuple[int, int],
                 name: str, dest: Path, caption: str | None = None) -> Still:
    """With a `caption` (for example one checked after `upsample`) the model draws from it and the upsampler is not
    run again."""
    input_dir = private_dir(cfg.comfy_io_dir / "input")
    files, squares, copied = [], [], []
    try:
        try:
            for i, ref in enumerate(references, start=1):
                file_name = f"{name}_ref{i}.png"
                copy_into(ref.file, input_dir, file_name)
                copied.append(file_name)
                files.append(file_name)
                if caption is not None or ref.square == ref.file:
                    squares.append(file_name)
                else:
                    square_name = f"{name}_ref{i}_square.png"
                    copy_into(ref.square, input_dir, square_name)
                    copied.append(square_name)
                    squares.append(square_name)
        except (ValueError, OSError) as error:
            raise RenderError(f"unsafe input folder: {error}") from None
        workflow = graph(files, squares, request, seed=seed, size=size, prefix=f"scenes/{name}", caption=caption)
        started = time.monotonic()
        entry = runtime.client(cfg).run(workflow, timeout_s=1800)
        elapsed = time.monotonic() - started
    finally:
        for file_name in copied:
            remove_in(cfg.comfy_io_dir, ["input", file_name])
    _collect(cfg, entry, dest)
    if caption is None:
        text = entry.get("outputs", {}).get(CAPTION_NODE, {}).get("text") or [""]
        caption = text[0] if isinstance(text, list) else str(text)
    return Still(path=Path(dest), caption=caption, elapsed_s=elapsed, graph=workflow)


def paint_event(cfg: Config, base: Path, box: list[float], references: list[Reference], caption: str, *, seed: int,
                name: str, dest: Path) -> Still:
    """`base` with an event added near `box`: the model redraws the whole frame from `caption` (an edit instruction)
    with the base as the first image and `references` (pictures of the event) after it, then only what the edit
    added near the box is carried into the base (`blend_event`). A masked redraw of the box alone draws nothing: the
    model fills a small hole with more of what surrounds it. The redrawn frame is kept next to `dest` as -edit.png."""
    with Image.open(base) as image:
        size, pixels = image.size, np.asarray(image.convert("RGB"), dtype=np.float32)
    edit = dest.with_name(f"{dest.stem}-edit.png")
    still = render_still(cfg, [Reference("extra", base, base, base.name), *references], caption, seed=seed, size=size,
                         name=name, dest=edit, caption=caption)
    with Image.open(edit) as image:
        redrawn = np.asarray(image.convert("RGB"), dtype=np.float32)
    out, found = blend_event(pixels, redrawn, box)
    Image.fromarray(out.round().clip(0, 255).astype(np.uint8)).save(dest)
    return Still(path=Path(dest), caption=caption, elapsed_s=still.elapsed_s, graph=still.graph, edit=edit,
                 event_px=found)
