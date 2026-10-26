"""Image-to-video through the exported LTX-2.5 workflow: a still and a prompt in, a clip with sound out."""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from studio.config import Config
from studio.fs import copy_into, copy_out, private_dir
from studio.gpu import runtime
from studio.gpu.comfy import RenderError, output_files
from studio.gpu.vram import VramSampler
from studio.media import ffmpeg
from studio.workflow import fill, strip_prompt_enhancer

WORKFLOWS = Path(__file__).resolve().parent.parent / "config" / "workflows"
# LTX-2.5 sizes: sides in multiples of 32, about 0.9 megapixels.
LANDSCAPE = (1280, 704)
PORTRAIT = (704, 1280)


def target_size(width: int, height: int) -> tuple[int, int]:
    return PORTRAIT if height > width else LANDSCAPE


def image_size(cfg: Config, path: Path) -> tuple[int, int]:
    result = subprocess.run([str(ffmpeg.binary(cfg, "ffprobe")), "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=True)
    width, height = result.stdout.strip().split(",")[:2]
    return int(width), int(height)


def prepare_still(cfg: Config, picture: Path, out: Path, size: tuple[int, int]) -> Path:
    """Center-crops `picture` to the aspect of `size` and scales it to exactly `size`."""
    width, height = size
    ffmpeg.run(cfg, "-y", "-v", "error", "-i", str(picture),
               "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
               "-frames:v", "1", str(out)).check_returncode()
    return Path(out)


@dataclass
class Render:
    clip: Path
    elapsed_s: float
    peak_gib: float | None


def render_i2v(cfg: Config, still: Path, prompt: str, seconds: int, seed: int, size: tuple[int, int],
               name: str, dest: Path) -> Render:
    """One render with `still` (already `size`) as the first frame, saved to `dest`.

    ComfyUI's folders are writable from the container, so files go in and come out without following links."""
    input_name = f"{name}.png"
    try:
        copy_into(still, private_dir(cfg.comfy_io_dir / "input"), input_name)
    except (ValueError, OSError) as error:
        raise RenderError(f"unsafe input folder: {error}") from None
    workflow = fill(strip_prompt_enhancer(json.loads((WORKFLOWS / "ltx25_i2v.json").read_text())),
                    json.loads((WORKFLOWS / "ltx25_i2v.slots.json").read_text()),
                    {"prompt": prompt, "image": input_name, "seconds": seconds, "width": size[0],
                     "height": size[1], "seed": seed, "prefix": f"studio/{name}"})
    started = time.monotonic()
    with VramSampler() as vram:
        entry = runtime.client(cfg).run(workflow, timeout_s=1800)
    elapsed = time.monotonic() - started
    videos = [f for f in output_files(entry) if f[1].endswith(".mp4")]
    if not videos:
        raise RenderError(f"the run finished without an MP4: {output_files(entry)}")
    subfolder, filename, _ = videos[0]
    try:
        copy_out(cfg.comfy_io_dir, ["output", *Path(subfolder).parts, filename], dest)
    except (ValueError, OSError) as error:
        raise RenderError(f"unsafe output path from ComfyUI: {subfolder}/{filename} ({error})") from None
    return Render(clip=Path(dest), elapsed_s=elapsed,
                  peak_gib=vram.peak_mib / 1024 if vram.peak_mib is not None else None)
