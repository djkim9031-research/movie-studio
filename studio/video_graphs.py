"""Video graphs built in code from ComfyUI's official templates: LTX-2.5 single-stage with guide frames
(video_ltx2_5_flf2v) and Wan 2.2 A14B image-to-video (video_wan2_2_14B_i2v and video_wan2_2_14B_flf2v).
LoadImage nodes hold placeholder names that render_graph() replaces with the copied images."""

import time
from dataclasses import dataclass
from pathlib import Path

from studio.config import Config
from studio.fs import copy_into, copy_out, private_dir, remove_in, safe_parts
from studio.gpu import runtime
from studio.gpu.comfy import RenderError, cached_nodes
from studio.gpu.vram import VramSampler

# The distilled schedule of the template: 8 steps.
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
# Wan 2.2's own negative prompt (sample_neg_prompt in its configs); it only acts when cfg is above 1.
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
                "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
                "三条腿，背景人很多，倒着走")
WAN_LORAS_1022 = ("wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                  "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors")


def ltx_single(*, prompt: str, first: str, size: tuple[int, int], seconds: int, seed: int, unet: str, prefix: str,
               last: str | None = None, first_strength: float = 1.0, last_strength: float = 1.0, fps: int = 24,
               img_compression: int = 18) -> dict:
    """One pass at full size: the first frame (and optionally the last) placed as guides, 8 distilled steps."""
    frames = seconds * fps + 1
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "type": "ltxv",
            "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0],
                                                           "frame_rate": float(fps)}},
        "10": {"class_type": "LoadImage", "inputs": {"image": first}},
        "11": {"class_type": "LTXVPreprocess", "inputs": {"image": ["10", 0], "img_compression": img_compression}},
        "12": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": size[0], "height": size[1],
                                                                "length": frames, "batch_size": 1}},
        "13": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0],
                                                        "latent": ["12", 0], "image": ["11", 0], "frame_idx": 0,
                                                        "strength": first_strength}},
    }
    guided = "13"
    if last is not None:
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": last}}
        g["21"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": img_compression}}
        g["22"] = {"class_type": "LTXVAddGuide", "inputs": {"positive": ["13", 0], "negative": ["13", 1],
                                                            "vae": ["3", 0], "latent": ["13", 2], "image": ["21", 0],
                                                            "frame_idx": -1, "strength": last_strength}}
        guided = "22"
    g.update({
        "30": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps,
                                                                "batch_size": 1, "audio_vae": ["4", 0]}},
        "31": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": [guided, 2], "audio_latent": ["30", 0]}},
        "32": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["1", 0], "positive": [guided, 0],
                                                             "negative": [guided, 1], "video_cfg": 1.0,
                                                             "audio_cfg": 1.0}},
        "33": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "34": {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 0.0, "s_noise": 1.0}},
        "35": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS}},
        "36": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["33", 0], "guider": ["32", 0],
                                                                 "sampler": ["34", 0], "sigmas": ["35", 0],
                                                                 "latent_image": ["31", 0]}},
        "37": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["36", 1]}},
        "38": {"class_type": "LTXVCropGuides", "inputs": {"positive": [guided, 0], "negative": [guided, 1],
                                                          "latent": ["37", 0]}},
        "39": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["38", 2], "vae": ["3", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 64,
                                                          "temporal_overlap": 16}},
        "40": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["37", 1], "audio_vae": ["4", 0]}},
        "41": {"class_type": "CreateVideo", "inputs": {"images": ["39", 0], "audio": ["40", 0], "fps": float(fps)}},
        "42": {"class_type": "SaveVideo", "inputs": {"video": ["41", 0], "filename_prefix": prefix, "format": "auto",
                                                     "codec": "auto"}},
    })
    return g


def wan_i2v(*, prompt: str, start: str, size: tuple[int, int], seconds: int, seed: int, steps: int, cfg: float,
            split: int, loras: tuple[str, str] | None, prefix: str, end: str | None = None, fps: int = 16,
            shift: float = 5.0, negative: str = WAN_NEGATIVE) -> dict:
    """Both experts in fp16: the high-noise one for steps [0, split), the low-noise one for the rest."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp16.safetensors",
                                                      "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp16.safetensors",
                                                      "weight_dtype": "default"}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan",
                                                      "device": "default"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": negative}},
        "11": {"class_type": "LoadImage", "inputs": {"image": start}},
    }
    high, low = ["1", 0], ["2", 0]
    if loras:
        g["3"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": high, "lora_name": loras[0],
                                                                  "strength_model": 1.0}}
        g["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": low, "lora_name": loras[1],
                                                                  "strength_model": 1.0}}
        high, low = ["3", 0], ["4", 0]
    g["5"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": high, "shift": shift}}
    g["6"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": low, "shift": shift}}
    frames = {"positive": ["9", 0], "negative": ["10", 0], "vae": ["8", 0], "width": size[0], "height": size[1],
              "length": int(seconds * fps + 1), "batch_size": 1, "start_image": ["11", 0]}
    if end is None:
        g["13"] = {"class_type": "WanImageToVideo", "inputs": frames}
    else:
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": end}}
        g["13"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {**frames, "end_image": ["12", 0]}}
    g["14"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["5", 0], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["13", 0], "negative": ["13", 1],
        "latent_image": ["13", 2], "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}}
    g["15"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["6", 0], "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["13", 0], "negative": ["13", 1],
        "latent_image": ["14", 0], "start_at_step": split, "end_at_step": steps,
        "return_with_leftover_noise": "disable"}}
    g["16"] = {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}}
    g["17"] = {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": float(fps)}}
    g["18"] = {"class_type": "SaveVideo", "inputs": {"video": ["17", 0], "filename_prefix": prefix,
                                                     "format": "auto", "codec": "auto"}}
    return g


class NoSampler:
    """Stands in for VramSampler where GPU memory is not measured."""

    peak_mib = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


@dataclass
class Rendered:
    clip: Path
    elapsed_s: float
    peak_gib: float | None


def render_graph(cfg: Config, graph: dict, images: dict[str, Path], *, name: str, dest: Path,
                 timeout_s: float = 3600) -> Rendered:
    """Copies `images` in for the LoadImage placeholders, runs the graph, copies its video or image out to `dest`,
    and leaves nothing of this run in ComfyUI's folders."""
    loads = [node for node in graph.values() if node["class_type"] == "LoadImage"]
    missing = [node["inputs"]["image"] for node in loads if node["inputs"]["image"] not in images]
    if missing:
        raise ValueError(f"no image given for {', '.join(missing)}")
    input_dir, copied = private_dir(cfg.comfy_io_dir / "input"), []
    try:
        for node in loads:
            key = node["inputs"]["image"]
            copy_into(images[key], input_dir, f"{name}_{key}.png")
            copied.append(f"{name}_{key}.png")
            node["inputs"]["image"] = f"{name}_{key}.png"
        started = time.monotonic()
        with VramSampler() as vram:
            entry = runtime.client(cfg).run(graph, timeout_s=timeout_s)
        elapsed = time.monotonic() - started
    finally:
        for file_name in copied:
            remove_in(cfg.comfy_io_dir, ["input", file_name])
    save = next(k for k, n in graph.items() if n["class_type"] in ("SaveVideo", "SaveImage"))
    items = entry.get("outputs", {}).get(save, {}).get("images", [])
    parts = [["output", *Path(i.get("subfolder", "")).parts, i["filename"]] for i in items]
    try:
        if save in cached_nodes(entry):
            raise RenderError("ComfyUI took this result from an earlier run; use another seed")
        if not parts:
            raise RenderError("the run finished without an output")
        try:
            copy_out(cfg.comfy_io_dir, parts[0], dest)
        except (ValueError, OSError) as error:
            raise RenderError(f"unsafe output path from ComfyUI: {'/'.join(parts[0])} ({error})") from None
    finally:
        for p in parts:
            if safe_parts(p):
                remove_in(cfg.comfy_io_dir, p)
    peak = vram.peak_mib / 1024 if vram.peak_mib is not None else None
    return Rendered(clip=Path(dest), elapsed_s=elapsed, peak_gib=peak)
