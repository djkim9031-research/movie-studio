"""Keyframe graphs for the Apache-2.0 track, built in code from ComfyUI's official templates: Qwen-Image-Edit-2511
(image_qwen_image_edit_2511, 1 to 3 references) and FLUX.2 [klein] 4B (image_flux2_klein_image_edit_4b_distilled,
1 to 4 references). Images from these may be face-scored; FLUX 3 Klein's may not. LoadImage nodes hold placeholder
names for video_graphs.render_graph()."""


def qwen_edit(*, prompt: str, refs: list[str], size: tuple[int, int], seed: int, prefix: str, steps: int = 40,
              cfg: float = 4.0, shift: float = 3.1) -> dict:
    """The template's full-quality setting: 40 steps at cfg 4, no 4-step LoRA. The vision-language text encoder sees
    the references; only image 1 is scaled first, as in the template."""
    if not 1 <= len(refs) <= 3:
        raise ValueError(f"Qwen-Image-Edit takes 1 to 3 references, got {len(refs)}")
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_bf16.safetensors",
                                                      "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b.safetensors", "type": "qwen_image",
                                                      "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift,
                                                                "sampling": "flow"}},
        "5": {"class_type": "CFGNorm", "inputs": {"model": ["4", 0], "strength": 1.0, "pre_cfg": False}},
    }
    images = {}
    for i, ref in enumerate(refs, start=1):
        g[f"1{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        images[f"image{i}"] = [f"1{i}", 0]
    g["20"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["11", 0]}}
    images["image1"] = ["20", 0]
    for node, text in (("30", prompt), ("31", "")):
        g[node] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": text,
                                                                           "vae": ["3", 0], **images}}
    g["32"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["30", 0], "reference_latents_method": "index_timestep_zero"}}
    g["33"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["31", 0], "reference_latents_method": "index_timestep_zero"}}
    g["40"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["41"] = {"class_type": "KSampler", "inputs": {"model": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                    "sampler_name": "euler", "scheduler": "simple",
                                                    "positive": ["32", 0], "negative": ["33", 0],
                                                    "latent_image": ["40", 0], "denoise": 1.0}}
    g["42"] = {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["3", 0]}}
    g["43"] = {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": prefix}}
    return g


def flux2_klein(*, prompt: str, refs: list[str], size: tuple[int, int], seed: int, prefix: str,
                steps: int = 4) -> dict:
    """The distilled 4B in 4 steps at cfg 1; each reference scaled to 1 MP, encoded and attached to both sides."""
    if not 1 <= len(refs) <= 4:
        raise ValueError(f"FLUX.2 klein 4B takes 1 to 4 references, got {len(refs)}")
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors",
                                                      "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2",
                                                      "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
    }
    positive, negative = ["4", 0], ["5", 0]
    for i, ref in enumerate(refs):
        g[f"1{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        g[f"2{i}"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [f"1{i}", 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1}}
        g[f"3{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"2{i}", 0], "vae": ["3", 0]}}
        g[f"4{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [f"3{i}", 0]}}
        g[f"5{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [f"3{i}", 0]}}
        positive, negative = [f"4{i}", 0], [f"5{i}", 0]
    g.update({
        "60": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": positive, "negative": negative,
                                                     "cfg": 1.0}},
        "61": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "62": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "63": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": size[0], "height": size[1]}},
        "64": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1],
                                                                 "batch_size": 1}},
        "65": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["61", 0], "guider": ["60", 0],
                                                                 "sampler": ["62", 0], "sigmas": ["63", 0],
                                                                 "latent_image": ["64", 0]}},
        "66": {"class_type": "VAEDecode", "inputs": {"samples": ["65", 0], "vae": ["3", 0]}},
        "67": {"class_type": "SaveImage", "inputs": {"images": ["66", 0], "filename_prefix": prefix}},
    })
    return g
