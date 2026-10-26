# Measurements

## 2026-10-26: LTX-2.5 distilled int8, image-to-video

RTX PRO 5000 72 GB, driver 580.126.09, ComfyUI v0.42.1, PyTorch 2.14.1+cu130, attention through PyTorch SDPA.
Stock LTX-2.5 image-to-video template with the prompt enhancer removed: stage 1 at 640x352 for 8 steps, x2
latent upscale, stage 2 at 1280x704 for 3 steps. 5 s clip: 121 frames at 24 fps, with audio. Start frame:
public-domain photo of a misty pond, no people.

| Run | Wall time | GPU memory peak on the card |
|---|---|---|
| Cold: weights loaded from disk | 38.4 s | 42.9 GiB |
| Warm: weights already resident | 26.9 s | not recorded |

GPU memory used: 1.4 GiB with the display alone, 1.7 GiB with ComfyUI idle, 39.2 GiB with the LTX-2.5 weights
resident, 1.4 GiB again after `studio gpu down`.
