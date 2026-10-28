"""Numbers that sort takes without looking at faces: how much moves, how long a take stands still at its head and
tail, and how far its colors drift from the first frames to the last. Computed on small frames, CPU only; they sort
takes for the user's pick and never make it."""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from studio.config import Config
from studio.media import ffmpeg

WIDTH = 160  # pixels: enough for motion and color, small enough to be quick
FROZEN_BELOW = 0.6  # mean absolute difference between frames, in 8-bit gray levels, under which nothing moved
ENDS = 4  # frames averaged at each end for the color drift


def frames(cfg: Config, clip: Path) -> np.ndarray:
    """The clip's RGB frames, (n, h, w, 3), scaled to WIDTH pixels wide."""
    info = ffmpeg.probe(cfg, clip)
    height = max(2, round(info["height"] * WIDTH / info["width"] / 2) * 2)
    raw = subprocess.run([str(ffmpeg.binary(cfg)), "-v", "error", "-i", str(clip), "-vf",
                          f"scale={WIDTH}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, WIDTH, 3)


def clip(cfg: Config, path: Path, fps: int) -> dict:
    """motion: mean over frame pairs of the 90th percentile of optical flow, in pixels per frame at WIDTH; still_head_s
    and still_tail_s: how long nothing moves at each end; color_drift: CIE76 distance between the mean colors of the
    first and the last ENDS frames."""
    video = frames(cfg, path)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in video]
    pairs = list(zip(gray, gray[1:]))
    moved = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) >= FROZEN_BELOW for a, b in pairs]
    if any(moved):
        head, tail = moved.index(True), moved[::-1].index(True)
    else:
        head = tail = len(video)
    flow = [np.percentile(np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2),
                          90) for a, b in pairs]
    lab = [cv2.cvtColor(part.reshape(-1, WIDTH, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
           .mean(axis=0) for part in (video[:ENDS], video[-ENDS:])]
    return {"motion": round(float(np.mean(flow)) if flow else 0.0, 2), "still_head_s": round(head / fps, 2),
            "still_tail_s": round(tail / fps, 2), "color_drift": round(float(np.linalg.norm(lab[0] - lab[1])), 1)}
