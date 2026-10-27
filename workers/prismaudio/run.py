"""PrismAudio video-to-audio worker.

Usage: <venv>/bin/python -I run.py < job.json

One JSON job on stdin. Writes a 44.1 kHz stereo 16-bit WAV of exactly
seconds * 44100 samples and prints one JSON line on stdout; everything else
goes to stderr. Exit code 2 means a bad job, 1 any other failure.

Job keys: source, clip, prompt, seconds, seed, out, device ("cuda" or "cpu"),
weights {prismaudio, vae, synchformer, videoprism, videoprism_tokenizer, t5gemma}.
Optional: sync ("upstream" or "features"), videoprism_jit ("closure" or
"argument"), videoprism_device ("cpu" runs VideoPrism on the CPU),
text_features ("zeros", test only; t5gemma is then not needed).

The PrismAudio source stays at its pinned commit. This file puts it on
sys.path, stubs packages that are only imported, and patches a few lookups
so the run is offline. Upstream code is called as is wherever possible:
FeaturesUtils' encode methods and predict.predict_step.
"""

import hashlib
import importlib.util
import json
import os
import random
import resource
import socket
import sys
import time
import traceback
import types
import wave
from importlib.machinery import ModuleSpec

T0 = time.monotonic()
sys.dont_write_bytecode = True  # keep the source tree free of __pycache__

SAMPLE_RATE = 44100
CLIP_FPS = 4
CLIP_SIZE = 288
SYNC_FPS = 25
SYNC_SIZE = 224
MAX_SECONDS = 30
ZERO_TEXT_TOKENS = 64  # length of the test-only zero text features
VIDEOPRISM_MODEL = "videoprism_lvt_public_v1_large"
VIDEOPRISM_FILE = ("google/videoprism-lvt-large-f8r288", "flax_lvt_large_f8r288_repeated.npz")
MODEL_CONFIG = "PrismAudio/configs/model_configs/prismaudio.json"
WEIGHT_FILES = ("prismaudio", "vae", "synchformer", "videoprism", "videoprism_tokenizer")

# Upstream files this worker imports or mirrors, ThinkSound branch prismaudio at 86eed40.
SOURCE_FILES = (
    "predict.py",
    "data_utils/prismaudio_data_process.py",
    "data_utils/v2a_utils/feature_utils_288.py",
    "data_utils/v2a_utils/thinksound_288_al.py",
    "data_utils/ext/synchformer/__init__.py",
    "data_utils/ext/synchformer/synchformer.py",
    "data_utils/ext/synchformer/motionformer.py",
    "data_utils/ext/synchformer/video_model_builder.py",
    "data_utils/ext/synchformer/vit_helper.py",
    "data_utils/ext/synchformer/utils.py",
    "data_utils/ext/synchformer/divided_224_16x4.yaml",
    "PrismAudio/__init__.py",
    "PrismAudio/models/__init__.py",
    "PrismAudio/models/factory.py",
    "PrismAudio/models/diffusion.py",
    "PrismAudio/models/dit.py",
    "PrismAudio/models/transformer.py",
    "PrismAudio/models/conditioners.py",
    "PrismAudio/models/autoencoders.py",
    "PrismAudio/models/pretransforms.py",
    "PrismAudio/models/bottleneck.py",
    "PrismAudio/models/blocks.py",
    "PrismAudio/models/adp.py",
    "PrismAudio/models/utils.py",
    "PrismAudio/models/mmmodules/model/low_level.py",
    "PrismAudio/inference/__init__.py",
    "PrismAudio/inference/sampling.py",
    "PrismAudio/inference/generation.py",
    "PrismAudio/inference/utils.py",
    "PrismAudio/data/__init__.py",
    "PrismAudio/data/utils.py",
    MODEL_CONFIG,
)
SOURCE_DIGEST = "4e04b2036e467cf9a03c29018f9322013f435047f0e0fe5a28063b6c9a3073ce"


class JobError(Exception):
    """Bad job input."""


def log(msg):
    print(f"prismaudio: {msg}", file=sys.stderr, flush=True)


def claim_stdout():
    """Point fd 1 at stderr so library output cannot reach stdout; return a writer for the result line."""
    sys.stdout.flush()
    fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(fd, "w")


def parse_job(text):
    try:
        job = json.loads(text)
    except json.JSONDecodeError as e:
        raise JobError(f"job is not valid JSON ({e})") from None
    if not isinstance(job, dict):
        raise JobError("job must be a JSON object")
    for key in ("source", "clip", "prompt", "seconds", "seed", "out", "device", "weights"):
        if key not in job:
            raise JobError(f"job is missing '{key}'")
    for key in ("source", "clip", "out"):
        if not isinstance(job[key], str) or not job[key]:
            raise JobError(f"{key} must be a path string")
    if not isinstance(job["prompt"], str) or not job["prompt"].strip():
        raise JobError("prompt must be a non-empty string")
    seconds, seed = job["seconds"], job["seed"]
    if type(seconds) is not int or not 1 <= seconds <= MAX_SECONDS:
        raise JobError(f"seconds must be an integer from 1 to {MAX_SECONDS}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise JobError("seed must be an integer from 0 to 2**32 - 1")
    if job["device"] not in ("cuda", "cpu"):
        raise JobError("device must be 'cuda' or 'cpu'")
    vp_device = job.setdefault("videoprism_device", job["device"])
    if vp_device not in ("cuda", "cpu") or (job["device"] == "cpu" and vp_device == "cuda"):
        raise JobError("videoprism_device must be 'cpu', or 'cuda' on a cuda job")
    if job.setdefault("sync", "upstream") not in ("upstream", "features"):
        raise JobError("sync must be 'upstream' or 'features'")
    if job.setdefault("videoprism_jit", "closure") not in ("closure", "argument"):
        raise JobError("videoprism_jit must be 'closure' or 'argument'")
    if job.setdefault("text_features", None) not in (None, "zeros"):
        raise JobError("text_features may only be 'zeros' (test only)")
    if not os.path.isdir(job["source"]):
        raise JobError(f"source is not a directory: {job['source']}")
    if not os.path.isfile(job["clip"]):
        raise JobError(f"clip not found: {job['clip']}")
    out = job["out"]
    if not out.endswith(".wav") or not os.path.isdir(os.path.dirname(os.path.abspath(out))):
        raise JobError("out must be a .wav path in an existing directory")
    weights = job["weights"]
    if not isinstance(weights, dict):
        raise JobError("weights must be an object")
    for key in WEIGHT_FILES:
        if not isinstance(weights.get(key), str) or not os.path.isfile(weights[key]):
            raise JobError(f"weights.{key} is not a file: {weights.get(key)}")
    if job["text_features"] is None:
        t5 = weights.get("t5gemma")
        if not isinstance(t5, str) or not os.path.isfile(os.path.join(t5, "config.json")):
            raise JobError(f"weights.t5gemma is not a model directory: {t5}")
    return job


def configure_env(job):
    """Must run before torch, jax or transformers are imported."""
    if job["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cuda" if job["videoprism_device"] == "cuda" else "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ.update(USE_TF="0", USE_FLAX="0", USE_TORCH="1", TOKENIZERS_PARALLELISM="false")
    os.environ["ENABLE_TORCH_COMPILE"] = "0"  # upstream default


def block_network():
    """Refuse any IP connection or DNS lookup from this process."""

    def refuse(*args, **kwargs):
        raise OSError("network access is disabled in this worker")

    def guard(connect):
        def guarded(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                refuse()
            return connect(self, address)

        return guarded

    socket.socket.connect = guard(socket.socket.connect)
    socket.socket.connect_ex = guard(socket.socket.connect_ex)
    socket.getaddrinfo = refuse


def source_digest(src):
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        try:
            with open(os.path.join(src, rel), "rb") as f:
                data = f.read()
        except OSError:
            raise JobError(f"source is missing {rel}") from None
        h.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def check_source(src):
    if source_digest(src) != SOURCE_DIGEST:
        raise JobError("source tree does not match the pinned PrismAudio commit 86eed40")


def _module(name, package=False, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, None, is_package=package)
    if package:
        mod.__path__ = []
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, mod)
    return mod


def _missing(name):
    def fail(*args, **kwargs):
        raise RuntimeError(f"{name} is stubbed out in this worker")

    return fail


def _missing_class(name):
    return type(name.rpartition(".")[2], (), {"__init__": _missing(name)})


def _no_attributes(name):
    def getattr_(attr):
        raise AttributeError(f"{name}.{attr} is stubbed out in this worker")

    return getattr_


def install_stubs():
    """Modules that the inference path imports but never uses, plus the three dac layers it does use."""
    import torch
    from torch import nn
    from torch.nn.utils import weight_norm

    # videoprism reads its tokenizer through tf.io.gfile; local files only.
    def gfile_open(path, mode="r"):
        if "://" in str(path):
            raise RuntimeError(f"remote path refused: {path}")
        return open(path, mode)

    # Tensor and RaggedTensor appear only in annotations; to_int_tf_op is never called.
    tf_names = ("tensorflow", "tensorflow.io", "tensorflow.io.gfile")
    _module("tensorflow", package=True, Tensor=_missing_class("tensorflow.Tensor"),
            RaggedTensor=_missing_class("tensorflow.RaggedTensor"))
    _module("tensorflow.io", package=True)
    _module("tensorflow.io.gfile", GFile=gfile_open, exists=os.path.exists)
    import videoprism.models  # noqa: F401

    # einops and others probe sys.modules for tensorflow; videoprism keeps its own references.
    for name in tf_names:
        del sys.modules[name]

    # descript-audio-codec 1.0.0 dac/nn/layers.py; the Oobleck VAE uses WNConv1d and WNConvTranspose1d.
    def WNConv1d(*args, **kwargs):
        return weight_norm(nn.Conv1d(*args, **kwargs))

    def WNConvTranspose1d(*args, **kwargs):
        return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def snake(x, alpha):  # torch.jit.script dropped; Snake1d is not on this path
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)

    class Snake1d(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, x):
            return snake(x, self.alpha)

    _module("dac", package=True)
    _module("dac.nn", package=True)
    _module("dac.nn.layers", WNConv1d=WNConv1d, WNConvTranspose1d=WNConvTranspose1d, snake=snake, Snake1d=Snake1d)
    _module("dac.nn.quantize", ResidualVectorQuantize=_missing_class("dac.nn.quantize.ResidualVectorQuantize"))
    _module("dac.model", package=True)
    _module("dac.model.dac", Encoder=_missing_class("dac.model.dac.Encoder"),
            Decoder=_missing_class("dac.model.dac.Decoder"))

    _module("k_diffusion", __getattr__=_no_attributes("k_diffusion"))
    _module("vector_quantize_pytorch", ResidualVQ=_missing_class("vector_quantize_pytorch.ResidualVQ"),
            FSQ=_missing_class("vector_quantize_pytorch.FSQ"))
    _module("alias_free_torch", Activation1d=_missing_class("alias_free_torch.Activation1d"))
    _module("einops_exts", rearrange_many=_missing("einops_exts.rearrange_many"))

    # PrismAudio.training pulls wandb and lightning; conditioners only imports copy_state_dict.
    import PrismAudio  # noqa: F401

    _module("PrismAudio.training", package=True)
    _module("PrismAudio.training.utils", copy_state_dict=_missing("PrismAudio.training.utils.copy_state_dict"))

    # predict.py top-level imports; the runner does main()'s seeding itself.
    _module("prefigure", package=True)
    _module("prefigure.prefigure", get_all_args=_missing("prefigure.get_all_args"),
            push_wandb_config=_missing("prefigure.push_wandb_config"))
    _module("lightning", package=True)
    _module("lightning.pytorch", seed_everything=_missing("lightning.pytorch.seed_everything"))


def decode_frames(path, seconds):
    """rgb24 uint8 frames (T, 3, H, W) at the clip and sync rates.

    Same filter chain torio's add_basic_video_stream builds: fps=N,format=pix_fmts=rgb24.
    Like fill_buffer/pop_chunks, keeps the first int(N * seconds) frames of each stream.
    """
    import av
    import numpy as np
    import torch

    rates = (CLIP_FPS, SYNC_FPS)
    need = {rate: int(rate * seconds) for rate in rates}
    frames = {rate: [] for rate in rates}
    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = None
        if duration is not None and duration + 1e-6 < seconds:
            raise JobError(f"clip is {duration:.3f} s, shorter than {seconds} s")

        graphs = {}
        for rate in rates:
            graph = av.filter.Graph()
            graph.link_nodes(
                graph.add_buffer(template=stream),
                graph.add("fps", str(rate)),
                graph.add("format", "pix_fmts=rgb24"),
                graph.add("buffersink"),
            )
            graph.configure()
            graphs[rate] = graph

        def drain(rate):
            while True:
                try:
                    frame = graphs[rate].pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return
                if len(frames[rate]) < need[rate]:
                    frames[rate].append(frame.to_ndarray())

        for frame in container.decode(stream):
            for rate in rates:
                graphs[rate].push(frame)
                drain(rate)
            if all(len(frames[rate]) >= need[rate] for rate in rates):
                break
        else:
            for rate in rates:
                graphs[rate].push(None)
                drain(rate)

    if not all(frames.values()):
        raise JobError("clip has no decodable video frames")
    return {rate: torch.from_numpy(np.stack(frames[rate])).permute(0, 3, 1, 2).contiguous() for rate in rates}


def pad_to_square(video_tensor):
    """Copy of pad_to_square in data_utils/v2a_utils/thinksound_288_al.py."""
    import torch.nn.functional as F

    if len(video_tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (l, c, h, w)")
    l, c, h, w = video_tensor.shape
    max_side = max(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    video_padded = F.pad(video_tensor, pad=padding, mode="constant", value=0)
    return F.interpolate(video_padded, size=(CLIP_SIZE, CLIP_SIZE), mode="bilinear", align_corners=False)


def preprocess(frames, seconds):
    """VGGSound.sample (thinksound_288_al.py) in inference mode, with duration_sec = seconds.

    Returns clip frames (T, 288, 288, 3) float32 in [0, 1] and sync frames (T, 3, 224, 224) float32.
    """
    import numpy as np
    import torch
    from torchvision.transforms import v2

    sync_transform = v2.Compose([
        v2.Resize(SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    duration_sec = float(seconds)
    clip_chunk, sync_chunk = frames[CLIP_FPS], frames[SYNC_FPS]

    clip_expected_length = int(CLIP_FPS * duration_sec)
    sync_expected_length = int(SYNC_FPS * duration_sec)

    clip_chunk = clip_chunk[:clip_expected_length]
    if clip_chunk.shape[0] != clip_expected_length:
        padding_needed = clip_expected_length - clip_chunk.shape[0]
        if padding_needed > 0:
            last_frame = clip_chunk[-1]
            padding = last_frame.repeat(padding_needed, 1, 1, 1)
            clip_chunk = torch.cat((clip_chunk, padding), dim=0)
    clip_chunk = pad_to_square(clip_chunk)
    clip_chunk = clip_chunk.permute(0, 2, 3, 1)
    clip_chunk = np.asarray(clip_chunk).astype(np.float32) / np.float32(255)  # mediapy.to_float01 for uint8

    sync_chunk = sync_chunk[:sync_expected_length]
    if sync_chunk.shape[0] != sync_expected_length:
        last_frame = sync_chunk[-1]
        padding = last_frame.repeat(sync_expected_length - sync_chunk.shape[0], 1, 1, 1)
        sync_chunk = torch.cat((sync_chunk, padding), dim=0)
    sync_chunk = sync_transform(sync_chunk)
    return clip_chunk, sync_chunk


def new_features_utils():
    """Upstream FeaturesUtils without its __init__, which loads T5Gemma by hub name; models are set per stage."""
    import torch
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils

    fu = FeaturesUtils.__new__(FeaturesUtils)
    torch.nn.Module.__init__(fu)
    return fu


def free_device_cache():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def videoprism_features(fu, clip_chunk, prompt, weights, jit_mode):
    """FeaturesUtils._init_jax with local files, then the upstream encode method.

    jit_mode "closure" is upstream: the jitted lambda closes over the 2.3 GB state, which JAX bakes
    into the executable as constants. "argument" passes the state in instead; less memory and
    compile time, float32 rounding differs slightly where XLA folds constants.
    """
    import jax
    import numpy as np
    import torch
    from videoprism import models as vp

    def hf_hub_download(repo_id, filename, **kwargs):
        if (repo_id, filename) != VIDEOPRISM_FILE:
            raise RuntimeError(f"unexpected hub download {repo_id}/{filename}")
        return weights["videoprism"]

    vp.huggingface_hub = types.SimpleNamespace(hf_hub_download=hf_hub_download)
    vp.TEXT_TOKENIZERS["c4_en"]["model_path"] = weights["videoprism_tokenizer"]

    backend = jax.default_backend()
    if backend != "gpu":
        log(f"VideoPrism runs on JAX {backend}")
        jax_dev = jax.devices()[0]
    else:
        devices = jax.devices()
        jax_dev = devices[int(os.environ.get("LOCAL_RANK", 0)) % len(devices)]
    flax_model = vp.get_model(VIDEOPRISM_MODEL)
    state = vp.load_pretrained_weights(VIDEOPRISM_MODEL)
    loaded_state = jax.device_put(state, device=jax_dev)
    del state
    fu.flax_model = flax_model
    fu.loaded_state = loaded_state
    fu.text_tokenizer = vp.load_text_tokenizer("c4_en")
    # upstream passes device= to jax.jit, which current JAX removed; inputs are already placed on jax_dev
    if jit_mode == "closure":
        fu.apply_jit = jax.jit(lambda x, y, z: flax_model.apply(
            loaded_state, x, y, z, train=False, return_intermediate=True))
    else:
        jitted = jax.jit(lambda s, x, y, z: flax_model.apply(
            s, x, y, z, train=False, return_intermediate=True))
        fu.apply_jit = lambda x, y, z: jitted(loaded_state, x, y, z)
    fu.jax_dev = jax_dev

    clip_input = torch.from_numpy(clip_chunk).unsqueeze(0)
    video_feat, frame_embed, _, text_feat = fu.encode_video_and_text_with_videoprism(clip_input, [prompt])
    feats = {
        "global_video_features": np.array(video_feat),
        "video_features": np.array(frame_embed),
        "global_text_features": np.array(text_feat),
    }
    peak = (jax_dev.memory_stats() or {}).get("peak_bytes_in_use", 0) if backend == "gpu" else 0
    del fu.flax_model, fu.loaded_state, fu.text_tokenizer, fu.apply_jit, fu.jax_dev
    del video_feat, frame_embed, text_feat, loaded_state
    jax.clear_caches()
    return feats, peak


def synchformer_features(fu, sync_chunk, weights, device):
    """Synchformer as FeaturesUtils.__init__ loads it, then the upstream encode method."""
    import torch
    from data_utils.ext.synchformer import Synchformer

    synchformer = Synchformer()
    synchformer.load_state_dict(torch.load(weights["synchformer"], weights_only=True, map_location="cpu"))
    fu.synchformer = synchformer
    fu.eval().to(device)
    sync_features = fu.encode_video_with_sync(sync_chunk.unsqueeze(0).to(device))
    out = sync_features.detach().cpu().numpy()[0]
    del fu.synchformer, synchformer, sync_features
    free_device_cache()
    return out


def t5gemma_features(fu, prompt, path, device):
    """T5Gemma as FeaturesUtils.__init__ loads it, from a local directory, then the upstream encode method."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    fu.t5 = AutoModelForSeq2SeqLM.from_pretrained(path).get_encoder()
    fu.t5tokenizer = AutoTokenizer.from_pretrained(path)
    fu.eval().to(device)
    text_features = fu.encode_t5_text([prompt])
    out = text_features.detach().cpu().numpy()[0]
    del fu.t5, fu.t5tokenizer, text_features
    free_device_cache()
    return out


def load_predict(src):
    spec = importlib.util.spec_from_file_location("prismaudio_predict", os.path.join(src, "predict.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keep_sync_features(model, latent_length):
    """sync = "features": give the DiT the Synchformer features, as training and GRPO did.

    Upstream predict_step does conditioning['sync_features'][~video_exist] = empty_sync_feat on a
    Python list; with video_exist True that index is 0, so the features are replaced by the zero
    empty_sync_feat. This puts the conditioner output back, zero padded or cut to the latent length
    (Sync_MLP gives int(194 * n / 216) steps, which only equals the latent length for 8-13 s).
    """
    import torch.nn.functional as F

    conditioner = model.conditioner
    conditioner_forward = conditioner.forward
    get_inputs = model.get_conditioning_inputs
    kept = {}

    def forward(batch_metadata, device):
        out = conditioner_forward(batch_metadata, device)
        kept["sync"] = out["sync_features"][0]
        return out

    def get_conditioning_inputs(conditioning_tensors, negative=False):
        sync = kept["sync"]
        n = sync.shape[1]
        if n < latent_length:
            sync = F.pad(sync, (0, 0, 0, latent_length - n))
        elif n > latent_length:
            sync = sync[:, :latent_length]
        conditioning_tensors["sync_features"] = [sync, conditioning_tensors["sync_features"][1]]
        return get_inputs(conditioning_tensors, negative)

    conditioner.forward = forward
    model.get_conditioning_inputs = get_conditioning_inputs


def generate(job, feats, device):
    """predict.main for one clip, then the upstream predict_step. Returns int16 audio (2, n)."""
    import numpy as np
    import torch
    from PrismAudio.models import create_model_from_config
    from PrismAudio.models.utils import load_ckpt_state_dict

    src, seconds, seed = job["source"], job["seconds"], job["seed"]
    predict = load_predict(src)

    # seed_everything(seed, workers=True) as in lightning
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(os.path.join(src, MODEL_CONFIG)) as f:
        model_config = json.load(f)
    duration = float(seconds)
    model_config["sample_size"] = duration * model_config["sample_rate"]
    diffusion_config = model_config["model"]["diffusion"]["config"]
    if "sync_seq_len" in diffusion_config:
        diffusion_config["sync_seq_len"] = 24 * int(duration)
    if "clip_seq_len" in diffusion_config:
        diffusion_config["clip_seq_len"] = 8 * int(duration)
    if "latent_seq_len" in diffusion_config:
        diffusion_config["latent_seq_len"] = round(44100 / 64 / 32 * duration)

    model = create_model_from_config(model_config)
    model.load_state_dict(torch.load(job["weights"]["prismaudio"], map_location="cpu", weights_only=True))
    model.pretransform.load_state_dict(load_ckpt_state_dict(job["weights"]["vae"], prefix="autoencoder."))
    model.eval()

    latent_length = round(44100 / 64 / 32 * duration)
    audio = torch.zeros((1, 64, latent_length), dtype=torch.float32)
    meta = {key: torch.from_numpy(value) for key, value in feats.items()}
    meta.update(id="demo", relpath="demo.npz", path="demo.npz", caption_cot=job["prompt"],
                video_exist=torch.tensor(True))
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            meta[key] = value.to(device)
    if job["sync"] == "features":
        keep_sync_features(model, latent_length)

    log(f"sampling: latent length {latent_length}, sync {job['sync']}")
    audio = predict.predict_step(
        model, batch=[audio, (meta,)],
        diffusion_objective=model_config["model"]["diffusion"]["diffusion_objective"], device=device)
    return audio[0, :, : seconds * SAMPLE_RATE]


def write_wav(path, audio):
    """audio: int16 tensor (2, n), the same samples torchaudio.save writes."""
    data = audio.t().contiguous().numpy().astype("<i2", copy=False).tobytes()
    tmp = path + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)
    os.replace(tmp, path)


def run(job):
    device = "cuda:0" if job["device"] == "cuda" else "cpu"
    stage = time.monotonic()

    def done(name):
        nonlocal stage
        now = time.monotonic()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        log(f"{name}: {now - stage:.1f} s, peak RSS so far {rss:.2f} GiB")
        stage = now

    import torch

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("device is cuda but torch sees no GPU")
    install_stubs()
    done("imports")

    frames = decode_frames(job["clip"], job["seconds"])
    clip_chunk, sync_chunk = preprocess(frames, job["seconds"])
    del frames
    log(f"clip frames {tuple(clip_chunk.shape)}, sync frames {tuple(sync_chunk.shape)}")
    done("preprocess")

    fu = new_features_utils()
    feats, jax_peak = videoprism_features(fu, clip_chunk, job["prompt"], job["weights"], job["videoprism_jit"])
    done("VideoPrism")
    feats["sync_features"] = synchformer_features(fu, sync_chunk, job["weights"], device)
    done("Synchformer")
    if job["text_features"] == "zeros":
        import numpy as np

        log("TEST ONLY: T5Gemma text features replaced by zeros")
        feats["text_features"] = np.zeros((ZERO_TEXT_TOKENS, 1024), dtype=np.float32)
    else:
        feats["text_features"] = t5gemma_features(fu, job["prompt"], job["weights"]["t5gemma"], device)
        done("T5Gemma")
    del fu, clip_chunk, sync_chunk
    log("features " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in sorted(feats.items())))

    audio = generate(job, feats, device)
    done("generate")
    if tuple(audio.shape) != (2, job["seconds"] * SAMPLE_RATE):
        raise RuntimeError(f"unexpected audio shape {tuple(audio.shape)}")
    write_wav(job["out"], audio)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if device == "cpu":
        peak = rss
    else:
        peak = torch.cuda.max_memory_reserved() + jax_peak
        log(f"GPU peak: torch reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, "
            f"JAX {jax_peak / 2**30:.2f} GiB; host RSS {rss / 2**30:.2f} GiB")
    return {"ok": True, "elapsed_s": round(time.monotonic() - T0, 2), "peak_gib": round(peak / 2**30, 3)}


def main():
    result_out = claim_stdout()
    try:
        job = parse_job(sys.stdin.read())
        configure_env(job)
        block_network()
        check_source(job["source"])
        sys.path.insert(0, os.path.abspath(job["source"]))
        result = run(job)
    except JobError as e:
        log(f"error: {e}")
        return 2
    except Exception as e:
        traceback.print_exc()
        log(f"error: {type(e).__name__}: {str(e)[:300]}")
        return 1
    result_out.write(json.dumps(result) + "\n")
    result_out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
