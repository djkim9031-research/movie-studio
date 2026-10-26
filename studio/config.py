import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "studio.yaml"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Price(BaseModel):
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float


class Route(BaseModel):
    model: str
    effort: Effort


class ClaudeConfig(BaseModel):
    routes: dict[str, Route] = {}
    prices: dict[str, Price] = {}
    # Charged when a hop names a model missing from `prices`: the highest current rates, so caps stay safe.
    unknown_price: Price = Price(input=10.0, output=50.0, cache_read=1.0)
    cache_write_5m: float = 1.25
    cache_write_1h: float = 2.0
    film_cap_usd: float = 20.0
    month_cap_usd: float = 40.0


class GpuConfig(BaseModel):
    agent_service: str = "agent-vllm"
    min_free_gib: float = 60.0


class DiskConfig(BaseModel):
    home_min_free_gb: float = 300.0
    docker_root: Path = Path("/var/lib/docker")
    docker_min_free_gb: float = 50.0


class FfmpegConfig(BaseModel):
    release_tag: str
    asset: str
    sha256: str


class ComfyConfig(BaseModel):
    image_repo: str = "studio-comfyui"
    shm_gb: int = 16
    # An internal Docker network: the host reaches ComfyUI at `ip`, the container reaches nothing outside.
    network: str = "studio-net"
    subnet: str = "192.168.131.0/29"
    gateway: str = "192.168.131.1"
    ip: str = "192.168.131.2"
    # Build inputs, pinned before the first build: base image by digest, ComfyUI by commit.
    base_image: str | None = None
    commit: str | None = None
    torch: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:8188"


class ModelFile(BaseModel):
    path: str  # in the repo
    size: int
    sha256: str
    dest: str | None = None  # under the set's root, when the repo's layout differs from ComfyUI's model folders

    @property
    def target(self) -> str:
        return self.dest or self.path


class ModelSet(BaseModel):
    """Files of one Hugging Face repo at a pinned commit, under models/<root>; root comfy is ComfyUI's model folders."""

    repo: str
    revision: str
    root: str = "comfy"
    # Whether the license lets face recognition run on what this set makes; anything but "allowed" is refused.
    biometric_outputs: Literal["allowed", "forbidden", "unknown"] = "unknown"
    files: list[ModelFile]

    @field_validator("root")
    @classmethod
    def _plain_name(cls, value):
        if not re.fullmatch(r"[a-z0-9_-]+", value):
            raise ValueError(f"model root must be a plain folder name: {value!r}")
        return value


class Config(BaseModel):
    data_root: Path
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    disk: DiskConfig = Field(default_factory=DiskConfig)
    ffmpeg: FfmpegConfig | None = None
    models: dict[str, ModelSet] = {}
    comfyui: ComfyConfig = Field(default_factory=ComfyConfig)

    @field_validator("data_root", mode="before")
    @classmethod
    def _expand(cls, value):
        return Path(value).expanduser()

    @property
    def bin_dir(self) -> Path:
        return self.data_root / "bin"

    @property
    def ledger_dir(self) -> Path:
        return self.data_root / "ledger"

    @property
    def projects_dir(self) -> Path:
        return self.data_root / "projects"

    @property
    def secrets_dir(self) -> Path:
        return self.data_root / "secrets"

    @property
    def anthropic_env(self) -> Path:
        return self.secrets_dir / "anthropic.env"

    @property
    def hf_env(self) -> Path:
        return self.secrets_dir / "hf.env"

    @property
    def comfy_io_dir(self) -> Path:
        """Mounted at /io in the container: input, output, temp, user and home."""
        return self.data_root / "comfy-io"

    @property
    def models_dir(self) -> Path:
        """ComfyUI's model root: diffusion_models/, text_encoders/, vae/ and so on."""
        return self.data_root / "models" / "comfy"

    def model_root(self, name: str) -> Path:
        return self.data_root / "models" / self.models[name].root


def load_config(path: Path | None = None) -> Config:
    with open(path or REPO_CONFIG, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
