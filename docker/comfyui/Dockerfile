# ComfyUI for studio: one pinned commit, PyTorch for CUDA 13.0, no manager and no custom nodes.
# Build inputs come from config/studio.yaml through `studio gpu build`.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COMFY_COMMIT
ARG TORCH_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# torch first and pinned: ComfyUI's requirements name torch without a version.
RUN uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu130

RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout --detach "${COMFY_COMMIT}" \
 && uv pip install -r /opt/ComfyUI/requirements.txt

COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-partner-nodes", \
            "--input-directory", "/io/input", "--output-directory", "/io/output", \
            "--temp-directory", "/io/temp", "--user-directory", "/io/user", \
            "--extra-model-paths-config", "/opt/ComfyUI/extra_model_paths.yaml"]
