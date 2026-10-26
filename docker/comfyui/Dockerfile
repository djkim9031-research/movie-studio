# ComfyUI for studio: one pinned commit, PyTorch for CUDA 13.0, no manager and no custom nodes.
# --offline turns off the paid partner nodes and stops the browser page from reaching the internet.
# Build inputs come from config/studio.yaml through `studio gpu build`.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COMFY_COMMIT
ARG TORCH_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python

# gcc: Triton compiles its kernel launcher at run time; the int8 weights run through Triton kernels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gcc git libc6-dev libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 /uv /usr/local/bin/uv
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# Every package comes from the hashed lock: frozen from the image that passed the first renders.
COPY requirements.lock check_env.py /opt/studio/
RUN uv pip install --require-hashes --no-deps -r /opt/studio/requirements.lock \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
      --index-strategy unsafe-best-match

RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout --detach "${COMFY_COMMIT}" \
 && python /opt/studio/check_env.py "${TORCH_VERSION}+cu130"

COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--offline", \
            "--input-directory", "/io/input", "--output-directory", "/io/output", \
            "--temp-directory", "/io/temp", "--user-directory", "/io/user", \
            "--extra-model-paths-config", "/opt/ComfyUI/extra_model_paths.yaml"]
