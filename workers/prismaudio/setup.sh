#!/usr/bin/env bash
# Builds the PrismAudio sound worker under the data root: the upstream source at its pinned commit, Python 3.11 with
# every package pinned (requirements.txt), and VideoPrism's text tokenizer checked by sha256. Safe to rerun.
# The weights are model sets: uv run studio models fetch prismaudio, videoprism-lvt and t5gemma.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
data=${STUDIO_DATA:-$HOME/studio-data}
home=$data/workers/prismaudio
commit=86eed40e1e2eddc54787a31ae66df01eb8befd3f
vocab=$data/models/audio/videoprism/cc_en.32000.sentencepiece.model
vocab_url=https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model
vocab_sha256=1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec

mkdir -p "$home"
chmod 700 "$home"

# ThinkSound's prismaudio branch at the pinned commit; run.py checks the files it uses against a digest.
if [ ! -f "$home/src/predict.py" ]; then
  git clone -q --no-checkout https://github.com/QwenAudio/ThinkSound.git "$home/src"
  git -C "$home/src" checkout -q "$commit"
fi

# Python 3.11.15 kept inside the worker folder, then the pinned packages without resolving dependencies.
export UV_PYTHON_INSTALL_DIR=$home/python UV_CACHE_DIR=$home/.uv-cache
uv python install -q 3.11.15 --no-bin
[ -x "$home/.venv/bin/python" ] ||
  uv venv -q --no-project --python "$(UV_PYTHON_PREFERENCE=only-managed uv python find --no-project 3.11.15)" \
    "$home/.venv"
# Bytecode compiling starts one Python per CPU thread: more open files than the usual soft limit of 1024 allows.
ulimit -Sn "$(ulimit -Hn)"
uv pip install -q --python "$home/.venv/bin/python" --no-deps --compile-bytecode -r "$here/requirements.txt" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match

# VideoPrism's text tokenizer, from the public T5 bucket.
if ! echo "$vocab_sha256  $vocab" | sha256sum -c --quiet >/dev/null 2>&1; then
  mkdir -p "$(dirname "$vocab")"
  curl -fsSL -o "$vocab.part" "$vocab_url"
  echo "$vocab_sha256  $vocab.part" | sha256sum -c --quiet
  mv "$vocab.part" "$vocab"
fi
echo "PASS  PrismAudio worker in $home"
