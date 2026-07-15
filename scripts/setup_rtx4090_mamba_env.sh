#!/usr/bin/env bash
# Create a reproducible, isolated CUDA 12.8 environment for an Ada RTX 4090.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-rtx4090}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIP_NO_CACHE_DIR=1
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf-cache}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0"
"$PY" -m pip install \
  "protobuf>=6.30.2,<7" "z3-solver>=4.13,<4.15.5" "fsspec[http]>=2023.1.0,<=2026.4.0" \
  datasets sentencepiece pyarrow pandas einops ninja safetensors tensorboard cloudpickle \
  psutil tqdm typing-extensions transformers
"$PY" -m pip install --no-deps "apache-tvm-ffi==0.1.10" "tilelang==0.1.10" \
  torch-c-dlpack-ext quack-kernels
MAMBA_FORCE_BUILD=TRUE "$PY" -m pip install --no-deps --force-reinstall --no-build-isolation \
  "git+https://github.com/state-spaces/mamba.git@main"
"$PY" -m pip install --no-deps -e "$ROOT"
"$PY" - <<'PY'
import torch
from mamba_ssm.modules.mamba3 import Mamba3

assert torch.cuda.is_available(), "CUDA is unavailable"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "mamba3": Mamba3.__name__})
PY
printf '%s\n' "RTX4090_PYTHON=$PY"
