#!/usr/bin/env bash
# One-time environment setup for FastMix.
#
# This installs the core Python deps and the two source-built third-party packages.
# Both are already vendored in this repo under third_party/:
#   third_party/pytorch-lightning-2.5.1.post0   (patched PyTorch-Lightning)
#   third_party/lm-evaluation-harness           (only needed by train_fastmix_sft.py)
#
# You should install a CUDA-enabled `torch` and `flash-attention` separately first
# (see README.md "Setup").
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] Installing core Python dependencies..."
pip install -r "$REPO_DIR/requirements.txt"

PL_DIR="${PL_DIR:-$REPO_DIR/third_party/pytorch-lightning-2.5.1.post0}"
if [ -d "$PL_DIR" ]; then
  echo "[setup] Installing patched PyTorch-Lightning from $PL_DIR ..."
  (cd "$PL_DIR" && python setup.py install)
else
  echo "[setup] WARN: $PL_DIR not found. Install your patched pytorch-lightning manually."
fi

LM_EVAL_DIR="${LM_EVAL_DIR:-$REPO_DIR/third_party/lm-evaluation-harness}"
if [ -d "$LM_EVAL_DIR" ]; then
  echo "[setup] Installing lm-evaluation-harness from $LM_EVAL_DIR ..."
  pip install -e "$LM_EVAL_DIR" --break-system-packages
else
  echo "[setup] WARN: $LM_EVAL_DIR not found. Skipping (only needed for the SFT variant)."
fi

echo "[setup] Done."
