#!/usr/bin/env bash
# Run FastMix with *downstream SFT data* as the search target.
#
# This variant additionally needs:
#   - a tokenizer (TOKENIZER_PATH, defaults to the public EleutherAI/gpt-neox-20b)
#   - the SFT search-target files under data/sft (SFT_DATA_DIR)
#   - lm-evaluation-harness installed (for downstream benchmark scoring)
#
# Override any of the variables below via the environment, e.g.:
#   TRAIN_DATA_DIR=/data/welldata TOKENIZER_PATH=/models/gpt-neox-20b bash scripts/run_sft.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Directory of preprocessed .bin shards (produced by preprocess/run_preprocess.sh).
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$REPO_DIR/data/welldata}"
VAL_DATA_DIR="${VAL_DATA_DIR:-$TRAIN_DATA_DIR}"
DATA_YAML="${DATA_YAML:-$REPO_DIR/configs/pile.yaml}"
OUT_NAME="${OUT_NAME:-fastmix_sft_lr0p01}"
LR_DATASET="${LR_DATASET:-0.01}"
DEVICES="${DEVICES:-1}"
EVAL_STEP="${EVAL_STEP:-10}"

# Tokenizer + SFT search-target location (consumed by the script via env vars).
export TOKENIZER_PATH="${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}"
export SFT_DATA_DIR="${SFT_DATA_DIR:-$REPO_DIR/data/sft}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python pretrain/train_fastmix_sft.py \
    --devices "$DEVICES" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --data_yaml_file "$DATA_YAML" \
    --out_name "$OUT_NAME" \
    --resume False \
    --eval_step "$EVAL_STEP" \
    --learning_rate_dataset "$LR_DATASET"
