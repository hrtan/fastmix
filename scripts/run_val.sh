#!/usr/bin/env bash
# Run FastMix with the *validation split of the training data* as the search target.
#
# Override any of the variables below via the environment, e.g.:
#   TRAIN_DATA_DIR=/data/welldata LR_DATASET=0.01 bash scripts/run_val.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Directory of preprocessed .bin shards (LITPKDS format), as produced by
# preprocess/run_preprocess.sh (defaults to data/welldata).
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$REPO_DIR/data/welldata}"
VAL_DATA_DIR="${VAL_DATA_DIR:-$TRAIN_DATA_DIR}"
DATA_YAML="${DATA_YAML:-$REPO_DIR/configs/pile.yaml}"
OUT_NAME="${OUT_NAME:-fastmix_val_lr0p01}"
LR_DATASET="${LR_DATASET:-0.01}"
DEVICES="${DEVICES:-1}"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python pretrain/train_fastmix_val.py \
    --devices "$DEVICES" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --data_yaml_file "$DATA_YAML" \
    --out_name "$OUT_NAME" \
    --resume False \
    --learning_rate_dataset "$LR_DATASET"
