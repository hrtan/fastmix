#!/usr/bin/env bash
# Download and tokenize the RegMix data into packed .bin shards (LITPKDS format).
#
# The output directory produced here is exactly what you pass to the training scripts
# as --train_data_dir / --val_data_dir.
#
# Override any path via the environment, e.g.:
#   SOURCE_PATH=sail/regmix-data-sample DEST_PATH=/data/welldata bash preprocess/run_preprocess.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR/preprocess"

export DATASET_SHORT_NAME="${DATASET_SHORT_NAME:-the_pile}"

# Raw JSONL dataset (expects train/*.jsonl and valid/*.jsonl underneath). When set to a
# HuggingFace repo id, download it first with download_dataset.py (step 1 below); it lands
# under preprocess/<repo_id>/ which this relative path then resolves to.
SOURCE_PATH="${SOURCE_PATH:-sail/regmix-data-sample}"

# gpt-neox tokenizer (bundled in this repo; matches the model vocab and the SFT script).
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_DIR/preprocess/tokenizer/gptneox}"

# Output dir of packed .bin shards -> use as --train_data_dir / --val_data_dir.
DEST_PATH="${DEST_PATH:-$REPO_DIR/data/welldata}"

# 1) (optional) Download the sample dataset from the HuggingFace Hub.
#    For the full ~1TB dataset use: --dataset_name sail/regmix-data
# python download_dataset.py --dataset_name sail/regmix-data-sample

# We use the gptneox tokenizer to stay consistent with the RegMix / DoReMi setup.

# 2) Tokenize + pack the TRAIN split.
python prepare_file_domain.py \
    --source_path "$SOURCE_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --destination_path "$DEST_PATH" \
    --short_name "$DATASET_SHORT_NAME" \
    --split train

# 3) Tokenize + pack the VALID split.
#    131136 = 2049 * 64 -> a smaller chunk size suited to the small validation set,
#    especially for low-resource domains.
python prepare_file_domain.py \
    --source_path "$SOURCE_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --destination_path "$DEST_PATH" \
    --short_name "$DATASET_SHORT_NAME" \
    --split valid \
    --chunk_size 131136
