# FastMix: Fast Data Mixture Optimization via Gradient Descent

**FastMix** jointly trains a small proxy language model and *searches for the optimal
data-mixture weights* over a set of training sources. Instead of doing an expensive grid
search over mixtures, it estimates, online, how reweighting each data source would affect a
target objective, and updates the mixture weights by gradient descent during a single
training run.

The core idea: at each search step we compare the **per-source training gradient** on the
model's LM head against the **gradient of a target objective**. Sources whose gradients
align with the target objective get their sampling weight increased (via a softmax over the
learnable `dataset_probs`); sources that hurt the target get down-weighted.

There are two variants of the search target, one per entrypoint:

| Script | Search target | Notes |
| --- | --- | --- |
| `pretrain/train_fastmix_val.py` | A held-out **validation split of the training data** (`--val_data_dir`) | Pure language-modeling target. |
| `pretrain/train_fastmix_sft.py` | Downstream **SFT data** (`data/sft/*_sft.jsonl`) | Optimizes the mixture toward downstream tasks; tracks benchmark accuracy via `lm-evaluation-harness`. |

The learned mixture is periodically written to
`checkpoints/<out_name>/FastMixtureOut/probs_module_step*.pt`.

---

## Repository layout

```
FastMix/
├── pretrain/
│   ├── train_fastmix_val.py   # search target = validation split of training data
│   └── train_fastmix_sft.py   # search target = downstream SFT data
├── lit_gpt/                   # data pipeline, fused kernels, speed monitor, utils
├── configs/
│   └── pile.yaml              # data sources + initial weights + hyper-parameter overrides
├── data/
│   └── sft/                   # SFT search-target files (used by train_fastmix_sft.py)
│       ├── hellaswag_sft.jsonl
│       ├── piqa_sft.jsonl
│       ├── sciq_sft.jsonl
│       └── arc_challenge_sft.jsonl
├── preprocess/                # official RegMix data-prep: JSONL -> packed .bin shards
│   ├── download_dataset.py    # fetch sail/regmix-data[-sample] from the HF Hub
│   ├── prepare_file_domain.py # tokenize + pack into LITPKDS .bin shards
│   ├── run_preprocess.sh
│   └── tokenizer/             # bundled gpt-neox (+ starcoder) tokenizers
├── scripts/
│   ├── setup_env.sh
│   ├── run_val.sh
│   └── run_sft.sh
├── third_party/               # vendored, source-built deps (see "Setup")
│   ├── pytorch-lightning-2.5.1.post0/   # patched PyTorch-Lightning (install-only subset)
│   └── lm-evaluation-harness/           # benchmark harness (install-only subset)
├── requirements.txt
└── README.md
```

> The two packages under `third_party/` are vendored here for reproducibility and are
> trimmed to the subset needed to install and run (their `docs/`, `tests/`, `examples/`,
> CI configs, and build artifacts have been removed). They are still installed from source
> during setup.

---

## Setup

FastMix depends on a few source-built packages in addition to the pip requirements.
A CUDA-capable GPU is required.

### 1. Install PyTorch (with CUDA) and flash-attention

Install a `torch` build matching your CUDA version, then build flash-attention
(it provides the fused cross-entropy / RMSNorm kernels used by `lit_gpt`):

```bash
# Example only — match the index URL to your CUDA version.
pip install --index-url https://download.pytorch.org/whl/cu118 --pre 'torch>=2.1.0dev'

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \
    cd ../..
```

### 2. The source-built third-party packages are bundled

Both source-built dependencies are already included under `third_party/`:

- `third_party/pytorch-lightning-2.5.1.post0` — the patched PyTorch-Lightning.
- `third_party/lm-evaluation-harness` — only required by `train_fastmix_sft.py`.

You don't need to download them separately; `setup_env.sh` installs them from there.

### 3. Run the setup script

```bash
bash scripts/setup_env.sh
```

This installs `requirements.txt`, then:

```bash
# equivalent to what setup_env.sh runs for you:
cd third_party/pytorch-lightning-2.5.1.post0 && python setup.py install && cd -
pip install -e third_party/lm-evaluation-harness --break-system-packages
```

### 4. (Optional) Weights & Biases

Logging is enabled automatically when `WANDB_API_KEY` is set; otherwise it falls back to
anonymous/offline behavior. To disable it entirely:

```bash
export WANDB_MODE=disabled
# or, to log to your account:
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> The original hard-coded W&B key has been removed — set your own via the env var.

---

## Data preparation

Training reads tokenized, packed `.bin` shards in the `LITPKDS` format. These are produced
by the official [RegMix](https://github.com/sail-sg/regmix) data-prep pipeline, vendored
here under `preprocess/`. Each source in `configs/pile.yaml` maps to a set of shards by
file-name prefix, e.g. `train_the_pile_arxiv` → `train_the_pile_arxiv_*_*.bin` inside
`--train_data_dir`.

### 1. Download the raw dataset

```bash
cd preprocess
# small sample (recommended to start); use sail/regmix-data for the full ~1TB set
python download_dataset.py --dataset_name sail/regmix-data-sample
cd ..
```

This downloads JSONL files into `preprocess/sail/regmix-data-sample/{train,valid}/`.

### 2. Tokenize and pack into `.bin` shards

```bash
bash preprocess/run_preprocess.sh
```

`run_preprocess.sh` calls `prepare_file_domain.py` to tokenize each domain with the
bundled **gpt-neox** tokenizer (`preprocess/tokenizer/gptneox`, consistent with RegMix /
DoReMi) and pack it into shards. By default the output goes to `data/welldata/`, which is
exactly the directory the training scripts default to for `--train_data_dir` /
`--val_data_dir`. Override paths via env vars:

```bash
SOURCE_PATH=sail/regmix-data-sample \
DEST_PATH=/path/to/welldata \
bash preprocess/run_preprocess.sh
```

The valid split is packed with a smaller chunk size (`131136 = 2049 * 64`) to suit the
small validation set, especially for low-resource domains.

### SFT search target

The SFT search target (`data/sft/*_sft.jsonl`) consists of `{"question", "answer"}` pairs.
`train_fastmix_sft.py` masks the question tokens and computes the loss/gradient only on the
answer tokens.

---

## Training

The convenience scripts are env-var driven and default to the preprocessing output
(`data/welldata`). If you preprocessed to a different location, set `TRAIN_DATA_DIR`
(and optionally `VAL_DATA_DIR`).

### Variant A — validation-split target

```bash
TRAIN_DATA_DIR=/path/to/welldata \
OUT_NAME=fastmix_val_lr0p01 \
LR_DATASET=0.01 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_val.sh
```

### Variant B — SFT target

```bash
TRAIN_DATA_DIR=/path/to/welldata \
TOKENIZER_PATH=/path/to/gpt-neox-20b \
OUT_NAME=fastmix_sft_lr0p01 \
LR_DATASET=0.01 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft.sh
```

### Running the entrypoints directly

The scripts are thin wrappers around the Python entrypoints. Equivalent raw command:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain/train_fastmix_val.py \
    --devices 1 \
    --train_data_dir /path/to/welldata \
    --val_data_dir   /path/to/welldata \
    --data_yaml_file configs/pile.yaml \
    --out_name fastmix_val_lr0p01 \
    --resume False \
    --learning_rate_dataset 0.01
```

### Key arguments

| Argument | Meaning |
| --- | --- |
| `--train_data_dir` | Directory of preprocessed training shards. |
| `--val_data_dir` | Validation shards (used by the `val` variant as the target). |
| `--data_yaml_file` | Mixture config (sources + initial weights + overrides). |
| `--out_name` | Run name; checkpoints go to `checkpoints/<out_name>/`. |
| `--learning_rate_dataset` | Learning rate for the mixture weights `dataset_probs`. |
| `--resume` | `False`, `True` (latest checkpoint), or a path. |
| `--eval_step` | (SFT variant only) how often the mixture search step runs. |

Environment variables consumed by `train_fastmix_sft.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOKENIZER_PATH` | `EleutherAI/gpt-neox-20b` | Tokenizer for encoding the SFT target. |
| `SFT_DATA_DIR` | `data/sft` | Directory of `*_sft.jsonl` search-target files. |

---

## Outputs

- `checkpoints/<out_name>/iter-XXXXXX-ckpt.pth` — proxy-model checkpoints.
- `checkpoints/<out_name>/FastMixtureOut/probs_module_step*.pt` — the learned mixture
  logits at each search step. Apply `softmax` to recover the sampling distribution over
  sources (the order matches the `train:` entries in the config).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tan2026fastmix,
  title={Fast Data Mixture Optimization via Gradient Descent},
  author={Tan, Haoru and Wu, Sitong and Chen, Yanfeng and Xia, Jun and Xie, Ruobing and Xia, Bin and Sun, Xingwu and Qi, Xiaojuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## Acknowledgements

The model / data-loading stack (`lit_gpt`, packed dataset, fused kernels) is built on the
excellent [TinyLlama](https://github.com/jzhang38/TinyLlama) and
[lit-gpt](https://github.com/Lightning-AI/lit-gpt) and [RegMix](https://github.com/sail-sg/regmix) codebases. Downstream evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
