# Pretrain for LLMs

Clean GPT-style pretraining with:
- simple launch commands,
- two optimizer modes (`adamw` and `muon`),
- built-in benchmark evaluation during/after training,
- optimizer and activation monitors for debugging training stability.

## Why this repo

- Clean and minimal pretrain pipeline.
- Two supported datasets today:
  - FineWeb
  - FineWeb-Edu sorted by score
- Benchmark coverage is broad and can run while training.
- Monitoring is first-class for diagnosing optimization issues.

## Setup

```bash
python -m pip install -r requirements.txt
```

Prerequisites:
- NVIDIA GPU + CUDA runtime compatible with `torch==2.7.1`.
- Multi-GPU / multi-node launches use `torchrun` + NCCL.
- Tokenization is GPT-2 BPE (`tiktoken`), same as the data scripts.

### Tracker login (SwanLab / W&B)

This repo supports both trackers:
- SwanLab is enabled by default (`--use_swanlab` / `--no_use_swanlab`).
- Weights & Biases is disabled by default (`--use_wandb` / `--no_use_wandb`).

Login once per machine/user:

```bash
# SwanLab (interactive)
swanlab login

# Or non-interactive
# swanlab login -k <your-api-key>
```

```bash
wandb login
```

Enable both trackers in one run:

```bash
torchrun --standalone --nproc_per_node=8 main.py \
  --use_swanlab --swanlab_project "act-descent" \
  --use_wandb --wandb_project "act-descent"
```

Disable cloud tracking completely with:
- `--no_use_swanlab --no_use_wandb`

For more login and launch recipes (including multi-node), see `docs/README.md`.

## Data

### 1) Download pre-tokenized shards (recommended)

```bash
# FineWeb 10B (pass an int to download fewer train shards)
python data/cached_fineweb10B.py 8

# FineWeb 100B
# python data/cached_fineweb100B.py 8

# FineWeb-Edu 10B
# python data/cached_finewebedu10B.py 8
```

### 2) Build shards yourself

```bash
# FineWeb
python data/fineweb.py --version 10B

# FineWeb-Edu (score-prioritized token order)
python data/finewebedu.py --stage all --overwrite
```

### FineWeb-Edu 100B (sorted by score)

- Hugging Face dataset link (to be filled): `TODO: add your finewebedu100B-sorted HF link here`

## Optimizers

Only two optimizer modes are public:

1. `adamw`
   - Single AdamW optimizer over all trainable parameters.

2. `muon`
   - Muon on 2D transformer-block matrices.
   - AdamW on remaining parameters (embedding, lm_head, norms, 1D params).

This split keeps Muon where it is intended to be used and avoids mixing optimizer assumptions.

## Training

### Batch math (important)

Effective global batch is:
- `batch_size = device_batch_size * WORLD_SIZE * grad_accum_steps`

In this repo, `grad_accum_steps` is computed automatically, so startup requires:

```text
batch_size % (device_batch_size * WORLD_SIZE) == 0
```

### Single node, 8 GPUs

```bash
torchrun --standalone --nproc_per_node=8 main.py \
  --optimizer muon \
  --adamw_learning_rate 0.0012 \
  --muon_learning_rate 0.002 \
  --muon_momentum 0.95 \
  --weight_decay 0.1 \
  --batch_size 512 \
  --device_batch_size 8 \
  --sequence_length 1024 \
  --lr_scheduler cosine
```

### Multi-node, multi-GPU (`torchrun`)

Run the same command on every node, with different `NODE_RANK` values (`0..NNODES-1`):

```bash
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0            # node0=0, node1=1, ...
export MASTER_ADDR=10.0.0.1   # IP/hostname of node0
export MASTER_PORT=29500

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py \
  --optimizer muon \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --batch_size 1024 \
  --device_batch_size 8 \
  --sequence_length 1024
```

Multi-node checklist:
- Use the same code revision and Python environment on every node.
- Ensure all nodes can resolve/reach `MASTER_ADDR:MASTER_PORT`.
- Keep data paths valid on all nodes (`--input_bin`, `--input_val_bin`).
- Launch each `torchrun` stage in the same order on all nodes (important when chaining runs).
- For long eval-after-train on rank0, tune `--eval_after_train_timeout_seconds` if needed.

### Important args

- Core train:
  - `--batch_size`, `--device_batch_size`, `--sequence_length`
  - `--num_iterations`, `--warmup_iters`, `--warmdown_iters`
  - `--weight_decay`, `--lr_scheduler`
- AdamW:
  - `--adamw_learning_rate`, `--adamw_beta1`, `--adamw_beta2`, `--adamw_eps`
- Muon:
  - `--muon_learning_rate`, `--muon_momentum`
  - `--muon_nesterov/--no_muon_nesterov`
  - `--muon_backend`, `--muon_backend_steps`

### Monitoring args

- Activation:
  - `--log_activation_norm`
  - `--log_activation_update_norm`
  - `--activation_log_every`
- Parameter update:
  - `--log_param_update_norm`
  - `--param_update_norm_every`

Monitoring details are in `docs/monitoring.md`; launcher/login recipes are in `docs/README.md`.

## Evaluation during and after training

- During training:
  - `--eval_during_train_tasks mmlu_fineweb,hellaswag,...`
- After training:
  - `--eval_tasks fineweb,bench,pretrain,...`
  - disable with `--no_eval_after_train`
- Common eval config:
  - `--eval_dtype`
  - `--eval_max_seq_len`
  - `--eval_limit` (global per-task limit for debugging)
  - `--eval_mmlu_nshot`
  - `--eval_mmlu_subjects`

Post-train JSON outputs are written under:
- `logs/<run_id>/eval/latest.json`
- `logs/<run_id>/eval/eval_stepXXXXXX.json`

## Benchmark tasks and metrics

| Task | Main metric(s) in this repo | Notes |
|---|---|---|
| `pretrain` | `loss`, `perplexity`, `token_accuracy`, `tokens_per_second` | Validation over `.bin` shards |
| `mmlu` | `accuracy`, `accuracy_norm` (`acc_norm`) | Few-shot configurable by `--eval_mmlu_nshot` |
| `mmlu_fineweb` | `acc`, `acc_norm` | FineWeb-style 0-shot, full-answer continuation targets |
| `hellaswag` | `acc_norm` | Multiple-choice |
| `arc_easy` | `acc_norm` | ARC-Easy |
| `arc_challenge` | `acc_norm` | ARC-Challenge |
| `piqa` | `accuracy`, `accuracy_norm` | Multiple-choice |
| `winogrande` | `acc_norm` | Multiple-choice |
| `openbookqa` | `acc_norm` | Multiple-choice |
| `siqa` | `acc_norm` | Multiple-choice |
| `commonsense_qa` | `acc_norm` | Multiple-choice |

### Why `acc_norm` for MMLU / MC tasks?

`acc_norm` is length-normalized multiple-choice accuracy (log-likelihood divided by answer length).
It is commonly used in evaluation frameworks for MC tasks (including lm-eval-style and lighteval-style setups) to reduce answer-length bias.

In this repo:
- `mmlu` exposes both raw `accuracy` and normalized `accuracy_norm`.
- FineWeb aggregate uses `acc_norm`-based columns by design.

## Offline checkpoint evaluation (`evaluate.py`)

Examples:

```bash
# Pretrain validation metrics
python evaluate.py \
  --ckpt logs/<run_id>/state_stepXXXXXX.pt \
  pretrain \
  --input_bin "data/fineweb10B/fineweb_val_*.bin" \
  --eval_tokens 1048576
```

```bash
# MMLU
python evaluate.py \
  --ckpt logs/<run_id>/state_stepXXXXXX.pt \
  mmlu --nshot 5 --subjects all
```

```bash
# FineWeb aggregate suite
python evaluate.py \
  --ckpt logs/<run_id>/state_stepXXXXXX.pt \
  fineweb --mmlu_subjects all
```

## Repo layout

- `main.py`, `trainer.py`: training entrypoint + loop
- `args.py`: CLI configuration
- `optim/`, `optimizers.py`: optimizer implementations and wiring
- `train_metrics.py`: monitoring implementation
- `data/`: dataset download/preprocessing scripts
- `evals/`, `evaluate.py`: evaluation framework and tasks
- `docs/README.md`: tracker login + launch cookbook
