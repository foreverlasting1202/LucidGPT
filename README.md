# Activation Descent

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
pip install -r requirements.txt
```

Notes:
- Training assumes CUDA + NCCL (`torchrun`).
- Tokenization is GPT-2 BPE (`tiktoken`), same as the data scripts.

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

Single node, 8 GPUs:

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

Details are documented in `docs/monitoring.md`.

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