# Launch and Tracking Cookbook

This page focuses on practical run commands:
- how to log in to SwanLab / W&B,
- how to launch single-node and multi-node training,
- and what to check when distributed jobs fail.

For metric definitions, see `monitoring.md`.

## 1) Tracker login

### SwanLab

Interactive login:

```bash
swanlab login
```

Non-interactive login:

```bash
swanlab login -k <your-api-key>
```

Alternative (environment variable):

```bash
export SWANLAB_API_KEY=<your-api-key>
```

Log out:

```bash
swanlab logout
```

### Weights & Biases (W&B)

This repo sets `WANDB_BASE_URL=https://api.bandw.top` in `main.py`, so login should target the same host:

```bash
wandb login --host https://api.bandw.top
```

Alternative (environment variable):

```bash
export WANDB_API_KEY=<your-api-key>
```

## 2) Tracker flags in this repo

- SwanLab default: enabled (`--use_swanlab`)
- W&B default: disabled (`--no_use_wandb`)

Typical combinations:

```bash
# SwanLab only (default)
--use_swanlab --no_use_wandb

# W&B only
--no_use_swanlab --use_wandb

# Both
--use_swanlab --use_wandb

# Neither
--no_use_swanlab --no_use_wandb
```

Naming/project flags:
- `--swanlab_project`, `--swanlab_run_name`
- `--wandb_project`, `--wandb_run_name`

Note: logging is initialized on rank0 only.

## 3) Single-node launch recipes

### 8-GPU training

```bash
torchrun --standalone --nproc_per_node=8 main.py \
  --optimizer muon \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --adamw_learning_rate 0.0012 \
  --muon_learning_rate 0.002 \
  --muon_momentum 0.95 \
  --weight_decay 0.1 \
  --num_iterations 6000 \
  --batch_size 512 \
  --device_batch_size 8 \
  --sequence_length 1024 \
  --lr_scheduler cosine \
  --use_swanlab --swanlab_project "act-descent"
```

### 1-GPU quick debug

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --num_iterations 50 \
  --batch_size 8 \
  --device_batch_size 8 \
  --eval_limit 64 \
  --eval_tasks "pretrain" \
  --no_use_swanlab --no_use_wandb
```

## 4) Multi-node, multi-GPU launch

Use the same command on every node, but set different `NODE_RANK` values.

### Shared variables (all nodes)

```bash
export NNODES=2
export GPUS_PER_NODE=8
export MASTER_ADDR=10.0.0.1   # node0 IP/hostname
export MASTER_PORT=29500
```

### Node-specific variable

```bash
# node0
export NODE_RANK=0

# node1
export NODE_RANK=1
```

### Launch command (run on each node)

```bash
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
  --sequence_length 1024 \
  --eval_after_train_timeout_seconds 43200
```

## 5) Batch-size rule (required)

This repo computes gradient-accumulation steps automatically with:

```text
grad_accum_steps = batch_size / (device_batch_size * WORLD_SIZE)
```

So `batch_size` must satisfy:

```text
batch_size % (device_batch_size * WORLD_SIZE) == 0
```

Where `WORLD_SIZE = NNODES * GPUS_PER_NODE`.

## 6) Multi-node troubleshooting

### `DistStoreError` / rendezvous timeout

Check:
- All nodes use the same values for `NNODES`, `MASTER_ADDR`, `MASTER_PORT`.
- `NODE_RANK` is unique per node and starts from `0`.
- Firewalls/security groups allow `MASTER_PORT`.
- All nodes start the same `torchrun` stage in the same order.

### NCCL init/connectivity errors

Check:
- GPU driver/CUDA/NCCL compatibility across nodes.
- Network interface selection and routing.
- Try setting `NCCL_DEBUG=INFO` for more details.

### One run exits, others keep waiting

Usually one node crashed earlier (OOM, import error, data path issue).
Inspect per-node logs and fix the first failing rank.

### No metrics visible in dashboards

- Ensure login succeeded (`swanlab login`, `wandb login ...`).
- Confirm `--use_swanlab` / `--use_wandb` flags.
- Metrics are emitted from rank0 only in this codebase.

