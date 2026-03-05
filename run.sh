#!/usr/bin/env bash
set -euo pipefail

# Multi-node defaults (works on single node too).
MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT_RAW=${ARNOLD_WORKER_0_PORT:-29500}
MASTER_PORT=${MASTER_PORT_RAW%%,*}
NPROC_PER_NODE=${ARNOLD_WORKER_GPU:-8}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

# Pass extra CLI overrides directly to main.py.
EXTRA_ARGS=("$@")

torchrun \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  main.py \
  --optimizer muon \
  --adamw_learning_rate 0.0012 \
  --muon_learning_rate 0.002 \
  --muon_momentum 0.95 \
  --weight_decay 0.1 \
  --lr_scheduler cosine \
  --batch_size 512 \
  --device_batch_size 8 \
  --sequence_length 1024 \
  --log_activation_norm \
  --log_activation_update_norm \
  --log_param_update_norm \
  --eval_tasks fineweb,bench,pretrain \
  --eval_during_train_tasks mmlu_fineweb,hellaswag \
  "${EXTRA_ARGS[@]}"