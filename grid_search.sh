#!/usr/bin/env bash
set -euo pipefail

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT_RAW=${ARNOLD_WORKER_0_PORT:-29500}
MASTER_PORT=${MASTER_PORT_RAW%%,*}
NPROC_PER_NODE=${ARNOLD_WORKER_GPU:-8}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

# Small, public-friendly sweep examples.
optimizers=("adamw" "muon")
weight_decays=(0.1 0.01)
adamw_lrs=(0.0036 0.0018)
muon_lrs=(0.02 0.01)
muon_momentums=(0.9 0.95)

for opt in "${optimizers[@]}"; do
  for wd in "${weight_decays[@]}"; do
    for idx in "${!adamw_lrs[@]}"; do
      adam_lr=${adamw_lrs[$idx]}

      base_cmd=(
        torchrun
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
        --nproc_per_node="${NPROC_PER_NODE}"
        --nnodes="${NNODES}"
        --node_rank="${NODE_RANK}"
        main.py
        --optimizer "${opt}"
        --adamw_learning_rate "${adam_lr}"
        --weight_decay "${wd}"
        --num_iterations 20000
        --warmup_iters 2000
        --warmdown_iters 14000
        --batch_size 512
        --device_batch_size 8
        --sequence_length 1024
        --lr_scheduler cosine
      )

      if [[ "${opt}" == "muon" ]]; then
        for mu_idx in "${!muon_lrs[@]}"; do
          mu_lr=${muon_lrs[$mu_idx]}
          mu_mom=${muon_momentums[$mu_idx]}
          "${base_cmd[@]}" \
            --muon_learning_rate "${mu_lr}" \
            --muon_momentum "${mu_mom}"
        done
      else
        "${base_cmd[@]}"
      fi
    done
  done
done