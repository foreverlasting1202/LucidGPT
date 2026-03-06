#!/usr/bin/env bash
set -euo pipefail

export TORCH_DIST_INIT_BARRIER_TIMEOUT=3600  
export TORCHELASTIC_TIMEOUT=3600          
export NCCL_TIMEOUT=3600000  

# Pass extra CLI overrides directly to main.py.
EXTRA_ARGS=("$@")

torchrun --standalone --nproc_per_node=8 main.py \
  --optimizer muon \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --adamw_learning_rate 0.0012 \
  --muon_learning_rate 0.002 \
  --muon_momentum 0.95 \
  --weight_decay 0.1 \
  --num_iterations 6000 \
  --warmup_iters 600 \
  --warmdown_iters 6000 \
  --batch_size 512 \
  --device_batch_size 8 \
  --sequence_length 1024 \
  --lr_scheduler cosine \
  --log_activation_norm \
  --log_activation_update_norm \
  --log_param_update_norm \
  --eval_during_train_tasks "mmlu_fineweb,hellaswag,arc_challenge" \
  --eval_tasks "pretrain,mmlu,mmlu_fineweb,hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,siqa,commonsense_qa" \
  --eval_limit 1000 \
  --swanlab_project "agd_final_test"

torchrun --standalone --nproc_per_node=8 main.py \
  --optimizer adamw \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --adamw_learning_rate 0.0036 \
  --adamw_beta1 0.9 \
  --adamw_beta2 0.98 \
  --adamw_eps 1e-8 \
  --weight_decay 0.1 \
  --num_iterations 6000 \
  --warmup_iters 600 \
  --warmdown_iters 6000 \
  --batch_size 512 \
  --device_batch_size 8 \
  --sequence_length 1024 \
  --lr_scheduler cosine \
  --log_activation_norm \
  --log_activation_update_norm \
  --log_param_update_norm \
  --eval_tasks "pretrain,mmlu,mmlu_fineweb,hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,siqa,commonsense_qa" \
  --eval_limit 1000 \
  --swanlab_project "agd_final_test"