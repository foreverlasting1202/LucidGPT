# Monitoring

This repo provides optional training-time monitors for optimizer health and activation behavior.
All monitors are logging-only: they never stop training.

## What is monitored

- `ActivationMonitor` (`train_metrics.py`)
  - Block activation RMS (`activation/rms_*`)
  - Attention/MLP residual-branch update RMS (`activation/attn_*`, `activation/mlp_*`)
  - Embedding/logits activation RMS summaries

- `ParamUpdateMonitor` (`train_metrics.py`)
  - Aggregated per-component parameter/update stats from optimizer internals
  - Components: `attn`, `mlp`, `embedding`, `lm_head`
  - New norms available for both `MonitoredAdamW` and `Muon`:
    - `update_fro_norm`
    - `update_rms_norm`

## How to enable

- Activation monitoring:
  - `--log_activation_norm`
  - `--log_activation_update_norm`
  - `--activation_log_every N`

- Parameter-update monitoring:
  - `--log_param_update_norm`
  - `--param_update_norm_every N`

## Main metric families

- Activation metrics:
  - `activation/rms_mean`, `activation/rms_max`
  - `activation/attn_update_rms_mean`, `activation/mlp_update_rms_mean`
  - `activation/act_<component>_rms_mean`

- Parameter-update metrics (per component):
  - `param/<component>_update_fro_norm_mean`
  - `param/<component>_update_fro_norm_max`
  - `param/<component>_update_rms_norm_mean`
  - `param/<component>_update_rms_norm_max`
  - `param/<component>_rms_mean`
  - `param/<component>_rms_max`

- Optimizer-wide detail metrics:
  - `optimizer/adamw_update_fro_norm_mean`, `optimizer/adamw_update_rms_norm_mean`
  - `optimizer/muon_update_fro_norm_mean`, `optimizer/muon_update_rms_norm_mean`

## Runtime behavior

- Monitors are created on rank0 only (to minimize overhead).
- Capturing is step-gated by `*_log_every` flags.
- Logged metrics are merged into the normal train logging stream (console/file/W&B/SwanLab).

