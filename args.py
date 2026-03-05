"""
Argument parsing and configuration management for training.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    # Data
    input_bin: str
    input_val_bin: str

    # Optimization
    optimizer: str
    lr_scheduler: str
    batch_size: int
    device_batch_size: int
    sequence_length: int
    num_iterations: int
    warmup_iters: int
    warmdown_iters: int
    weight_decay: float

    # AdamW (used by both optimizer modes)
    adamw_learning_rate: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_eps: float

    # Muon-specific (only used when optimizer=muon)
    muon_learning_rate: float
    muon_momentum: float
    muon_nesterov: bool
    muon_backend: str
    muon_backend_steps: int

    # Model
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

    # Validation/checkpointing
    val_loss_every: int
    val_tokens: int
    save_every: int

    # Monitoring
    log_activation_norm: bool
    log_activation_update_norm: bool
    activation_log_every: int
    log_param_update_norm: bool
    param_update_norm_every: int

    # Evaluation
    eval_after_train: bool
    eval_after_train_timeout_seconds: int
    eval_tasks: str
    eval_during_train_tasks: str
    eval_dtype: str
    eval_max_seq_len: Optional[int]
    eval_pretrain_tokens: int
    eval_limit: Optional[int]
    eval_mmlu_nshot: int
    eval_mmlu_subjects: str

    # Tracking backends
    use_wandb: bool
    wandb_project: str
    wandb_run_name: Optional[str]
    use_swanlab: bool
    swanlab_project: str
    swanlab_run_name: Optional[str]

    # Profiler
    profile: bool
    profile_dir: str
    profile_wait: int
    profile_warmup: int
    profile_active: int
    profile_repeat: int
    profile_memory: bool
    profile_with_stack: bool
    profile_rank0_only: bool


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    default: bool,
    help_true: str,
    help_false: str,
) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_true)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false", help=help_false)
    parser.set_defaults(**{dest: bool(default)})


def _default_run_name(config: TrainingConfig) -> str:
    parts = [
        config.optimizer,
        f"adamwlr-{config.adamw_learning_rate:g}",
        f"wd-{config.weight_decay:g}",
        f"nlayer-{config.n_layer}",
        f"nhead-{config.n_head}",
        f"nembd-{config.n_embd}",
        f"sched-{config.lr_scheduler}",
    ]
    if config.optimizer == "muon":
        parts.extend(
            [
                f"muonlr-{config.muon_learning_rate:g}",
                f"muonmom-{config.muon_momentum:g}",
                f"ns-{config.muon_backend_steps}",
            ]
        )
    return "-".join(parts)


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig object."""
    parser = argparse.ArgumentParser(description="Train GPT model")

    # Data
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin files for training",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin files for validation loss / pretrain eval",
    )

    # Optimization
    parser.add_argument(
        "--optimizer",
        type=str,
        default="muon",
        choices=["adamw", "muon"],
        help="optimizer mode: adamw (single optimizer) or muon (Muon+AdamW)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["trapezoidal", "cosine"],
        help="learning-rate scheduler",
    )
    parser.add_argument("--batch_size", type=int, default=8 * 64, help="global batch size (sequences)")
    parser.add_argument("--device_batch_size", type=int, default=8, help="per-device batch size (sequences)")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length in tokens")
    parser.add_argument("--num_iterations", type=int, default=6000, help="number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=600, help="linear warmup iterations")
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=6000,
        help="linear warmdown iterations for trapezoidal scheduler",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="decoupled weight decay")

    # AdamW hyperparameters
    parser.add_argument("--adamw_learning_rate", type=float, default=0.0036, help="AdamW learning rate")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adamw_beta2", type=float, default=0.98, help="AdamW beta2")
    parser.add_argument("--adamw_eps", type=float, default=1e-8, help="AdamW epsilon")

    # Muon hyperparameters
    parser.add_argument("--muon_learning_rate", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--muon_momentum", type=float, default=0.9, help="Muon momentum")
    _add_bool_arg(
        parser,
        name="muon_nesterov",
        default=True,
        help_true="enable Nesterov momentum in Muon",
        help_false="disable Nesterov momentum in Muon",
    )
    parser.add_argument(
        "--muon_backend",
        type=str,
        default="newtonschulz5",
        choices=["newtonschulz5", "svd"],
        help="orthogonalization backend for Muon",
    )
    parser.add_argument(
        "--muon_backend_steps",
        type=int,
        default=5,
        help="iterations for Muon backend when using iterative orthogonalization",
    )

    # Model
    parser.add_argument("--vocab_size", type=int, default=50304, help="vocabulary size")
    parser.add_argument("--n_layer", type=int, default=12, help="number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="embedding dimension")

    # Validation/checkpointing
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=125,
        help="evaluate val loss every N steps (0: only at the end)",
    )
    parser.add_argument("--val_tokens", type=int, default=10240 * 1024, help="validation tokens per eval")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="save checkpoint every N steps (0: only at the end)",
    )

    # Monitoring
    parser.add_argument("--log_activation_norm", action="store_true", help="log activation RMS summaries")
    parser.add_argument(
        "--log_activation_update_norm",
        action="store_true",
        help="log activation update RMS summaries",
    )
    parser.add_argument(
        "--activation_log_every",
        type=int,
        default=1,
        help="capture activation metrics every N steps",
    )
    parser.add_argument(
        "--log_param_update_norm",
        action="store_true",
        help="log optimizer-side parameter update metrics",
    )
    parser.add_argument(
        "--param_update_norm_every",
        type=int,
        default=1,
        help="capture parameter update metrics every N steps",
    )

    # Evaluation
    _add_bool_arg(
        parser,
        name="eval_after_train",
        default=True,
        help_true="run eval automatically after training",
        help_false="skip post-train eval",
    )
    parser.add_argument(
        "--eval_after_train_timeout_seconds",
        type=int,
        default=12 * 60 * 60,
        help="timeout for post-train cross-rank barrier",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="fineweb",
        help=(
            "comma-separated post-train tasks: "
            "pretrain,mmlu,mmlu_fineweb,hellaswag,arc_easy,arc_challenge,piqa,"
            "winogrande,openbookqa,siqa,commonsense_qa,fineweb,bench"
        ),
    )
    parser.add_argument(
        "--eval_during_train_tasks",
        type=str,
        default="",
        help="comma-separated tasks to run after each validation",
    )
    parser.add_argument(
        "--eval_dtype",
        type=str,
        default="bf16",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="autocast dtype for eval on CUDA",
    )
    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=None,
        help="override max_seq_len used by eval (default: training sequence_length)",
    )
    parser.add_argument(
        "--eval_pretrain_tokens",
        type=int,
        default=1024 * 1024,
        help="tokens used by pretrain-style eval",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=None,
        help="global per-task sample limit for benchmark evals (debug)",
    )
    parser.add_argument("--eval_mmlu_nshot", type=int, default=5, help="few-shot count for MMLU")
    parser.add_argument(
        "--eval_mmlu_subjects",
        type=str,
        default="all",
        help="MMLU subjects: 'all' or comma-separated subject names",
    )

    # Tracking
    _add_bool_arg(
        parser,
        name="use_wandb",
        default=False,
        help_true="enable Weights & Biases logging",
        help_false="disable Weights & Biases logging",
    )
    parser.add_argument("--wandb_project", type=str, default="act-descent", help="wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    _add_bool_arg(
        parser,
        name="use_swanlab",
        default=True,
        help_true="enable SwanLab logging",
        help_false="disable SwanLab logging",
    )
    parser.add_argument("--swanlab_project", type=str, default="act-descent", help="swanlab project")
    parser.add_argument("--swanlab_run_name", type=str, default=None, help="swanlab run name")

    # Profiler
    parser.add_argument("--profile", action="store_true", help="enable PyTorch profiler")
    parser.add_argument("--profile_dir", type=str, default="profiles", help="profiler output directory")
    parser.add_argument("--profile_wait", type=int, default=1, help="profiler wait steps")
    parser.add_argument("--profile_warmup", type=int, default=1, help="profiler warmup steps")
    parser.add_argument("--profile_active", type=int, default=3, help="profiler active steps")
    parser.add_argument("--profile_repeat", type=int, default=1, help="profiler repeat count")
    _add_bool_arg(
        parser,
        name="profile_memory",
        default=True,
        help_true="enable memory profiling",
        help_false="disable memory profiling",
    )
    _add_bool_arg(
        parser,
        name="profile_with_stack",
        default=False,
        help_true="enable stack traces in profiler",
        help_false="disable stack traces in profiler",
    )
    _add_bool_arg(
        parser,
        name="profile_rank0_only",
        default=True,
        help_true="profile only rank0",
        help_false="profile all ranks",
    )

    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    if not config.wandb_run_name:
        config.wandb_run_name = _default_run_name(config)
    if not config.swanlab_run_name:
        config.swanlab_run_name = _default_run_name(config)
    return config