"""
Optimizer setup for AdamW and Muon+AdamW.
"""
from __future__ import annotations

import math
from typing import Any, List, Tuple

import torch

from args import TrainingConfig
from optim.adamw import MonitoredAdamW
from optim.muon import Muon


def _all_named_trainable_params(raw_model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in raw_model.named_parameters() if p.requires_grad]


def _canonical_param_name(name: str) -> str:
    """
    Remove common wrapper prefixes so routing works with torch.compile / wrappers.
    """
    prefixes = ("module.", "_orig_mod.", "raw_model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _adamw_kwargs(config: TrainingConfig) -> dict:
    return {
        "lr": config.adamw_learning_rate,
        "betas": (config.adamw_beta1, config.adamw_beta2),
        "eps": config.adamw_eps,
        "weight_decay": config.weight_decay,
    }


def create_adamw_optimizers(
    config: TrainingConfig,
    raw_model,
    ddp_rank: int,
    ddp_world_size: int,
) -> List[Any]:
    """
    Create a single AdamW optimizer over all trainable parameters.
    """
    _ = (ddp_rank, ddp_world_size)
    named_params = _all_named_trainable_params(raw_model)
    optimizer = MonitoredAdamW(named_params=named_params, **_adamw_kwargs(config))
    return [optimizer]


def _split_muon_adamw_param_groups(
    raw_model,
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
    """
    Split trainable params into AdamW and Muon groups.

    Policy:
    - Muon: 2D matrices in transformer blocks (`transformer.h.*`)
    - AdamW: everything else (embeddings, lm_head, norms, biases, 1D params)
    """
    all_named_params: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        all_named_params.append((_canonical_param_name(name), param))

    adamw_named_params: List[Tuple[str, torch.nn.Parameter]] = []
    muon_named_params: List[Tuple[str, torch.nn.Parameter]] = []

    # Preferred route: Muon on 2D matrices in transformer blocks.
    for name, param in all_named_params:
        if name.startswith("transformer.h.") and param.dim() == 2:
            muon_named_params.append((name, param))
        else:
            adamw_named_params.append((name, param))

    # Fallback for wrapped/custom models where names differ:
    # use Muon on any eligible 2D matrix except embeddings/lm_head.
    if not muon_named_params:
        adamw_named_params = []
        muon_named_params = []
        for name, param in all_named_params:
            is_forced_adamw = (
                name.startswith("transformer.wte.")
                or name.startswith("lm_head.")
            )
            if (param.dim() == 2) and (not is_forced_adamw):
                muon_named_params.append((name, param))
            else:
                adamw_named_params.append((name, param))
    return adamw_named_params, muon_named_params


def create_muon_optimizers(
    config: TrainingConfig,
    raw_model,
    ddp_rank: int,
    ddp_world_size: int,
) -> List[Any]:
    """
    Create Muon+AdamW optimizers:
    - AdamW for non-matrix / non-transformer-block params
    - Muon for 2D matrices in transformer blocks
    """
    adamw_named_params, muon_named_params = _split_muon_adamw_param_groups(raw_model)
    if not adamw_named_params:
        raise ValueError("Muon+AdamW split produced an empty AdamW parameter group")
    if not muon_named_params:
        sample_names = [name for name, _ in _all_named_trainable_params(raw_model)[:8]]
        raise ValueError(
            "Muon+AdamW split produced an empty Muon parameter group. "
            f"Sample parameter names: {sample_names}"
        )

    if int(ddp_rank) == 0:
        adamw_numel = sum(p.numel() for _, p in adamw_named_params)
        muon_numel = sum(p.numel() for _, p in muon_named_params)
        print(
            "optimizer split (Muon+AdamW): "
            f"adamw={len(adamw_named_params)} params / {adamw_numel} elems, "
            f"muon={len(muon_named_params)} params / {muon_numel} elems"
        )

    adamw_opt = MonitoredAdamW(named_params=adamw_named_params, **_adamw_kwargs(config))
    muon_opt = Muon(
        [p for _, p in muon_named_params],
        lr=config.muon_learning_rate,
        momentum=config.muon_momentum,
        nesterov=config.muon_nesterov,
        backend=config.muon_backend,
        backend_steps=config.muon_backend_steps,
        rank=ddp_rank,
        world_size=ddp_world_size,
        weight_decay=config.weight_decay,
        param_names=[name for name, _ in muon_named_params],
    )
    return [adamw_opt, muon_opt]


def create_optimizers(config: TrainingConfig, raw_model, ddp_rank: int, ddp_world_size: int) -> List[Any]:
    """
    Factory function to create optimizers based on configuration.
    """
    optimizer_factories = {
        "adamw": create_adamw_optimizers,
        "muon": create_muon_optimizers,
    }
    if config.optimizer not in optimizer_factories:
        raise ValueError(
            f"Unsupported optimizer: {config.optimizer}. "
            f"Supported optimizers: {list(optimizer_factories.keys())}"
        )
    return optimizer_factories[config.optimizer](config, raw_model, ddp_rank, ddp_world_size)


def create_lr_schedulers(optimizers: List[Any], config: TrainingConfig) -> List[Any]:
    """
    Create learning-rate schedulers for all optimizers.
    """

    def get_lr_trapezoidal(it: int) -> float:
        """Linear warmup + constant plateau + linear warmdown."""
        assert it <= config.num_iterations
        warmup = max(int(config.warmup_iters), 1)
        warmdown = max(int(config.warmdown_iters), 1)
        if it < warmup:
            return (it + 1) / warmup
        if it < config.num_iterations - warmdown:
            return 1.0
        return (config.num_iterations - it) / warmdown

    def get_lr_cosine(it: int) -> float:
        """Linear warmup + cosine decay."""
        assert it <= config.num_iterations
        warmup = max(int(config.warmup_iters), 1)
        if it < warmup:
            return (it + 1) / warmup
        denom = max(config.num_iterations - warmup, 1)
        decay_ratio = (it - warmup) / denom
        return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    get_lr = get_lr_cosine if config.lr_scheduler == "cosine" else get_lr_trapezoidal
    return [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]