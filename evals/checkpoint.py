from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


def load_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint produced by `utils.Logger.save_checkpoint()`.
    """
    return torch.load(path, map_location=map_location)


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Be forgiving about common wrapper prefixes:
    - torch.compile: `_orig_mod.`
    - DDP: `module.`
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def get_model_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract the model state dict from either:
    - a full training checkpoint dict with key `model`
    - a raw state dict itself
    """
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict-like state dict, got {type(state)}")
    return _strip_known_prefixes(state)


def get_model_config(ckpt: Dict[str, Any]) -> Optional[ModelConfig]:
    """
    Try to recover model hyperparameters from the checkpoint if present.

    Newer checkpoints (after this change) store `model_config`.
    Older checkpoints likely won't, in which case callers must provide config via CLI.
    """
    mc = ckpt.get("model_config")
    if isinstance(mc, dict):
        try:
            return ModelConfig(
                vocab_size=int(mc["vocab_size"]),
                n_layer=int(mc["n_layer"]),
                n_head=int(mc["n_head"]),
                n_embd=int(mc["n_embd"]),
            )
        except Exception:
            return None

    # Fallback: use training config if available
    tc = ckpt.get("training_config") or ckpt.get("config")
    if isinstance(tc, dict):
        keys = ("vocab_size", "n_layer", "n_head", "n_embd")
        if all(k in tc for k in keys):
            try:
                return ModelConfig(
                    vocab_size=int(tc["vocab_size"]),
                    n_layer=int(tc["n_layer"]),
                    n_head=int(tc["n_head"]),
                    n_embd=int(tc["n_embd"]),
                )
            except Exception:
                return None

    return None

