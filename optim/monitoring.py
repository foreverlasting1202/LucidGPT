from __future__ import annotations

import math
from typing import Dict, Optional

import torch


COMPONENTS = ("attn", "mlp", "lm_head", "embedding")


def tensor_rms(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.dtype != torch.float32:
        x = x.float()
    return x.pow(2).mean().sqrt()


def update_fro_and_rms(update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (Frobenius norm, RMS norm) for an update tensor in FP32.
    """
    u = update.detach()
    if u.dtype != torch.float32:
        u = u.float()
    update_fro = u.norm()
    update_rms = update_fro / math.sqrt(max(int(u.numel()), 1))
    return update_fro, update_rms


def classify_param_name(name: str) -> Optional[str]:
    # Allow names with optional wrappers (DDP/compile/raw-model aliases).
    prefixes = ("module.", "_orig_mod.", "raw_model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True

    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if name.startswith("lm_head.") or ".lm_head." in name:
        return "lm_head"
    if name.startswith("transformer.wte.") or ".transformer.wte." in name:
        return "embedding"
    return None


def init_component_stats(device: torch.device) -> Dict[str, Dict[str, torch.Tensor | int]]:
    def _zero() -> torch.Tensor:
        return torch.zeros((), device=device, dtype=torch.float32)

    return {
        key: {
            "count": 0,
            "update_fro_sum": _zero(),
            "update_fro_max": _zero(),
            "update_rms_sum": _zero(),
            "update_rms_max": _zero(),
            "param_rms_sum": _zero(),
            "param_rms_max": _zero(),
        }
        for key in COMPONENTS
    }


def accumulate_component_stats(
    stats: Dict[str, Dict[str, torch.Tensor | int]],
    *,
    component: str,
    update_tensor: torch.Tensor,
    param_tensor: torch.Tensor,
) -> None:
    if component not in stats:
        return

    update_fro, update_rms = update_fro_and_rms(update_tensor)
    param_rms = tensor_rms(param_tensor)

    bucket = stats[component]
    bucket["count"] = int(bucket["count"]) + 1
    bucket["update_fro_sum"] = bucket["update_fro_sum"] + update_fro
    bucket["update_fro_max"] = torch.maximum(bucket["update_fro_max"], update_fro)
    bucket["update_rms_sum"] = bucket["update_rms_sum"] + update_rms
    bucket["param_rms_sum"] = bucket["param_rms_sum"] + param_rms
    bucket["update_rms_max"] = torch.maximum(bucket["update_rms_max"], update_rms)
    bucket["param_rms_max"] = torch.maximum(bucket["param_rms_max"], param_rms)


def finalize_component_stats(
    stats: Dict[str, Dict[str, torch.Tensor | int]],
) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for key, bucket in stats.items():
        count = int(bucket["count"])
        if count <= 0:
            continue
        out[key] = {
            "count": count,
            "update_fro_sum": float(bucket["update_fro_sum"].item()),
            "update_fro_max": float(bucket["update_fro_max"].item()),
            "update_rms_sum": float(bucket["update_rms_sum"].item()),
            "update_rms_max": float(bucket["update_rms_max"].item()),
            "param_rms_sum": float(bucket["param_rms_sum"].item()),
            "param_rms_max": float(bucket["param_rms_max"].item()),
        }
    return out
