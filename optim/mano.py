"""
Mano optimizer implementation.

Reference style:
- https://github.com/MoonshotAI/Moonlight
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch

from optim.monitoring import (
    accumulate_component_stats,
    classify_param_name,
    finalize_component_stats,
    init_component_stats,
)


class Mano_v2(torch.optim.Optimizer):
    """
    Mano (manifold-normalized optimizer) with built-in AdamW fallback group.
    """

    def __init__(
        self,
        *,
        lr: float = 1e-3,
        adamw_lr: float = 0.0036,
        wd: float = 0.1,
        eps: float = 1e-8,
        mano_params: Optional[Iterable[torch.nn.Parameter]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_params: Optional[Iterable[torch.nn.Parameter]] = None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        param_names: Optional[List[str]] = None,
    ) -> None:
        def _dedupe_keep_order(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
            out: List[torch.nn.Parameter] = []
            seen_ids: set[int] = set()
            for p in params:
                pid = id(p)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                out.append(p)
            return out

        if mano_params is None:
            mano_params = []
        mano_params = [p for p in mano_params if p.requires_grad]
        adamw_params = [p for p in (adamw_params or []) if p.requires_grad]

        # Deduplicate while preserving order.
        mano_params = _dedupe_keep_order(mano_params)
        mano_ids = {id(p) for p in mano_params}
        adamw_params = [p for p in _dedupe_keep_order(adamw_params) if id(p) not in mano_ids]

        params = mano_params + adamw_params
        if not params:
            raise ValueError("Mano_v2 received no trainable parameters")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        beta1, beta2 = adamw_betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"adamw_betas must be in [0,1), got {adamw_betas}")
        if adamw_eps <= 0:
            raise ValueError(f"adamw_eps must be > 0, got {adamw_eps}")
        if adamw_lr is None:
            adamw_lr = float(lr)
        

        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            wd=wd,
            eps=eps,
            momentum=momentum,
            nesterov=bool(nesterov),
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            steps=0,
        )
        super().__init__(params, defaults)

        self.track_update_stats: bool = False
        self.last_update_stats: Dict[str, Dict[str, float | int]] = {}
        self.last_update_details: Dict[str, Dict[str, float | int]] = {}
        self.param_name_by_id: Dict[int, str] = {}

        if param_names is not None:
            if len(param_names) != len(params):
                raise ValueError("param_names length must match Mano parameters length")
            self.param_name_by_id = {id(p): n for p, n in zip(params, param_names)}

        # Mark routing for each parameter.
        for p in mano_params:
            if p.ndim != 2:
                raise ValueError(f"Mano params must be 2D, got ndim={p.ndim}")
            self.state[p]["use_mano"] = True
        for p in adamw_params:
            self.state[p]["use_mano"] = False

    def set_monitoring_enabled(self, enabled: bool = True) -> None:
        self.track_update_stats = bool(enabled)
        if not self.track_update_stats:
            self.last_update_stats = {}
            self.last_update_details = {}

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        monitor_stats = None
        monitor_details: Dict[str, Dict[str, float | int]] = {}

        for group in self.param_groups:
            lr = float(group["lr"])
            adamw_lr = float(group["adamw_lr"])
            wd = float(group["wd"])
            eps = float(group["eps"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])

            dim = int(group.get("steps", 0) % 2)

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                use_mano = bool(state.get("use_mano", False))

                if use_mano:
                    # Mano branch (2D params only).
                    g_fp32 = g.detach().float()
                    p_fp32 = p.data.detach().float()

                    buf = state.get("momentum_buffer")
                    if buf is None or buf.shape != g_fp32.shape:
                        buf = torch.zeros_like(g_fp32)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(g_fp32)
                    m = g_fp32.add(buf, alpha=momentum) if nesterov else buf

                    tangent_mt = m - (torch.sum(m * p_fp32, dim=dim, keepdim=True) * p_fp32)
                    u = tangent_mt / (torch.norm(tangent_mt, p=2, dim=dim, keepdim=True) + eps)

                    p.data.mul_(1 - lr * wd)
                    adjusted_lr = lr * 0.2 * math.sqrt(float(m.shape[dim]))
                    delta_fp32 = u.mul(-adjusted_lr)
                    p.data.add_(delta_fp32.to(dtype=p.dtype))
                else:
                    # AdamW fallback branch.
                    g_fp32 = g.detach().float()
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g_fp32)
                        state["moment2"] = torch.zeros_like(g_fp32)

                    state["step"] += 1
                    step = int(state["step"])
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.mul_(beta1).add_(g_fp32, alpha=1 - beta1)
                    buf2.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1 - beta2)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = adamw_lr / max(bias_correction1, 1e-12)
                    denom = buf2.sqrt().div_(math.sqrt(max(bias_correction2, 1e-12))).add_(adamw_eps)
                    delta_fp32 = buf1.div(denom).mul_(-step_size)
                    if wd > 0.0:
                        delta_fp32.add_(p.data.detach().float(), alpha=-adamw_lr * wd)

                    p.data.add_(delta_fp32.to(dtype=p.dtype))

                if self.track_update_stats:
                    if monitor_stats is None:
                        monitor_stats = init_component_stats(p.device)
                    name = self.param_name_by_id.get(id(p), "")
                    component = classify_param_name(name)
                    if component is not None:
                        accumulate_component_stats(
                            monitor_stats,
                            component=component,
                            update_tensor=delta_fp32,
                            param_tensor=p.data,
                        )
                    key_name = name if name else "<unnamed>"
                    monitor_details[f"detail/{key_name}"] = {
                        "update_norm": float(delta_fp32.norm().item()),
                        "update_rms_norm": float(delta_fp32.pow(2).mean().sqrt().item()),
                        "update_max": float(delta_fp32.max().item()),
                        "update_min": float(delta_fp32.min().item()),
                    }

            group["steps"] = int(group.get("steps", 0)) + 1

        if self.track_update_stats and monitor_stats is not None:
            self.last_update_stats = finalize_component_stats(monitor_stats)
            self.last_update_details = monitor_details
        else:
            self.last_update_stats = {}
            self.last_update_details = {}

        return loss