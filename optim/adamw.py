from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor

from optim.monitoring import (
    accumulate_component_stats,
    classify_param_name,
    finalize_component_stats,
    init_component_stats,
    update_fro_and_rms,
)


class MonitoredAdamW(torch.optim.Optimizer):
    """
    AdamW with optional per-component update statistics for monitoring.

    The optimization rule follows decoupled weight decay:
      theta <- theta + delta
      delta = -lr * weight_decay * theta - step_size * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, Tensor]],
        *,
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        named_params = [(n, p) for n, p in named_params if p.requires_grad]
        if not named_params:
            raise ValueError("MonitoredAdamW received no trainable parameters")

        params = [p for _, p in named_params]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.param_name_by_id: Dict[int, str] = {id(p): n for n, p in named_params}
        self.track_update_stats: bool = False
        self.last_update_stats: Dict[str, Dict[str, float | int]] = {}
        self.last_update_details: Dict[str, Dict[str, float | int]] = {}

    def set_monitoring_enabled(self, enabled: bool = True) -> None:
        self.track_update_stats = bool(enabled)
        if not self.track_update_stats:
            self.last_update_stats = {}
            self.last_update_details = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        monitor_stats = None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MonitoredAdamW does not support sparse gradients")

                if monitor_stats is None and self.track_update_stats:
                    monitor_stats = init_component_stats(param.device)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                grad_fp32 = grad.detach().float()
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                delta_fp32 = exp_avg.div(denom).mul_(-step_size)

                if weight_decay > 0.0:
                    delta_fp32.add_(param.detach().float(), alpha=-lr * weight_decay)

                param.data.add_(delta_fp32.to(dtype=param.dtype))

                if self.track_update_stats and monitor_stats is not None:
                    name = self.param_name_by_id.get(id(param), "")
                    component = classify_param_name(name)
                    if component is not None:
                        accumulate_component_stats(
                            monitor_stats,
                            component=component,
                            update_tensor=delta_fp32,
                            param_tensor=param.data,
                        )
                    update_fro, update_rms = update_fro_and_rms(delta_fp32)
                    uf = float(update_fro.item())
                    ur = float(update_rms.item())
                    self.last_update_details[f'sec/{name}'] = {
                        'update_fro_norm': uf,
                        'update_rms_norm': ur,
                    }

        if self.track_update_stats and monitor_stats is not None:
            self.last_update_stats = finalize_component_stats(monitor_stats)
        else:
            self.last_update_stats = {}
            self.last_update_details = {}

        return loss