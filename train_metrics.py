from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _tensor_rms(x: torch.Tensor) -> torch.Tensor:
    # RMS over all elements, computed in FP32 for stability.
    x = x.detach()
    if x.dtype != torch.float32:
        x = x.float()
    return x.pow(2).mean().sqrt()


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


class ActivationMonitor:
    """
    Collect activation RMS summaries via forward hooks.

    Logged components:
    - attn / mlp residual-branch outputs
    - embedding output
    - lm_head output (logits)
    """

    def __init__(
        self,
        *,
        blocks: nn.ModuleList,
        embedding_module: nn.Module,
        lm_head_module: nn.Module,
        device: torch.device,
        log_activation_norm: bool,
        log_activation_update_norm: bool,
    ):
        self.blocks = blocks
        self.embedding_module = embedding_module
        self.lm_head_module = lm_head_module
        self.device = device
        self.log_activation_norm = bool(log_activation_norm)
        self.log_activation_update_norm = bool(log_activation_update_norm)

        self.capture_enabled: bool = False
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        self.n_layer = len(blocks)
        self.act_sum = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)
        self.act_cnt = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)
        self.attn_sum = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)
        self.attn_cnt = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)
        self.mlp_sum = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)
        self.mlp_cnt = torch.zeros(self.n_layer, device=self.device, dtype=torch.float32)

        self.embedding_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self.embedding_cnt = torch.zeros((), device=self.device, dtype=torch.float32)
        self.embedding_max = torch.zeros((), device=self.device, dtype=torch.float32)

        self.lm_head_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self.lm_head_cnt = torch.zeros((), device=self.device, dtype=torch.float32)
        self.lm_head_max = torch.zeros((), device=self.device, dtype=torch.float32)

        self._register_hooks()

    def _register_hooks(self) -> None:
        capture_components = self.log_activation_norm or self.log_activation_update_norm

        for i, block in enumerate(self.blocks):
            if self.log_activation_norm:
                self.handles.append(block.register_forward_hook(self._make_block_hook(i)))
            if capture_components:
                if hasattr(block, "attn"):
                    self.handles.append(block.attn.register_forward_hook(self._make_attn_hook(i)))
                if hasattr(block, "mlp"):
                    self.handles.append(block.mlp.register_forward_hook(self._make_mlp_hook(i)))

        if capture_components:
            self.handles.append(self.embedding_module.register_forward_hook(self._embedding_hook))
            self.handles.append(self.lm_head_module.register_forward_hook(self._lm_head_hook))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def begin_step(self, *, capture_enabled: bool) -> None:
        self.capture_enabled = bool(capture_enabled)
        if not self.capture_enabled:
            return
        self.act_sum.zero_()
        self.act_cnt.zero_()
        self.attn_sum.zero_()
        self.attn_cnt.zero_()
        self.mlp_sum.zero_()
        self.mlp_cnt.zero_()
        self.embedding_sum.zero_()
        self.embedding_cnt.zero_()
        self.embedding_max.zero_()
        self.lm_head_sum.zero_()
        self.lm_head_cnt.zero_()
        self.lm_head_max.zero_()

    def end_step(self) -> Dict[str, float]:
        if not self.capture_enabled:
            return {}

        out: Dict[str, float] = {}
        act_layer_mean = None
        attn_layer_mean = self.attn_sum / self.attn_cnt.clamp_min(1.0)
        mlp_layer_mean = self.mlp_sum / self.mlp_cnt.clamp_min(1.0)

        if self.log_activation_norm:
            # Keep existing block-level activation summary for backward compatibility.
            act_layer_mean = self.act_sum / self.act_cnt.clamp_min(1.0)
            out["activation/rms_mean"] = float(act_layer_mean.mean().item())
            out["activation/rms_max"] = float(act_layer_mean.max().item())

        attn_rms_mean = attn_layer_mean.mean()
        attn_rms_max = attn_layer_mean.max()
        mlp_rms_mean = mlp_layer_mean.mean()
        mlp_rms_max = mlp_layer_mean.max()

        self._add_component_metrics(out, component="attn", rms_mean=attn_rms_mean, rms_max=attn_rms_max)
        self._add_component_metrics(out, component="mlp", rms_mean=mlp_rms_mean, rms_max=mlp_rms_max)

        if self.embedding_cnt.item() > 0:
            emb_rms_mean = self.embedding_sum / self.embedding_cnt.clamp_min(1.0)
            emb_rms_max = self.embedding_max
            self._add_component_metrics(
                out,
                component="embedding",
                rms_mean=emb_rms_mean,
                rms_max=emb_rms_max,
            )

        if self.lm_head_cnt.item() > 0:
            lm_head_rms_mean = self.lm_head_sum / self.lm_head_cnt.clamp_min(1.0)
            lm_head_rms_max = self.lm_head_max
            self._add_component_metrics(
                out,
                component="lm_head",
                rms_mean=lm_head_rms_mean,
                rms_max=lm_head_rms_max,
            )
            # Alias logits keys for easier interpretation.
            self._add_component_metrics(
                out,
                component="logits",
                rms_mean=lm_head_rms_mean,
                rms_max=lm_head_rms_max,
            )

        if self.log_activation_update_norm:
            # Backward-compatible keys.
            out["activation/attn_update_rms_mean"] = float(attn_rms_mean.item())
            out["activation/attn_update_rms_max"] = float(attn_rms_max.item())
            out["activation/mlp_update_rms_mean"] = float(mlp_rms_mean.item())
            out["activation/mlp_update_rms_max"] = float(mlp_rms_max.item())

            if self.log_activation_norm and act_layer_mean is not None:
                denom = act_layer_mean.clamp_min(1e-12)
                out["activation/attn_update_over_act_mean"] = float((attn_layer_mean / denom).mean().item())
                out["activation/mlp_update_over_act_mean"] = float((mlp_layer_mean / denom).mean().item())

        return out

    def _add_component_metrics(
        self,
        out: Dict[str, float],
        *,
        component: str,
        rms_mean: torch.Tensor,
        rms_max: torch.Tensor,
    ) -> None:
        if self.log_activation_norm:
            out[f"activation/act_{component}_rms_mean"] = float(rms_mean.item())
            out[f"activation/act_{component}_rms_max"] = float(rms_max.item())
        if self.log_activation_update_norm:
            out[f"activation/act_{component}_update_rms_mean"] = float(rms_mean.item())
            out[f"activation/act_{component}_update_rms_max"] = float(rms_max.item())

    def _make_block_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not self.capture_enabled:
                return
            if not torch.is_grad_enabled():
                return
            out = _first_tensor(output)
            if out is None:
                return
            rms = _tensor_rms(out)
            self.act_sum[layer_idx].add_(rms)
            self.act_cnt[layer_idx].add_(1.0)

        return hook

    def _make_attn_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not self.capture_enabled:
                return
            if not torch.is_grad_enabled():
                return
            out = _first_tensor(output)
            if out is None:
                return
            rms = _tensor_rms(out)
            self.attn_sum[layer_idx].add_(rms)
            self.attn_cnt[layer_idx].add_(1.0)

        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not self.capture_enabled:
                return
            if not torch.is_grad_enabled():
                return
            out = _first_tensor(output)
            if out is None:
                return
            rms = _tensor_rms(out)
            self.mlp_sum[layer_idx].add_(rms)
            self.mlp_cnt[layer_idx].add_(1.0)

        return hook

    def _embedding_hook(self, _module, _inputs, output):
        if not self.capture_enabled:
            return
        if not torch.is_grad_enabled():
            return
        out = _first_tensor(output)
        if out is None:
            return
        rms = _tensor_rms(out)
        self.embedding_sum.add_(rms)
        self.embedding_cnt.add_(1.0)
        self.embedding_max = torch.maximum(self.embedding_max, rms)

    def _lm_head_hook(self, _module, _inputs, output):
        if not self.capture_enabled:
            return
        if not torch.is_grad_enabled():
            return
        out = _first_tensor(output)
        if out is None:
            return
        rms = _tensor_rms(out)
        self.lm_head_sum.add_(rms)
        self.lm_head_cnt.add_(1.0)
        self.lm_head_max = torch.maximum(self.lm_head_max, rms)


class ParamUpdateMonitor:
    """
    Collect full-parameter update statistics from optimizers.

    Optimizers can optionally expose:
      - set_monitoring_enabled(enabled: bool)
      - last_update_stats: Dict[component, Dict[str, scalar]]
    """

    _COMPONENTS = ("attn", "mlp", "lm_head", "embedding")

    def __init__(self, *, optimizers: List[torch.optim.Optimizer], device: torch.device):
        self.optimizers = list(optimizers)
        self.device = device
        self.capture_enabled: bool = False
        self.set_capture_enabled(False)

    def set_capture_enabled(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)
        for opt in self.optimizers:
            enable = getattr(opt, "set_monitoring_enabled", None)
            if callable(enable):
                enable(self.capture_enabled)

    @torch.no_grad()
    def capture_pre(self) -> None:
        # Kept as a no-op for backward compatibility with older callsites.
        return None

    @torch.no_grad()
    def capture_post_and_compute(self) -> Dict[str, float]:
        if not self.capture_enabled:
            return {}

        aggregate = {
            key: {
                "count": 0,
                "update_fro_sum": 0.0,
                "update_fro_max": 0.0,
                "update_rms_sum": 0.0,
                "update_rms_max": 0.0,
                "param_rms_sum": 0.0,
                "param_rms_max": 0.0,
            }
            for key in self._COMPONENTS
        }

        for opt in self.optimizers:
            stats = getattr(opt, "last_update_stats", None)
            if not isinstance(stats, dict):
                continue

            for key, bucket in stats.items():
                if key not in aggregate or not isinstance(bucket, dict):
                    continue
                count = int(bucket.get("count", 0))
                if count <= 0:
                    continue

                dst = aggregate[key]
                dst["count"] += count
                dst["update_fro_sum"] += float(bucket.get("update_fro_sum", 0.0))
                dst["update_rms_sum"] += float(bucket.get("update_rms_sum", 0.0))
                dst["param_rms_sum"] += float(bucket.get("param_rms_sum", 0.0))
                dst["update_fro_max"] = max(dst["update_fro_max"], float(bucket.get("update_fro_max", 0.0)))
                dst["update_rms_max"] = max(dst["update_rms_max"], float(bucket.get("update_rms_max", 0.0)))
                dst["param_rms_max"] = max(dst["param_rms_max"], float(bucket.get("param_rms_max", 0.0)))

        out: Dict[str, float] = {}
        for key, bucket in aggregate.items():
            count = int(bucket["count"])
            if count <= 0:
                continue

            denom = float(count)
            out[f"param/{key}_update_fro_mean"] = float(bucket["update_fro_sum"] / denom)
            out[f"param/{key}_update_fro_max"] = float(bucket["update_fro_max"])
            out[f"param/{key}_update_fro_norm_mean"] = out[f"param/{key}_update_fro_mean"]
            out[f"param/{key}_update_fro_norm_max"] = out[f"param/{key}_update_fro_max"]
            out[f"param/{key}_update_rms_mean"] = float(bucket["update_rms_sum"] / denom)
            out[f"param/{key}_update_rms_max"] = float(bucket["update_rms_max"])
            out[f"param/{key}_update_rms_norm_mean"] = out[f"param/{key}_update_rms_mean"]
            out[f"param/{key}_update_rms_norm_max"] = out[f"param/{key}_update_rms_max"]
            out[f"param/{key}_rms_mean"] = float(bucket["param_rms_sum"] / denom)
            out[f"param/{key}_rms_max"] = float(bucket["param_rms_max"])

            if key == "lm_head":
                # Alias for readability in dashboards.
                out["param/logits_update_fro_mean"] = out[f"param/{key}_update_fro_mean"]
                out["param/logits_update_fro_max"] = out[f"param/{key}_update_fro_max"]
                out["param/logits_update_fro_norm_mean"] = out[f"param/{key}_update_fro_norm_mean"]
                out["param/logits_update_fro_norm_max"] = out[f"param/{key}_update_fro_norm_max"]
                out["param/logits_update_rms_mean"] = out[f"param/{key}_update_rms_mean"]
                out["param/logits_update_rms_max"] = out[f"param/{key}_update_rms_max"]
                out["param/logits_update_rms_norm_mean"] = out[f"param/{key}_update_rms_norm_mean"]
                out["param/logits_update_rms_norm_max"] = out[f"param/{key}_update_rms_norm_max"]
                out["param/logits_rms_mean"] = out[f"param/{key}_rms_mean"]
                out["param/logits_rms_max"] = out[f"param/{key}_rms_max"]

        return out

