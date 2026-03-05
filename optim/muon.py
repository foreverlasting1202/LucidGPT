import torch
import torch.distributed as dist
from typing import Dict, List, Optional

from optim.monitoring import (
    accumulate_component_stats,
    classify_param_name,
    finalize_component_stats,
    init_component_stats,
    update_fro_and_rms,
)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

def _maybe_compile(fn):
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return fn
    try:
        return compile_fn(fn)
    except Exception:
        return fn


zeropower_via_newtonschulz5 = _maybe_compile(zeropower_via_newtonschulz5)
zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        nesterov=True,
        backend='newtonschulz5',
        backend_steps=5,
        rank=0,
        world_size=1,
        weight_decay=0.0,
        param_names: Optional[List[str]] = None,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self.total_error = 0.0
        self.weight_decay = weight_decay
        self.track_update_stats: bool = False
        self.last_update_stats: Dict[str, Dict[str, float | int]] = {}
        self.last_update_details: Dict[str, Dict[str, float | int]] = {}

        flat_params = [p for group in self.param_groups for p in group['params']]
        self.param_name_by_id: Dict[int, str] = {}
        if param_names is not None:
            if len(param_names) != len(flat_params):
                raise ValueError("param_names length must match Muon params length")
            self.param_name_by_id = {id(p): n for p, n in zip(flat_params, param_names)}

    def set_monitoring_enabled(self, enabled: bool = True) -> None:
        self.track_update_stats = bool(enabled)
        if not self.track_update_stats:
            self.last_update_stats = {}
            self.last_update_details = {}

    def step(self):
        monitor_stats = None

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            first_param = next(iter(group['params']))
            updates_flat = torch.zeros(total_params, device=first_param.device, dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                delta_fp32 = g.float().mul(-lr)
                if self.weight_decay > 0:
                    delta_fp32.add_(p.data.float(), alpha=-lr * self.weight_decay)
                    p.data.mul_(1 - lr * self.weight_decay)
                p.data.add_(g, alpha=-lr)

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
                    update_fro, update_rms = update_fro_and_rms(delta_fp32)
                    uf = float(update_fro.item())
                    ur = float(update_rms.item())
                    self.last_update_details[f'sec/{name}'] = {
                        'update_fro_norm': uf,
                        'update_rms_norm': ur,
                    }
                curr_idx += p.numel()

        if self.track_update_stats and monitor_stats is not None:
            self.last_update_stats = finalize_component_stats(monitor_stats)
        else:
            self.last_update_stats = {}
            self.last_update_details = {}