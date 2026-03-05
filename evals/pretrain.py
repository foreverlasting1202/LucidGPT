from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch

from data_loader import DistributedDataLoader


@torch.inference_mode()
def run(
    *,
    model,
    input_bin: str,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    eval_tokens: int,
) -> Dict[str, object]:
    """
    Compute "pretrain-style" validation metrics on tokenized `.bin` shards:
    - loss (cross-entropy, nats/token)
    - perplexity
    - bits_per_token
    - token_accuracy (top-1 next-token accuracy)
    - throughput (tokens/sec)
    """
    if eval_tokens <= 0:
        raise ValueError("--eval_tokens must be > 0")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("--batch_size and --sequence_length must be > 0")

    tokens_per_batch = batch_size * sequence_length
    steps = eval_tokens // tokens_per_batch
    if steps <= 0:
        raise ValueError(
            f"eval_tokens={eval_tokens} is too small for batch_size*sequence_length={tokens_per_batch}"
        )
    total_eval_tokens = steps * tokens_per_batch

    loader = DistributedDataLoader(input_bin, batch_size, sequence_length, 0, 1)
    loader.reset()

    model.eval()

    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    if device.type == "cuda" and autocast_dtype is not None:
        ctx_mgr = torch.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        ctx_mgr = torch.no_grad()

    t0 = time.time()
    for _ in range(steps):
        x, y = loader.next_batch(device=device)
        with ctx_mgr:
            logits, loss = model(x, y, return_logits=True)
        # `loss` is mean over all tokens; convert to total NLL.
        total_nll += float(loss.item()) * tokens_per_batch
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total_tokens += tokens_per_batch

    dt = time.time() - t0

    avg_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    bpt = avg_loss / math.log(2)
    acc = total_correct / max(1, total_tokens)
    tps = total_tokens / max(1e-9, dt)

    out: Dict[str, object] = {
        "task": "pretrain_metrics",
        "input_bin": input_bin,
        "eval_tokens": int(total_tokens),
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "loss": avg_loss,
        "perplexity": ppl,
        "bits_per_token": bpt,
        "token_accuracy": acc,
        "tokens_per_second": tps,
        "seconds": float(dt),
    }

    if device.type == "cuda":
        out["cuda_max_memory_allocated_mib"] = int(torch.cuda.max_memory_allocated() // 1024 // 1024)

    return out

