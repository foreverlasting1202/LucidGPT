from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from evals.tokenizer import encode


@dataclass(frozen=True)
class ScoredOption:
    text: str
    loglikelihood: float
    num_tokens: int


def _truncate_context_for_continuation(
    *,
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Ensure (context + continuation) fits into a model window.

    For next-token scoring we need (len(full) - 1) <= max_seq_len, i.e. len(full) <= max_seq_len + 1.
    We always keep the full continuation, and keep as much right-side context as fits.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    cont_len = len(continuation_ids)
    if cont_len == 0:
        raise ValueError("continuation_ids is empty")

    max_full_len = max_seq_len + 1
    if cont_len + 1 > max_full_len:
        raise ValueError(
            f"Continuation too long for max_seq_len={max_seq_len}: "
            f"need at least 1 context token + {cont_len} continuation tokens"
        )

    ctx = list(context_ids)
    cont = list(continuation_ids)
    full_len = len(ctx) + len(cont)
    if full_len <= max_full_len:
        return ctx, cont

    # keep full continuation + as many context tokens as fit (at least 1)
    ctx_keep = max_full_len - len(cont)
    if ctx_keep < 1:
        raise RuntimeError("unreachable: ctx_keep < 1 after checks")
    return ctx[-ctx_keep:], cont


@torch.inference_mode()
def loglikelihood_of_continuation(
    model,
    *,
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    max_seq_len: int,
    length_normalize: bool,
) -> float:
    """
    Compute log p(continuation | context) under an autoregressive LM.
    """
    ctx, cont = _truncate_context_for_continuation(
        context_ids=context_ids, continuation_ids=continuation_ids, max_seq_len=max_seq_len
    )

    full = ctx + cont
    # idx predicts next token; targets are shifted by 1
    idx = torch.tensor(full[:-1], dtype=torch.long, device=device)[None, :]
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)[None, :]

    cont_len = len(cont)
    # The continuation tokens correspond to the last `cont_len` positions in `targets`.
    cont_targets = targets[:, -cont_len:]

    if device.type == "cuda" and autocast_dtype is not None:
        ctx_mgr = torch.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        ctx_mgr = torch.no_grad()

    with ctx_mgr:
        # We only need logits for the last `cont_len` positions.
        logits = model.forward_logits(idx, logits_last_k=cont_len)  # (1, cont_len, vocab)
        logprobs = F.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(dim=-1, index=cont_targets[..., None]).squeeze(-1)  # (1, cont_len)
        ll = float(gathered.sum().item())

    if length_normalize:
        ll /= cont_len
    return ll


@torch.inference_mode()
def score_text_options(
    model,
    *,
    prompt: str,
    options: Sequence[str],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    max_seq_len: int,
    length_normalize: bool,
) -> List[ScoredOption]:
    """
    Score several text continuations under the same prompt.
    """
    ctx_ids = encode(prompt)
    out: List[ScoredOption] = []
    for opt in options:
        cont_ids = encode(opt)
        ll = loglikelihood_of_continuation(
            model,
            context_ids=ctx_ids,
            continuation_ids=cont_ids,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            length_normalize=length_normalize,
        )
        out.append(ScoredOption(text=opt, loglikelihood=ll, num_tokens=len(cont_ids)))
    return out


def argmax_scored(scored: Sequence[ScoredOption]) -> int:
    if len(scored) == 0:
        raise ValueError("empty scored list")
    best_i = 0
    best = -math.inf
    for i, s in enumerate(scored):
        if s.loglikelihood > best:
            best = s.loglikelihood
            best_i = i
    return best_i

