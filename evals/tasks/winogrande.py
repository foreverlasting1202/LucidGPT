from __future__ import annotations

from typing import Dict, Optional

from tqdm import tqdm

from evals.hf_datasets import load_dataset
from evals.scoring import loglikelihood_of_continuation
from evals.tokenizer import encode


def _coerce_winogrande_example(doc: dict) -> tuple[list[str], str, int]:
    """
    Match lm-eval-harness Winogrande "partial evaluation":
    - Score the *suffix after the blank* conditioned on (prefix + option).
    - Choose the option that maximizes log p(suffix | prefix+option).

    Reference: lm_eval/tasks/winogrande/preprocess_winogrande.py
    """
    sentence = str(doc["sentence"])
    if "_" not in sentence:
        raise ValueError("Expected '_' placeholder in Winogrande sentence")
    idx = sentence.index("_")

    # Two candidate contexts (prefix + option), without the suffix.
    opt1 = str(doc["option1"])
    opt2 = str(doc["option2"])
    contexts = [sentence[:idx] + opt1, sentence[:idx] + opt2]

    # Common continuation (suffix after the blank), stripped; we'll insert a delimiter when scoring.
    suffix = sentence[idx + 1 :].strip()

    ans = str(doc["answer"])
    answer_to_num = {"1": 0, "2": 1}
    if ans not in answer_to_num:
        raise ValueError(f"Bad answer: {ans!r}")
    gold = answer_to_num[ans]
    return contexts, suffix, gold


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    config: str = "winogrande_xl",
    split: str = "validation",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    """
    Winogrande is a 2-way multiple-choice coreference-style task.

    HF dataset: `winogrande`, with configs like `winogrande_xl`.
    Fields:
    - sentence: str with '_' placeholder
    - option1: str
    - option2: str
    - answer: str in {"1","2"}
    """
    import torch
    import torch.distributed as dist

    ds = load_dataset("winogrande", config, split=split)
    n_total = len(ds) if limit is None else min(int(limit), len(ds))
    distributed = (
        ddp_rank is not None
        and ddp_world_size is not None
        and int(ddp_world_size) > 1
        and dist.is_available()
        and dist.is_initialized()
    )
    if distributed:
        rank = int(ddp_rank)
        world_size = int(ddp_world_size)
        indices = range(rank, n_total, world_size)
    else:
        indices = range(n_total)

    correct = 0
    correct_norm = 0
    local_total = 0
    cont_cache: dict[str, list[int]] = {}

    # Match lm-eval-harness default target delimiter behavior.
    delim = " "

    it = indices
    if not distributed:
        it = tqdm(it, desc=f"winogrande:{config}:{split}", leave=True)
    for i in it:
        local_total += 1
        doc = ds[i]
        contexts, suffix, gold = _coerce_winogrande_example(doc)

        cont_ids = cont_cache.get(suffix)
        if cont_ids is None:
            cont_ids = encode(suffix)
            cont_cache[suffix] = cont_ids

        scores = []
        scores_norm = []
        for ctx in contexts:
            ctx_ids = encode(ctx + delim)
            ll = loglikelihood_of_continuation(
                model,
                context_ids=ctx_ids,
                continuation_ids=cont_ids,
                device=torch.device(device),
                autocast_dtype=autocast_dtype,
                max_seq_len=max_seq_len,
                length_normalize=False,
            )
            scores.append(ll)
            scores_norm.append(ll / max(1, len(cont_ids)))

        pred = 0 if scores[0] >= scores[1] else 1
        pred_norm = 0 if scores_norm[0] >= scores_norm[1] else 1
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)

    total = int(local_total)
    if distributed:
        dev = torch.device(device)
        backend = dist.get_backend()
        reduce_device = dev if backend == "nccl" else torch.device("cpu")
        tensor = torch.tensor([int(correct), int(correct_norm), int(local_total)], dtype=torch.long, device=reduce_device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        correct = int(tensor[0].item())
        correct_norm = int(tensor[1].item())
        total = int(tensor[2].item())

    acc = correct / total if total else 0.0
    acc_norm = correct_norm / total if total else 0.0
    return {
        "task": "winogrande",
        "config": config,
        "split": split,
        # Preserve repo schema: historically `accuracy` here was length-normalized.
        "accuracy": acc_norm,
        "accuracy_raw": acc,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(total),
        "correct": int(correct),
        "correct_norm": int(correct_norm),
    }

