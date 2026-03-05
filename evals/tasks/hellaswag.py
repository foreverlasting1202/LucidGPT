from __future__ import annotations

import re
from typing import Dict, Optional

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


def _preprocess(text: str) -> str:
    """
    Match lm-eval-harness preprocessing for HellaSwag.

    Notes:
    - Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    """
    text = str(text).strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _doc_to_query_choices_gold(ex: dict) -> tuple[str, list[str], int]:
    """
    lm-eval-harness constructs:
    - query: preprocess(activity_label + ": " + (ctx_a + " " + ctx_b.capitalize()))
    - choices: preprocess(endings[i])
    - gold: int(label)
    """
    ctx_a = str(ex.get("ctx_a") or "")
    ctx_b = str(ex.get("ctx_b") or "")
    act = str(ex.get("activity_label") or "")
    ctx = ctx_a + " " + ctx_b.capitalize()

    query = _preprocess(act + ": " + ctx)

    endings = list(ex["endings"])
    if len(endings) != 4:
        raise ValueError(f"Expected 4 endings, got {len(endings)}")
    choices = [_preprocess(e) for e in endings]

    gold = int(ex["label"])
    if not (0 <= gold < 4):
        raise ValueError(f"Bad label: {gold}")

    return query, choices, gold


class _HellaSwagTask(MultipleChoiceTask):
    task_name = "hellaswag"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        query, choices, gold = _doc_to_query_choices_gold(doc)
        # Match lighteval: prefix each ending with a space.
        return query, [" " + c for c in choices], int(gold)


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    split: str = "validation",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    ds = load_dataset("hellaswag", split=split)
    task = _HellaSwagTask()
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"hellaswag:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    # Preserve repo schema: historically `accuracy` here was length-normalized.
    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))
    return {
        "task": "hellaswag",
        "split": split,
        "accuracy": acc_norm,
        "accuracy_raw": acc,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }

