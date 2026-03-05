from __future__ import annotations

from typing import Dict, Optional, Tuple

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


def _coerce_siqa_example(ex: dict) -> Tuple[str, list[str], int]:
    """
    Match FineWeb blog's `lighteval_tasks.py` SIQA prompt:

    - query: `line["context"] + " " + line["question"]`
    - choices: `[" " + answerA, " " + answerB, " " + answerC]`
    - gold_index: int(label) - 1   (label is 1..3)
    """
    context = str(ex.get("context") or "").strip()
    question = str(ex.get("question") or "").strip()
    if not question:
        raise ValueError("Malformed Social IQa example: empty question")

    prompt = (context + " " + question).strip()
    choices = [str(ex.get("answerA") or ""), str(ex.get("answerB") or ""), str(ex.get("answerC") or "")]
    if any(c.strip() == "" for c in choices):
        raise ValueError("Malformed Social IQa example: empty answer option")

    gold = int(ex.get("label")) - 1
    if gold not in (0, 1, 2):
        raise ValueError(f"Bad label: {ex.get('label')!r}")
    return prompt, choices, gold


class _SiqaTask(MultipleChoiceTask):
    task_name = "siqa"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        prompt, choices, gold = _coerce_siqa_example(doc)
        return prompt, [f" {c.strip()}" for c in choices], gold


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
    # FineWeb uses `lighteval/siqa` to avoid script-based dataset loading.
    ds = load_dataset("lighteval/siqa", "default", split=split)
    task = _SiqaTask()
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"siqa:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))
    return {
        "task": "siqa",
        "split": split,
        "accuracy": acc_norm,
        "accuracy_raw": acc,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }

