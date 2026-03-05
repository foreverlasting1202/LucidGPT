from __future__ import annotations

from typing import Dict, Optional, Tuple

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


_LETTERS = ("A", "B", "C", "D", "E")


def _coerce_commonsenseqa_example(ex: dict) -> Tuple[str, list[str], int]:
    """
    Match FineWeb blog's `lighteval_tasks.py` commonsense_qa prompt:

    - query: `line["question"]`
    - choices: `[" " + c for c in line["choices"]["text"]]`
    - gold_index: index(answerKey in ["A","B","C","D","E"])
    """
    question = str(ex.get("question") or "").strip()
    raw_choices = ex.get("choices") or {}
    answer_key = str(ex.get("answerKey") or "").strip()

    if not question:
        raise ValueError("Malformed CommonsenseQA example: empty question")
    if not isinstance(raw_choices, dict):
        raise ValueError("Malformed CommonsenseQA example: choices is not a dict")

    texts = list(raw_choices.get("text") or [])
    if len(texts) != 5:
        raise ValueError(f"Expected 5 choices, got {len(texts)}")
    texts_s = [str(t).strip() for t in texts]

    if answer_key not in _LETTERS:
        raise ValueError(f"Bad answerKey: {answer_key!r}")
    gold = _LETTERS.index(answer_key)
    prompt = question
    options = [f" {t}" for t in texts_s]
    return prompt, options, int(gold)


class _CommonsenseQATask(MultipleChoiceTask):
    task_name = "commonsense_qa"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        prompt, choices, gold = _coerce_commonsenseqa_example(doc)
        return prompt, choices, gold


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
    ds = load_dataset("tau/commonsense_qa", split=split)
    task = _CommonsenseQATask()
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"commonsense_qa:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))
    return {
        "task": "commonsense_qa",
        "split": split,
        "accuracy": acc_norm,
        "accuracy_raw": acc,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }

