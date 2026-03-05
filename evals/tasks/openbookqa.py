from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


def _coerce_openbookqa_example(ex: dict) -> Tuple[str, list[str], int]:
    """
    Match lm-eval-harness `openbookqa` (multiple_choice):
      - doc_to_text: question_stem
      - doc_to_choice: choices.text
      - doc_to_target: index(answerKey in choices.label)
    """
    question = str(ex.get("question_stem") or "").strip()
    raw_choices = ex.get("choices") or {}
    answer_key = str(ex.get("answerKey") or "").lstrip().strip()

    if not question:
        raise ValueError("Malformed OpenBookQA example: empty question_stem")

    if not isinstance(raw_choices, dict):
        raise ValueError("Malformed OpenBookQA example: choices is not a dict")

    labels = list(raw_choices.get("label") or [])
    texts = list(raw_choices.get("text") or [])
    if len(texts) == 0 or len(texts) != len(labels):
        raise ValueError("Malformed OpenBookQA example: bad choices.{label,text}")

    labels_s = [str(x).strip() for x in labels]
    texts_s = [str(x).strip() for x in texts]
    if not answer_key:
        raise ValueError("Malformed OpenBookQA example: empty answerKey")
    try:
        gold = labels_s.index(answer_key)
    except ValueError as e:
        raise ValueError(f"answerKey not in choices.label: answerKey={answer_key!r} labels={labels_s!r}") from e

    return question, texts_s, int(gold)


class _OpenBookQATask(MultipleChoiceTask):
    task_name = "openbookqa"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        prompt, choices, gold = _coerce_openbookqa_example(doc)
        return prompt, [f" {c}" for c in choices], gold


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    split: str = "test",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    """
    OpenBookQA multiple-choice evaluation.

    lm-eval-harness uses `allenai/openbookqa` config `main`.
    """
    ds = load_dataset("openbookqa", "main", split=split)
    task = _OpenBookQATask()
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"openbookqa:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))
    return {
        "task": "openbookqa",
        "split": split,
        "accuracy": acc_norm,
        "accuracy_raw": acc,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }

