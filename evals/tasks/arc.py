from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


def _coerce_arc_example(ex: dict) -> Tuple[str, list[str], list[str], str]:
    """
    Coerce an ARC example into:
    - question: str
    - choice_texts: list[str]
    - choice_labels: list[str]
    - answer_key: str (label)

    Matches lm-eval-harness ARC task expectations.
    """
    q = ex.get("question")
    if isinstance(q, dict):
        question = str(q.get("stem") or "").strip()
        raw_choices = q.get("choices") or []
    else:
        question = str(q or "").strip()
        raw_choices = ex.get("choices") or []

    choice_labels: list[str] = []
    choice_texts: list[str] = []

    # HF may represent choices either as:
    # - {"label": [...], "text": [...]} (dict-of-lists style)  [common in `allenai/ai2_arc`]
    # - list[{label,text}, ...] (struct-of-structs style)
    if isinstance(raw_choices, dict):
        labels = list(raw_choices.get("label") or [])
        texts = list(raw_choices.get("text") or [])
        for lab, txt in zip(labels, texts):
            label = str(lab or "").strip()
            text = str(txt or "").strip()
            if label and text:
                choice_labels.append(label)
                choice_texts.append(text)
    else:
        for c in raw_choices:
            if not isinstance(c, dict):
                continue
            label = str(c.get("label") or "").strip()
            text = str(c.get("text") or "").strip()
            if label and text:
                choice_labels.append(label)
                choice_texts.append(text)

    answer_key = str(ex.get("answerKey") or "").strip()
    if not question or not choice_texts or not answer_key:
        raise ValueError("Malformed ARC example")
    if len(choice_texts) != len(choice_labels):
        raise ValueError("ARC choices have mismatched label/text lengths")
    return question, choice_texts, choice_labels, answer_key


def _build_prompt(question: str) -> str:
    # Match lm-eval-harness: do NOT inline options into the prompt; score answer texts directly.
    return f"Question: {question}\nAnswer:"


class _ARCTask(MultipleChoiceTask):
    task_name = "arc"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def __init__(self, *, config: str):
        self.config = str(config)

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        question, choice_texts, choice_labels, answer_key = _coerce_arc_example(doc)
        prompt = _build_prompt(question)
        try:
            gold = choice_labels.index(answer_key)
        except ValueError as e:
            raise ValueError(
                f"answerKey not in choices: answerKey={answer_key!r} labels={choice_labels!r}"
            ) from e
        # FineWeb's lighteval tasks prefix each choice with a space.
        return prompt, [f" {str(t).lstrip()}" for t in choice_texts], int(gold)


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    challenge: bool = True,
    split: str = "test",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("ai2_arc", config, split=split)
    task = _ARCTask(config=config)
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,  # HF Dataset supports __len__/__getitem__
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"arc:{config}:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))

    # Preserve the repo's previous output schema, while adding lm_eval-style fields.
    out: Dict[str, object] = {
        "task": "arc",
        "config": config,
        "split": split,
        "accuracy": acc,
        "accuracy_norm": acc_norm,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }
    return out

