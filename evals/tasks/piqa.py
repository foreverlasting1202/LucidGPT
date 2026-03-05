from __future__ import annotations

from typing import Dict, Optional

from evals.hf_datasets import load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


class _PIQATask(MultipleChoiceTask):
    task_name = "piqa"
    # FineWeb's lighteval tasks include a leading space in each choice.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        # Match lm-eval-harness formatting for PIQA:
        #   doc_to_text:  "Question: {{goal}}\nAnswer:"
        #   doc_to_choice: {{[sol1, sol2]}}
        # (i.e. score sol1/sol2 as continuations after "Answer:").
        goal = str(doc["goal"])
        sol1 = str(doc["sol1"])
        sol2 = str(doc["sol2"])
        gold = int(doc["label"])
        if gold not in (0, 1):
            raise ValueError(f"Bad label: {gold}")
        prompt = f"Question: {goal}\nAnswer:"
        options = [f" {sol1.lstrip()}", f" {sol2.lstrip()}"]
        return prompt, options, gold


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
    """
    PIQA is a 2-way multiple-choice *completion* task.

    Pythia (and most public benchmarks) evaluate PIQA by scoring the two candidate
    solutions as continuations of the `goal` prompt (not by asking the model to emit
    a letter like "A"/"B").

    Fields:
    - goal: str (prompt)
    - sol1: str (choice 0 continuation)
    - sol2: str (choice 1 continuation)
    - label: int in {0,1}
    """
    ds = load_dataset("piqa", split=split)
    task = _PIQATask()
    counts, payload = evaluate_multiple_choice(
        task=task,
        docs=ds,
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        max_seq_len=max_seq_len,
        limit=limit,
        nshot=0,
        desc=f"piqa:{split}",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    acc = float(payload["acc"])
    acc_norm = float(payload.get("acc_norm", acc))
    return {
        "task": "piqa",
        "split": split,
        # Preserve repo schema: `accuracy` is the raw (unnormalized) accuracy.
        "accuracy": acc,
        "accuracy_raw": acc,
        "accuracy_norm": acc_norm,
        "acc": acc,
        "acc_norm": acc_norm,
        "total": int(payload["total"]),
        "correct": int(counts.correct),
        "correct_norm": int(counts.correct_norm or 0),
    }

