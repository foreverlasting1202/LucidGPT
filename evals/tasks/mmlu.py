from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from evals.hf_datasets import get_dataset_config_names, load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


_CHOICE_LETTERS = ("A", "B", "C", "D")


def _format_subject(subject: str) -> str:
    # "high_school_physics" -> "high school physics"
    return subject.replace("_", " ")


def _get_splits_for_subject(subject: str):
    ds = load_dataset("cais/mmlu", subject)
    # Few-shot examples are typically from "dev". Keep a robust fallback order.
    if "dev" in ds:
        dev = ds["dev"]
    elif "validation" in ds:
        dev = ds["validation"]
    elif "train" in ds:
        dev = ds["train"]
    else:
        raise KeyError(f"Could not find a dev/validation/train split for subject={subject!r}")

    if "test" in ds:
        test = ds["test"]
    elif "validation" in ds:
        test = ds["validation"]
    else:
        raise KeyError(f"Could not find a test/validation split for subject={subject!r}")

    return dev, test


def _coerce_example(example: dict) -> tuple[str, List[str], int]:
    """
    MMLU examples (cais/mmlu) typically look like:
    - question: str
    - choices: List[str] (len=4)
    - answer: int in [0..3]
    """
    q = str(example["question"]).strip()
    choices = list(example["choices"])
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")
    ans = int(example["answer"])
    if not (0 <= ans < 4):
        raise ValueError(f"Bad answer index: {ans}")
    return q, [str(c).strip() for c in choices], ans


def _format_question_block(question: str, choices: Sequence[str]) -> str:
    """
    Match lm-eval-harness MMLU template:
      {{question.strip()}}
      A. {{choices[0]}}
      ...
      Answer:
    """
    lines: List[str] = []
    lines.append(str(question).strip())
    for i, c in enumerate(choices):
        lines.append(f"{_CHOICE_LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


class _MMLUSubjectTask(MultipleChoiceTask):
    task_name = "mmlu"

    def __init__(self, *, subject: str, dev_examples: Sequence[dict]):
        self.subject = str(subject)
        # Keep a deterministic ordering (HF datasets are already ordered).
        self._dev_examples = list(dev_examples)

    def fewshot_context(self, *, nshot: int) -> str:
        subj = _format_subject(self.subject)
        prompt = f"The following are multiple choice questions (with answers) about {subj}.\n\n"

        k = min(int(nshot), len(self._dev_examples))
        for ex in self._dev_examples[:k]:
            q, choices, ans = _coerce_example(ex)
            # Include the correct answer letter, followed by a blank line between examples.
            prompt += _format_question_block(q, choices) + f"{self.target_delimiter}{_CHOICE_LETTERS[ans]}\n\n"
        return prompt

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        q, choices, ans = _coerce_example(doc)
        text = _format_question_block(q, choices)
        # lm-eval-harness scores letter continuations for MMLU (with target_delimiter applied externally).
        options = [f"{_CHOICE_LETTERS[j]}" for j in range(4)]
        return text, options, int(ans)


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    nshot: int = 5,
    subjects: str = "all",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run MMLU evaluation (multiple-choice, zero/few-shot).
    """
    all_subjects = list(get_dataset_config_names("cais/mmlu"))
    # Some dataset versions expose an "all" config that concatenates subjects.
    # We treat the string "all" as "evaluate all subjects", so exclude this meta-config
    # to avoid wrong prompting ("about all") and double-counting.
    all_subjects_no_meta = [s for s in all_subjects if s != "all"]
    if subjects != "all":
        requested = [s.strip() for s in subjects.split(",") if s.strip()]
        # If users accidentally pass "all" in a list, expand it.
        if "all" in requested:
            requested = [s for s in requested if s != "all"] + all_subjects_no_meta
        unknown = [s for s in requested if s not in all_subjects]
        if unknown:
            raise ValueError(f"Unknown MMLU subjects: {unknown}. Use --subjects all or valid names.")
        eval_subjects = requested
    else:
        eval_subjects = all_subjects_no_meta

    overall_correct = 0
    overall_correct_norm = 0
    overall_total = 0
    per_subject: Dict[str, Dict[str, object]] = {}
    skipped: List[Dict[str, str]] = []

    for subject in eval_subjects:
        try:
            dev, test = _get_splits_for_subject(subject)
        except Exception as e:
            # Some dataset configs (e.g. "auxiliary_train") may not have a test split.
            # We skip them instead of failing the whole evaluation.
            skipped.append({"subject": subject, "reason": str(e)})
            continue

        # Deterministic: take the first nshot examples from the dev split.
        dev_examples = []
        if nshot > 0:
            dev_examples = [dev[i] for i in range(min(int(nshot), len(dev)))]

        task = _MMLUSubjectTask(subject=subject, dev_examples=dev_examples)
        counts, payload = evaluate_multiple_choice(
            task=task,
            docs=test,
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            limit=limit,
            nshot=int(nshot),
            desc=f"mmlu:{subject}",
            leave=False,
            collect_prompt_stats=True,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

        total = int(payload["total"])
        acc = float(payload["acc"])
        acc_norm = float(payload.get("acc_norm", acc))
        per_subject[subject] = {
            "accuracy": acc,
            "accuracy_norm": acc_norm,
            "acc": acc,
            "acc_norm": acc_norm,
            "total": total,
            "correct": int(counts.correct),
            "correct_norm": int(counts.correct_norm or 0),
        }
        overall_correct += int(counts.correct)
        overall_correct_norm += int(counts.correct_norm or 0)
        overall_total += total

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    overall_acc_norm = overall_correct_norm / overall_total if overall_total else 0.0
    evaluated_subjects = list(per_subject.keys())
    return {
        "task": "mmlu",
        "nshot": int(nshot),
        "requested_subjects": eval_subjects,
        "evaluated_subjects": evaluated_subjects,
        "overall": {
            "accuracy": overall_acc,
            "accuracy_norm": overall_acc_norm,
            "acc": overall_acc,
            "acc_norm": overall_acc_norm,
            "total": int(overall_total),
            "correct": int(overall_correct),
            "correct_norm": int(overall_correct_norm),
        },
        "per_subject": per_subject,
        "skipped_subjects": skipped,
    }

