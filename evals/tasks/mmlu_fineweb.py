from __future__ import annotations

from typing import Dict, List, Optional

from evals.hf_datasets import get_dataset_config_names, load_dataset
from evals.tasks.base import MultipleChoiceTask, evaluate_multiple_choice


def _format_topic(topic: str) -> str:
    return str(topic).replace("_", " ")


class _MMLUFinewebSubjectTask(MultipleChoiceTask):
    """
    FineWeb blog's lighteval MMLU variant:
    - 0-shot
    - score FULL answer texts (not letter targets)

    Reference:
    `https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py`
    """

    task_name = "mmlu"
    # Choices already include a leading space.
    target_delimiter = ""
    acc_norm_strip_leading_space = True

    def __init__(self, *, subject: str):
        self.subject = str(subject)

    def doc_to_mc_example(self, doc: dict) -> tuple[str, list[str], int]:
        topic = str(doc.get("subject") or self.subject)
        question = str(doc["question"])
        choices = list(doc["choices"])
        if len(choices) != 4:
            raise ValueError(f"Expected 4 choices, got {len(choices)}")
        gold = int(doc["answer"])
        if not (0 <= gold < 4):
            raise ValueError(f"Bad answer index: {gold}")

        prompt = f"The following are questions about {_format_topic(topic)}.\nQuestion: {question}\nAnswer:"
        options = [f" {str(c).strip()}" for c in choices]
        return prompt, options, gold


def run(
    *,
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    subjects: str = "all",
    limit: Optional[int] = None,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run the FineWeb blog's lighteval-style MMLU evaluation.

    This mirrors their intention for small-base models:
    use the full answer text as the continuation target (not A/B/C/D).
    """
    import torch
    import torch.distributed as dist

    all_subjects = list(get_dataset_config_names("lighteval/mmlu"))
    all_subjects_no_meta = [s for s in all_subjects if s != "all"]

    if subjects != "all":
        requested = [s.strip() for s in subjects.split(",") if s.strip()]
        if "all" in requested:
            requested = [s for s in requested if s != "all"] + all_subjects_no_meta
        unknown = [s for s in requested if s not in all_subjects]
        if unknown:
            raise ValueError(f"Unknown MMLU subjects: {unknown}. Use --subjects all or valid names.")
        eval_subjects = requested
    else:
        eval_subjects = all_subjects_no_meta

    distributed = (
        ddp_rank is not None
        and ddp_world_size is not None
        and int(ddp_world_size) > 1
        and dist.is_available()
        and dist.is_initialized()
    )

    if distributed:
        # Shard subjects across ranks to speed up evaluation (FineWeb used multi-GPU eval via lighteval).
        my_subjects = [s for i, s in enumerate(eval_subjects) if (i % int(ddp_world_size)) == int(ddp_rank)]
    else:
        my_subjects = list(eval_subjects)

    overall_correct = 0
    overall_correct_norm = 0
    overall_total = 0
    per_subject: Dict[str, Dict[str, object]] = {}
    skipped: List[Dict[str, str]] = []

    for subject in my_subjects:
        try:
            ds = load_dataset("lighteval/mmlu", subject, split="test")
            task = _MMLUFinewebSubjectTask(subject=str(subject))
            counts, payload = evaluate_multiple_choice(
                task=task,
                docs=ds,
                model=model,
                device=device,
                autocast_dtype=autocast_dtype,
                max_seq_len=max_seq_len,
                limit=limit,
                nshot=0,
                # Avoid noisy progress bars in distributed mode.
                desc=None if distributed else f"mmlu:{subject}",
                leave=False,
            )

            total = int(payload["total"])
            acc = float(payload["acc"])
            acc_norm = float(payload.get("acc_norm", acc))
            if not distributed:
                per_subject[str(subject)] = {
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
        except Exception as e:
            skipped.append({"subject": str(subject), "reason": str(e)})
            # Keep going; we'll still all-reduce totals so training can continue.
            continue

    if distributed:
        # Reduce totals across ranks (micro-average across all questions).
        t = torch.tensor(
            [overall_correct, overall_correct_norm, overall_total],
            dtype=torch.long,
            device=device if getattr(device, "type", None) == "cuda" else torch.device("cpu"),
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        overall_correct, overall_correct_norm, overall_total = (int(x) for x in t.tolist())

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    overall_acc_norm = overall_correct_norm / overall_total if overall_total else 0.0

    return {
        "task": "mmlu",
        "variant": "fineweb_lighteval_answer_text",
        "distributed": bool(distributed),
        "world_size": int(ddp_world_size) if ddp_world_size is not None else 1,
        "requested_subjects": eval_subjects,
        "evaluated_subjects": list(per_subject.keys()) if not distributed else None,
        "overall": {
            "accuracy": overall_acc,
            "accuracy_norm": overall_acc_norm,
            "acc": overall_acc,
            "acc_norm": overall_acc_norm,
            "total": int(overall_total),
            "correct": int(overall_correct),
            "correct_norm": int(overall_correct_norm),
        },
        "per_subject": per_subject if not distributed else None,
        "skipped_subjects": skipped,
    }

