from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from tqdm import tqdm

from evals.scoring import score_text_options
from evals.tokenizer import encode


def _argmax(scores: Sequence[float]) -> int:
    if len(scores) == 0:
        raise ValueError("empty scores")
    best_i = 0
    best = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best:
            best = scores[i]
            best_i = i
    return best_i


@dataclass(frozen=True)
class MultipleChoiceCounts:
    correct: int
    total: int
    correct_norm: Optional[int] = None

    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def acc_norm(self) -> Optional[float]:
        if self.correct_norm is None:
            return None
        return self.correct_norm / self.total if self.total else 0.0


class MultipleChoiceTask:
    """
    Minimal `lm_eval`-style interface for multiple-choice tasks.

    Subclasses implement:
    - `doc_to_text(doc) -> str`
    - `doc_to_choices(doc) -> list[str]` (each choice is a continuation)
    - `doc_to_gold(doc) -> int` (index into choices)
    """

    task_name: str = "task"
    # Match lm-eval-harness behavior: choices are scored after `doc_to_text + target_delimiter`.
    # Most tasks want a single separating space.
    target_delimiter: str = " "
    # Some evaluation suites (e.g. FineWeb's lighteval tasks) prefix each choice with a
    # single leading space to ensure correct tokenization, but want acc_norm to ignore
    # that leading space ("acc_norm_nospace").
    acc_norm_strip_leading_space: bool = False

    def doc_to_text(self, doc: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def doc_to_choices(self, doc: dict) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def doc_to_gold(self, doc: dict) -> int:  # pragma: no cover
        raise NotImplementedError

    def fewshot_context(self, *, nshot: int) -> str:
        # This repo mostly uses task-specific few-shot formatting (e.g. MMLU),
        # so the default is "no few-shot".
        _ = nshot
        return ""

    def should_compute_acc_norm(self) -> bool:
        # `lm_eval` often reports both acc and acc_norm for variable-length choices.
        return True


def evaluate_multiple_choice(
    *,
    task: MultipleChoiceTask,
    docs: Sequence[dict],
    model,
    device,
    autocast_dtype,
    max_seq_len: int,
    limit: Optional[int] = None,
    nshot: int = 0,
    desc: Optional[str] = None,
    leave: bool = True,
    compute_acc_norm: Optional[bool] = None,
    collect_prompt_stats: bool = False,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
) -> Tuple[MultipleChoiceCounts, Dict[str, object]]:
    """
    Generic multiple-choice evaluator (similar in spirit to `lm_eval`).

    Returns:
    - counts (correct/total and optional correct_norm)
    - a small JSON-serializable payload with acc / acc_norm
    """
    import torch
    import torch.distributed as dist

    n_total = len(docs) if limit is None else min(int(limit), len(docs))
    do_norm = task.should_compute_acc_norm() if compute_acc_norm is None else bool(compute_acc_norm)

    fewshot_prefix = task.fewshot_context(nshot=nshot)

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
        it = range(rank, n_total, world_size)
    else:
        it = range(n_total)

    correct = 0
    correct_norm = 0 if do_norm else None
    local_total = 0
    prompt_tokens_sum = 0
    prompt_tokens_max = 0
    num_truncated = 0
    max_full_len = int(max_seq_len) + 1

    # Avoid noisy per-rank progress bars in distributed runs.
    if desc and not distributed:
        it = tqdm(it, desc=desc, leave=leave)

    for i in it:
        local_total += 1
        doc = docs[i]
        if hasattr(task, "doc_to_mc_example"):
            # Optional fast-path: parse doc once.
            prompt_suffix, choices, gold = task.doc_to_mc_example(doc)  # type: ignore[attr-defined]
            prompt = fewshot_prefix + str(prompt_suffix)
        else:
            prompt = fewshot_prefix + task.doc_to_text(doc)
            choices = task.doc_to_choices(doc)
            gold = int(task.doc_to_gold(doc))
        if not (0 <= gold < len(choices)):
            raise ValueError(f"gold index out of range: gold={gold} n_choices={len(choices)}")

        # lm-eval-harness inserts `target_delimiter` between context and each candidate.
        delim = getattr(task, "target_delimiter", " ")
        prompt_for_scoring = prompt + str(delim)

        prompt_len = 0
        if collect_prompt_stats:
            prompt_len = len(encode(prompt_for_scoring))
            prompt_tokens_sum += prompt_len
            if prompt_len > prompt_tokens_max:
                prompt_tokens_max = prompt_len

        scored = score_text_options(
            model,
            prompt=prompt_for_scoring,
            options=choices,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            length_normalize=False,  # compute acc_norm from token lengths ourselves
        )

        if collect_prompt_stats:
            max_cont = 0
            for s in scored:
                if int(s.num_tokens) > max_cont:
                    max_cont = int(s.num_tokens)
            if prompt_len + max_cont > max_full_len:
                num_truncated += 1

        pred = _argmax([s.loglikelihood for s in scored])
        correct += int(pred == gold)

        if do_norm:
            if bool(getattr(task, "acc_norm_strip_leading_space", False)):
                norm_scores = []
                for s in scored:
                    txt = str(s.text)
                    if txt.startswith(" "):
                        txt = txt[1:]
                    denom = max(1, len(encode(txt)))
                    norm_scores.append(s.loglikelihood / denom)
            else:
                norm_scores = [s.loglikelihood / max(1, int(s.num_tokens)) for s in scored]
            pred_norm = _argmax(norm_scores)
            assert correct_norm is not None
            correct_norm += int(pred_norm == gold)

    total = int(local_total)
    if distributed:
        dev = torch.device(device)
        backend = dist.get_backend()
        reduce_device = dev if backend == "nccl" else torch.device("cpu")

        total_correct_norm = int(correct_norm or 0)
        sum_tensor = torch.tensor(
            [int(correct), int(total_correct_norm), int(local_total), int(prompt_tokens_sum), int(num_truncated)],
            dtype=torch.long,
            device=reduce_device,
        )
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        correct = int(sum_tensor[0].item())
        total_correct_norm = int(sum_tensor[1].item())
        total = int(sum_tensor[2].item())
        prompt_tokens_sum = int(sum_tensor[3].item())
        num_truncated = int(sum_tensor[4].item())
        if do_norm:
            correct_norm = int(total_correct_norm)
        else:
            correct_norm = None

        max_tensor = torch.tensor([int(prompt_tokens_max)], dtype=torch.long, device=reduce_device)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        prompt_tokens_max = int(max_tensor[0].item())

    counts = MultipleChoiceCounts(correct=correct, total=total, correct_norm=correct_norm)
    payload: Dict[str, object] = {
        "acc": float(counts.acc()),
        "total": int(counts.total),
    }
    if do_norm:
        payload["acc_norm"] = float(counts.acc_norm() or 0.0)
    if collect_prompt_stats:
        denom = int(counts.total) if counts.total else 1
        payload.update(
            {
                "max_seq_len": int(max_seq_len),
                "prompt_tokens_avg": float(prompt_tokens_sum / denom),
                "prompt_tokens_max": int(prompt_tokens_max),
                "num_truncated": int(num_truncated),
                "truncated_frac": float(num_truncated / denom),
            }
        )
    return counts, payload

