from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _device_from_arg(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _autocast_dtype_from_arg(dtype: str, device) -> Optional["torch.dtype"]:
    """
    Returns:
    - torch.bfloat16 / torch.float16 if autocast should be enabled
    - None if we should run in FP32 (no autocast)
    """
    import torch

    if dtype == "fp32":
        return None
    if device.type != "cuda":
        # Keep it simple/portable: only autocast for CUDA.
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "auto":
        return torch.bfloat16
    raise ValueError(f"Unknown --dtype: {dtype}")


def _resolve_model_config(ckpt: Dict[str, Any], args):
    from evals.checkpoint import ModelConfig, get_model_config

    mc = get_model_config(ckpt)
    if mc is not None:
        return mc

    missing = [k for k in ("vocab_size", "n_layer", "n_head", "n_embd") if getattr(args, k) is None]
    if missing:
        raise ValueError(
            "Checkpoint does not contain model_config. Provide: "
            + ", ".join(f"--{k}" for k in missing)
        )
    return ModelConfig(
        vocab_size=int(args.vocab_size),
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_embd=int(args.n_embd),
    )


def _default_max_seq_len(ckpt: Dict[str, Any], fallback: int = 1024) -> int:
    tc = ckpt.get("training_config") or ckpt.get("config")
    if isinstance(tc, dict) and "sequence_length" in tc:
        try:
            return int(tc["sequence_length"])
        except Exception:
            pass
    return int(fallback)


def _load_model(args):
    import torch
    from evals.checkpoint import get_model_state_dict, load_checkpoint
    from models.nanoGPT import GPT, GPTConfig

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    mc = _resolve_model_config(ckpt, args)

    device = _device_from_arg(args.device)
    autocast_dtype = _autocast_dtype_from_arg(args.dtype, device)
    max_seq_len = int(args.max_seq_len or _default_max_seq_len(ckpt))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig(vocab_size=mc.vocab_size, n_layer=mc.n_layer, n_head=mc.n_head, n_embd=mc.n_embd))
    state_dict = get_model_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=not args.non_strict)
    model.to(device)
    model.eval()
    return model, ckpt, device, autocast_dtype, max_seq_len


def _write_output(payload: Dict[str, Any], out_path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo-local evaluation runner")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a training checkpoint .pt")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps|cuda:0 ...")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Override model context length for eval truncation (default: from checkpoint training config, else 1024)",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--non_strict", action="store_true", help="Allow missing/unexpected keys when loading weights")

    # Only needed for older checkpoints that don't store model_config.
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_pre = subparsers.add_parser("pretrain", help="Compute pretrain-style validation metrics on .bin shards")
    p_pre.add_argument("--input_bin", type=str, required=True, help="Glob pattern for validation .bin shards")
    p_pre.add_argument("--batch_size", type=int, default=4, help="Batch size in sequences")
    p_pre.add_argument("--sequence_length", type=int, default=None, help="Sequence length (default: from checkpoint)")
    p_pre.add_argument("--eval_tokens", type=int, default=1024 * 1024, help="How many tokens to evaluate")

    p_mmlu = subparsers.add_parser("mmlu", help="Run MMLU zero/few-shot evaluation")
    p_mmlu.add_argument("--nshot", type=int, default=5)
    p_mmlu.add_argument("--subjects", type=str, default="all", help="'all' or comma-separated subject names")
    p_mmlu.add_argument("--limit", type=int, default=None, help="Limit questions per subject (debug)")

    p_hs = subparsers.add_parser("hellaswag", help="Run HellaSwag validation/test evaluation")
    p_hs.add_argument("--split", type=str, default="validation")
    p_hs.add_argument("--limit", type=int, default=None)

    p_arc = subparsers.add_parser("arc", help="Run ARC-Easy / ARC-Challenge evaluation")
    p_arc.add_argument("--challenge", action="store_true", help="If set, use ARC-Challenge (default: ARC-Easy)")
    p_arc.add_argument("--split", type=str, default="test")
    p_arc.add_argument("--limit", type=int, default=None)

    p_piqa = subparsers.add_parser("piqa", help="Run PIQA validation evaluation")
    p_piqa.add_argument("--split", type=str, default="validation")
    p_piqa.add_argument("--limit", type=int, default=None)

    p_wg = subparsers.add_parser("winogrande", help="Run Winogrande validation evaluation")
    p_wg.add_argument("--config", type=str, default="winogrande_xl")
    p_wg.add_argument("--split", type=str, default="validation")
    p_wg.add_argument("--limit", type=int, default=None)

    p_obqa = subparsers.add_parser("openbookqa", help="Run OpenBookQA evaluation")
    p_obqa.add_argument("--split", type=str, default="test")
    p_obqa.add_argument("--limit", type=int, default=None)

    p_csqa = subparsers.add_parser("commonsense_qa", help="Run CommonsenseQA evaluation")
    p_csqa.add_argument("--split", type=str, default="validation")
    p_csqa.add_argument("--limit", type=int, default=None)

    p_siqa = subparsers.add_parser("siqa", help="Run Social IQa (siqa) evaluation")
    p_siqa.add_argument("--split", type=str, default="validation")
    p_siqa.add_argument("--limit", type=int, default=None)

    p_fw = subparsers.add_parser("fineweb", help="Run FineWeb-v1-style eval suite (agg_score)")
    # FineWeb blog ran lighteval with --max_samples 1000.
    p_fw.add_argument("--limit", type=int, default=1000, help="Per-task example limit (default: 1000)")
    p_fw.add_argument("--mmlu_subjects", type=str, default="all")

    p_bench = subparsers.add_parser("bench", help="Run a small benchmark suite")
    p_bench.add_argument(
        "--tasks",
        type=str,
        default="mmlu,hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,siqa,commonsense_qa",
        help=(
            "Comma-separated: "
            "mmlu,hellaswag,arc_easy,arc_challenge,arc,piqa,winogrande,openbookqa,siqa,commonsense_qa"
        ),
    )
    p_bench.add_argument("--limit", type=int, default=None, help="Per-task example limit (debug)")
    p_bench.add_argument("--mmlu_nshot", type=int, default=5)
    p_bench.add_argument("--mmlu_subjects", type=str, default="all")
    p_bench.add_argument("--winogrande_config", type=str, default="winogrande_xl")

    args = parser.parse_args()
    model, ckpt, device, autocast_dtype, max_seq_len = _load_model(args)

    if args.cmd == "pretrain":
        from evals.pretrain import run as run_pretrain

        seq_len = int(args.sequence_length or _default_max_seq_len(ckpt))
        payload = run_pretrain(
            model=model,
            input_bin=args.input_bin,
            batch_size=int(args.batch_size),
            sequence_length=seq_len,
            device=device,
            autocast_dtype=autocast_dtype,
            eval_tokens=int(args.eval_tokens),
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "mmlu":
        from evals.tasks.mmlu import run as run_mmlu

        payload = run_mmlu(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            nshot=int(args.nshot),
            subjects=str(args.subjects),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "hellaswag":
        from evals.tasks.hellaswag import run as run_hellaswag

        payload = run_hellaswag(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "arc":
        from evals.tasks.arc import run as run_arc

        payload = run_arc(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            challenge=bool(args.challenge),
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "piqa":
        from evals.tasks.piqa import run as run_piqa

        payload = run_piqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "winogrande":
        from evals.tasks.winogrande import run as run_winogrande

        payload = run_winogrande(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            config=str(args.config),
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "openbookqa":
        from evals.tasks.openbookqa import run as run_openbookqa

        payload = run_openbookqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "commonsense_qa":
        from evals.tasks.commonsense_qa import run as run_commonsense_qa

        payload = run_commonsense_qa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "siqa":
        from evals.tasks.siqa import run as run_siqa

        payload = run_siqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split=str(args.split),
            limit=args.limit,
        )
        _write_output(payload, args.out)
        return

    if args.cmd == "fineweb":
        from evals.tasks.arc import run as run_arc
        from evals.tasks.commonsense_qa import run as run_commonsense_qa
        from evals.tasks.hellaswag import run as run_hellaswag
        from evals.tasks.mmlu_fineweb import run as run_mmlu_fineweb
        from evals.tasks.openbookqa import run as run_openbookqa
        from evals.tasks.piqa import run as run_piqa
        from evals.tasks.siqa import run as run_siqa
        from evals.tasks.winogrande import run as run_winogrande

        results = []
        metrics: Dict[str, float] = {}

        res_csqa = run_commonsense_qa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split="validation",
            limit=args.limit,
        )
        results.append(res_csqa)
        metrics["commonsense_qa/acc_norm"] = float(res_csqa["acc_norm"])

        res_hs = run_hellaswag(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split="validation",
            limit=args.limit,
        )
        results.append(res_hs)
        metrics["hellaswag/acc_norm"] = float(res_hs["acc_norm"])

        res_obqa = run_openbookqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split="test",
            limit=args.limit,
        )
        results.append(res_obqa)
        metrics["openbookqa/acc_norm"] = float(res_obqa["acc_norm"])

        res_piqa = run_piqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split="validation",
            limit=args.limit,
        )
        results.append(res_piqa)
        metrics["piqa/acc_norm"] = float(res_piqa["acc_norm"])

        res_siqa = run_siqa(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            split="validation",
            limit=args.limit,
        )
        results.append(res_siqa)
        metrics["siqa/acc_norm"] = float(res_siqa["acc_norm"])

        res_wg = run_winogrande(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            config="winogrande_xl",
            split="validation",
            limit=args.limit,
        )
        results.append(res_wg)
        metrics["winogrande/acc_norm"] = float(res_wg["acc_norm"])

        res_arc = run_arc(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            challenge=True,
            split="test",
            limit=args.limit,
        )
        results.append(res_arc)
        metrics["arc/acc_norm"] = float(res_arc["acc_norm"])

        res_mmlu = run_mmlu_fineweb(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            subjects=str(args.mmlu_subjects),
            limit=args.limit,
        )
        results.append(res_mmlu)
        metrics["mmlu/acc_norm"] = float(res_mmlu["overall"]["acc_norm"])

        required = [
            "commonsense_qa/acc_norm",
            "hellaswag/acc_norm",
            "openbookqa/acc_norm",
            "piqa/acc_norm",
            "siqa/acc_norm",
            "winogrande/acc_norm",
            "arc/acc_norm",
            "mmlu/acc_norm",
        ]
        agg_score = sum(metrics[k] for k in required) / len(required)

        payload = {
            "task": "fineweb",
            "agg_score": float(agg_score),
            "metrics": metrics,
            "required_metrics": required,
            "results": results,
        }
        _write_output(payload, args.out)
        return

    if args.cmd == "bench":
        from evals.tasks.arc import run as run_arc
        from evals.tasks.commonsense_qa import run as run_commonsense_qa
        from evals.tasks.hellaswag import run as run_hellaswag
        from evals.tasks.mmlu import run as run_mmlu
        from evals.tasks.openbookqa import run as run_openbookqa
        from evals.tasks.piqa import run as run_piqa
        from evals.tasks.siqa import run as run_siqa
        from evals.tasks.winogrande import run as run_winogrande

        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        results = []
        for t in tasks:
            if t == "mmlu":
                results.append(
                    run_mmlu(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        nshot=int(args.mmlu_nshot),
                        subjects=str(args.mmlu_subjects),
                        limit=args.limit,
                    )
                )
            elif t == "hellaswag":
                results.append(
                    run_hellaswag(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=args.limit,
                    )
                )
            elif t == "arc_easy":
                results.append(
                    run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=False,
                        split="test",
                        limit=args.limit,
                    )
                )
            elif t == "arc":
                results.append(
                    run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=True,
                        split="test",
                        limit=args.limit,
                    )
                )
            elif t == "arc_challenge":
                results.append(
                    run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=True,
                        split="test",
                        limit=args.limit,
                    )
                )
            elif t == "piqa":
                results.append(
                    run_piqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=args.limit,
                    )
                )
            elif t == "openbookqa":
                results.append(
                    run_openbookqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="test",
                        limit=args.limit,
                    )
                )
            elif t == "siqa":
                results.append(
                    run_siqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=args.limit,
                    )
                )
            elif t == "commonsense_qa":
                results.append(
                    run_commonsense_qa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=args.limit,
                    )
                )
            elif t == "winogrande":
                results.append(
                    run_winogrande(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        config=str(args.winogrande_config),
                        split="validation",
                        limit=args.limit,
                    )
                )
            else:
                raise ValueError(f"Unknown task in --tasks: {t}")

        payload = {"task": "bench", "tasks": tasks, "results": results}
        _write_output(payload, args.out)
        return

    raise RuntimeError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    main()

