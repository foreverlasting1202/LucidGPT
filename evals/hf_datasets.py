from __future__ import annotations

from typing import Any, Dict, Tuple

from datasets import get_dataset_config_names as _get_dataset_config_names
from datasets import load_dataset as _load_dataset

# Some environments disable dataset loading scripts entirely (and even remove
# `trust_remote_code`). To keep eval runnable, we route well-known benchmarks to
# hub repos used by common evaluation harnesses.
#
# Each entry maps an incoming `load_dataset(path, ...)` to:
# - canonical repo id on the hub
# - optional default revision
_DATASET_ALIASES: Dict[str, Tuple[str, str | None]] = {
    # PIQA: the canonical `piqa` dataset is script-based in many environments.
    # Use `baber/piqa` (also used by lm-eval-harness) which is hub-hosted and does
    # not require dataset loading scripts / remote code.
    "piqa": ("baber/piqa", None),
    # HellaSwag: lm-eval-harness uses Rowan/hellaswag.
    "hellaswag": ("Rowan/hellaswag", None),
    # ARC: keep config names (ARC-Easy / ARC-Challenge), so avoid parquet revision.
    "ai2_arc": ("allenai/ai2_arc", None),
    # Winogrande: keep config names (winogrande_xl, ...), so avoid parquet revision.
    "winogrande": ("allenai/winogrande", None),
    # FineWeb/FineWeb-Edu evaluation suite datasets.
    "openbookqa": ("allenai/openbookqa", None),
    "commonsense_qa": ("tau/commonsense_qa", None),
    "social_i_qa": ("allenai/social_i_qa", None),
}


def _rewrite_path_and_kwargs(args: tuple[Any, ...], kwargs: dict) -> tuple[tuple[Any, ...], dict]:
    if not args:
        return args, kwargs
    path = args[0]
    if not isinstance(path, str):
        return args, kwargs

    alias = _DATASET_ALIASES.get(path)
    if alias is None:
        return args, kwargs

    new_path, default_revision = alias
    new_args = (new_path,) + args[1:]
    new_kwargs = dict(kwargs)
    if default_revision is not None and "revision" not in new_kwargs:
        new_kwargs["revision"] = default_revision
    return new_args, new_kwargs


def load_dataset(*args: Any, **kwargs: Any):
    """
    Wrapper around `datasets.load_dataset`.

    - Rewrites some dataset names to parquet-safe sources (see `_DATASET_ALIASES`).
    - Avoids `trust_remote_code` entirely (some environments don't support it anymore).
    """
    args2, kwargs2 = _rewrite_path_and_kwargs(args, kwargs)
    return _load_dataset(*args2, **kwargs2)


def get_dataset_config_names(*args: Any, **kwargs: Any):
    """
    Wrapper around `datasets.get_dataset_config_names` with the same alias rewriting.
    """
    args2, kwargs2 = _rewrite_path_and_kwargs(args, kwargs)
    return _get_dataset_config_names(*args2, **kwargs2)

