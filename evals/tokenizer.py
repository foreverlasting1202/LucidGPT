from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import tiktoken


@lru_cache(maxsize=1)
def get_gpt2_encoding():
    """
    Return the repo-standard tokenizer used in `data/*.py` preprocessing scripts.

    Notes:
    - We intentionally use `encode_ordinary` (no special token handling) for prompts.
    - The EOT token id can be accessed via `eot_token_id()`.
    """
    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> List[int]:
    return get_gpt2_encoding().encode_ordinary(text)


def decode(ids: Sequence[int]) -> str:
    return get_gpt2_encoding().decode(list(ids))


def eot_token_id() -> int:
    enc = get_gpt2_encoding()
    return int(enc._special_tokens["<|endoftext|>"])

