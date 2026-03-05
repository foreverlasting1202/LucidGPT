"""
Evaluation utilities and benchmark runners.

This package is intentionally lightweight and "repo-local":
- Loads checkpoints produced by `utils.Logger.save_checkpoint()`
- Uses the same GPT-2 tokenizer as data preprocessing (`tiktoken.get_encoding("gpt2")`)
- Runs common evals (MMLU, HellaSwag, ARC, PIQA, Winogrande) and pretrain metrics
"""

