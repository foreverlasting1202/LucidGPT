"""
FineWeb-EDU dataset preprocessing with score-prioritized ordering.

Goal:
- Convert raw FineWeb-EDU (local parquet/jsonl) into GPT-2 token .bin shards
  compatible with this repo's `data_loader.py`.
- Ensure that *higher score samples appear earlier* in the training stream
  (descending by score), even if the training loader does no shuffle.

Why not full sort?
- FineWeb-EDU is enormous (~97M docs). Sorting all rows is infeasible.
- But score is bounded (<= 5), so we can do an external bucket-sort:
  1) One pass: tokenize + append tokens to score buckets on disk (spool files)
  2) Second pass: read buckets from high->low and pack into fixed-size shards

This guarantees that the first tokens seen by the loader come from the highest
score region, while keeping memory usage small.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256


def write_datafile(filename: str, toks: np.ndarray) -> None:
    """
    Saves token data as a .bin file, for reading in `data_loader.py`.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(len(toks))
    assert isinstance(toks, np.ndarray) and toks.dtype == np.uint16
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def _score_bins_from_width(max_score: float, bin_width: float) -> int:
    # inclusive of max_score: e.g., max=5.0, width=0.1 -> 51 bins (0..50)
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")
    return int(math.floor(max_score / bin_width) + 1)


@dataclass(frozen=True)
class ScoreBucketer:
    max_score: float
    bin_width: float
    num_bins: int  # derived from max_score/bin_width, inclusive

    @classmethod
    def from_width(cls, *, max_score: float, bin_width: float) -> "ScoreBucketer":
        num_bins = _score_bins_from_width(max_score, bin_width)
        return cls(max_score=max_score, bin_width=bin_width, num_bins=num_bins)

    def bucket_id(self, score: float) -> int:
        """
        Map a score to a bucket id in [0, num_bins-1] where:
        - bucket 0 is the highest-score bucket (closest to max_score)
        - larger bucket ids correspond to lower scores
        """
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if not math.isfinite(s):
            s = 0.0
        if s < 0.0:
            s = 0.0
        if s > self.max_score:
            s = self.max_score
        q = int(math.floor(s / self.bin_width + 1e-12))
        if q < 0:
            q = 0
        elif q >= self.num_bins:
            q = self.num_bins - 1
        # invert so that bucket 0 is highest score
        return (self.num_bins - 1) - q

    def bucket_score_range(self, bucket_id: int) -> Tuple[float, float]:
        """Return [lo, hi] (inclusive hi for top bucket only) for a bucket."""
        if not (0 <= bucket_id < self.num_bins):
            raise ValueError("bucket_id out of range")
        q = (self.num_bins - 1) - bucket_id
        lo = q * self.bin_width
        hi = min(self.max_score, (q + 1) * self.bin_width)
        return lo, hi


class BucketSpoolWriter:
    """
    Append-only spool writer for score buckets.

    Each bucket file is raw uint16 tokens (no header). We do a second pass later
    to pack these tokens into fixed-size .bin shards.
    """

    def __init__(self, spool_dir: str, *, max_open_files: int = 64, buffer_mb: int = 8):
        self.spool_dir = spool_dir
        os.makedirs(self.spool_dir, exist_ok=True)
        self.max_open_files = max(1, int(max_open_files))
        self._buffer_bytes = max(1, int(buffer_mb)) * 1024 * 1024
        self._handles: "OrderedDict[str, object]" = OrderedDict()
        self.bucket_token_counts: Dict[int, int] = {}
        self.total_tokens: int = 0
        self.total_docs: int = 0

    def _bucket_path(self, bucket_id: int) -> str:
        return os.path.join(self.spool_dir, f"bucket_{bucket_id:05d}.u16")

    def _get_handle(self, bucket_id: int):
        path = self._bucket_path(bucket_id)
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        fh = open(path, "ab", buffering=self._buffer_bytes)
        self._handles[path] = fh
        self._handles.move_to_end(path)
        if len(self._handles) > self.max_open_files:
            _, old_fh = self._handles.popitem(last=False)
            old_fh.close()
        return fh

    def write(self, bucket_id: int, toks_uint16: np.ndarray) -> None:
        fh = self._get_handle(bucket_id)
        fh.write(toks_uint16.tobytes())
        n = int(len(toks_uint16))
        self.bucket_token_counts[bucket_id] = self.bucket_token_counts.get(bucket_id, 0) + n
        self.total_tokens += n
        self.total_docs += 1

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


class BufferedBucketSpoolWriter:
    """
    Faster spool writer:
    - Buffers tokens in memory per bucket (array('H'))
    - Flushes large blocks to disk (much fewer writes)
    - Opens files only on flush (avoids FD thrash when buckets are many)
    """

    def __init__(
        self,
        spool_dir: str,
        *,
        num_buckets: int,
        eot_token: int,
        total_buffer_mb: int = 256,
        min_flush_tokens: int = 8192,
    ):
        self.spool_dir = spool_dir
        os.makedirs(self.spool_dir, exist_ok=True)
        self.num_buckets = int(num_buckets)
        self.eot_token = int(eot_token)

        total_buffer_mb = max(1, int(total_buffer_mb))
        # target per-bucket threshold so worst-case memory is bounded
        total_tokens_budget = (total_buffer_mb * 1024 * 1024) // 2
        per_bucket = max(int(min_flush_tokens), int(total_tokens_budget // max(1, self.num_buckets)))
        self.flush_tokens = per_bucket

        self._buffers: List[array] = [array("H") for _ in range(self.num_buckets)]
        self.bucket_token_counts: List[int] = [0 for _ in range(self.num_buckets)]
        self.total_tokens: int = 0
        self.total_docs: int = 0

    def _bucket_path(self, bucket_id: int) -> str:
        return os.path.join(self.spool_dir, f"bucket_{bucket_id:05d}.u16")

    def write(self, bucket_id: int, tokens: List[int]) -> None:
        buf = self._buffers[bucket_id]
        buf.append(self.eot_token)
        buf.extend(tokens)
        n = 1 + len(tokens)
        self.bucket_token_counts[bucket_id] += n
        self.total_tokens += n
        self.total_docs += 1
        if len(buf) >= self.flush_tokens:
            self.flush(bucket_id)

    def flush(self, bucket_id: int) -> None:
        buf = self._buffers[bucket_id]
        if not buf:
            return
        path = self._bucket_path(bucket_id)
        # open only on flush to avoid "too many open files"
        with open(path, "ab") as f:
            f.write(buf.tobytes())
        del buf[:]

    def close(self) -> None:
        for bucket_id in range(self.num_buckets):
            self.flush(bucket_id)


def _iter_data_files(data_dir: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, pattern), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def _load_local_dataset(*, data_dir: str, data_format: str, data_glob: str):
    files = _iter_data_files(data_dir, data_glob)
    if len(files) == 0:
        raise FileNotFoundError(
            f"did not find any files under {data_dir} matching {data_glob!r}"
        )
    # streaming=True keeps memory bounded for massive datasets
    builder = "json" if data_format in ("json", "jsonl") else data_format
    return load_dataset(builder, data_files=files, split="train", streaming=True)


def _batched(iterable: Iterable[dict], batch_size: int) -> Iterator[List[dict]]:
    batch: List[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_ordinary_batch(enc, texts: List[str], num_threads: int) -> List[List[int]]:
    """
    Prefer the fastest available batch encoder in tiktoken.
    Tries to pass num_threads if supported; otherwise falls back gracefully.
    """
    fn = getattr(enc, "encode_ordinary_batch", None)
    if fn is not None:
        try:
            return fn(texts, num_threads=num_threads)
        except TypeError:
            return fn(texts)
    fn = getattr(enc, "encode_batch", None)
    if fn is not None:
        try:
            return fn(texts, num_threads=num_threads)
        except TypeError:
            return fn(texts)
    # slow fallback
    return [enc.encode_ordinary(t) for t in texts]


def _tokenize_docs_threaded(
    docs: Iterable[dict],
    *,
    text_key: str,
    score_key: str,
    bucketer: ScoreBucketer,
    num_workers: int,
    batch_docs: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Tokenize docs using a thread pool. This avoids IPC/pickling overhead.

    Note: This only helps if the tokenizer releases the GIL (tiktoken usually does).
    """
    from concurrent.futures import ThreadPoolExecutor

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def _proc(doc: dict) -> Tuple[int, np.ndarray]:
        score = doc.get(score_key, 0.0)
        bucket = bucketer.bucket_id(score)
        text = doc.get(text_key, "")
        if not isinstance(text, str):
            text = str(text)
        toks = [eot]
        toks.extend(enc.encode_ordinary(text))
        toks_np = np.asarray(toks, dtype=np.uint16)
        return bucket, toks_np

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for batch in _batched(docs, batch_docs):
            for bucket, toks_np in ex.map(_proc, batch):
                yield bucket, toks_np


def _tokenize_docs_processes(
    docs: Iterable[dict],
    *,
    text_key: str,
    score_key: str,
    bucketer: ScoreBucketer,
    num_workers: int,
    chunksize: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Tokenize docs using process pool (slower IPC, but robust if GIL is held).
    """
    import multiprocessing as mp

    with mp.Pool(
        processes=num_workers,
        initializer=_mp_init_worker,
        initargs=(text_key, score_key, bucketer),
    ) as pool:
        for bucket, toks_np in pool.imap_unordered(_mp_tokenize_and_bucket, docs, chunksize=chunksize):
            yield bucket, toks_np


# ---------------------------
# multiprocessing worker funcs

_MP_ENC = None
_MP_EOT = None
_MP_TEXT_KEY = None
_MP_SCORE_KEY = None
_MP_BUCKETER: Optional[ScoreBucketer] = None


def _mp_init_worker(text_key: str, score_key: str, bucketer: ScoreBucketer) -> None:
    global _MP_ENC, _MP_EOT, _MP_TEXT_KEY, _MP_SCORE_KEY, _MP_BUCKETER
    enc = tiktoken.get_encoding("gpt2")
    _MP_ENC = enc
    _MP_EOT = enc._special_tokens["<|endoftext|>"]
    _MP_TEXT_KEY = text_key
    _MP_SCORE_KEY = score_key
    _MP_BUCKETER = bucketer


def _mp_tokenize_and_bucket(doc: dict) -> Tuple[int, np.ndarray]:
    assert _MP_ENC is not None and _MP_EOT is not None and _MP_BUCKETER is not None
    score = doc.get(_MP_SCORE_KEY, 0.0)
    bucket = _MP_BUCKETER.bucket_id(score)
    text = doc.get(_MP_TEXT_KEY, "")
    if not isinstance(text, str):
        text = str(text)
    toks = [_MP_EOT]
    toks.extend(_MP_ENC.encode_ordinary(text))
    toks_np = np.asarray(toks, dtype=np.uint16)
    return bucket, toks_np


def _spool_stage(
    *,
    dataset,
    spool_dir: str,
    text_key: str,
    score_key: str,
    bucketer: ScoreBucketer,
    num_workers: int,
    worker_type: str,
    batch_docs: int,
    chunksize: int,
    max_open_spool_files: int,
    spool_buffer_mb: int,
) -> Dict[str, object]:
    pbar = tqdm(desc="spooling (docs)", unit="docs")

    # Fast path: single-process, batched encoder + buffered writer
    if worker_type == "batched":
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]
        writer = BufferedBucketSpoolWriter(
            spool_dir,
            num_buckets=bucketer.num_bins,
            eot_token=eot,
            total_buffer_mb=spool_buffer_mb,
        )
        for batch in _batched(dataset, batch_docs):
            texts: List[str] = []
            bucket_ids: List[int] = []
            for doc in batch:
                score = doc.get(score_key, 0.0)
                bucket_ids.append(bucketer.bucket_id(score))
                text = doc.get(text_key, "")
                texts.append(text if isinstance(text, str) else str(text))
            token_lists = _encode_ordinary_batch(enc, texts, num_threads=num_workers)
            for b, toks in zip(bucket_ids, token_lists):
                writer.write(b, toks)
            pbar.update(len(batch))
        pbar.close()
        writer.close()
        bucket_token_counts = {str(i): int(c) for i, c in enumerate(writer.bucket_token_counts) if c}
        total_docs = writer.total_docs
        total_tokens = writer.total_tokens
    else:
        # Compatibility path (older behavior). This is slower, but still works.
        writer_compat = BucketSpoolWriter(spool_dir, max_open_files=max_open_spool_files)
        if worker_type == "threads":
            it = _tokenize_docs_threaded(
                dataset,
                text_key=text_key,
                score_key=score_key,
                bucketer=bucketer,
                num_workers=num_workers,
                batch_docs=batch_docs,
            )
        elif worker_type == "processes":
            it = _tokenize_docs_processes(
                dataset,
                text_key=text_key,
                score_key=score_key,
                bucketer=bucketer,
                num_workers=num_workers,
                chunksize=chunksize,
            )
        else:
            raise ValueError("worker_type must be one of: batched, threads, processes")
        for bucket_id, toks_np in it:
            writer_compat.write(bucket_id, toks_np)
            pbar.update(1)
        pbar.close()
        writer_compat.close()
        bucket_token_counts = {str(k): int(v) for k, v in sorted(writer_compat.bucket_token_counts.items())}
        total_docs = writer_compat.total_docs
        total_tokens = writer_compat.total_tokens

    meta = {
        "magic": MAGIC,
        "version": VERSION,
        "tokenizer": "tiktoken:gpt2",
        "text_key": text_key,
        "score_key": score_key,
        "max_score": bucketer.max_score,
        "bin_width": bucketer.bin_width,
        "num_bins": bucketer.num_bins,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "bucket_token_counts": bucket_token_counts,
    }
    with open(os.path.join(spool_dir, "spool_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def _finalize_stage(
    *,
    spool_dir: str,
    out_dir: str,
    shard_size: int,
    delete_spool: bool,
    read_chunk_tokens: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Discover bucket spool files (already encoded so lexicographic sort == score desc)
    bucket_files = sorted(glob.glob(os.path.join(spool_dir, "bucket_*.u16")))
    if len(bucket_files) == 0:
        raise FileNotFoundError(f"no bucket spool files found under {spool_dir}")

    def _tokens_in_file(path: str) -> int:
        sz = os.path.getsize(path)
        if sz % 2 != 0:
            raise ValueError(f"spool file size not divisible by 2 bytes: {path}")
        return sz // 2

    total_tokens = sum(_tokens_in_file(p) for p in bucket_files)
    pbar = tqdm(total=total_tokens, desc="finalizing (tokens)", unit="tokens")

    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0

    def _write_shard(toks: np.ndarray) -> None:
        nonlocal shard_index
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(out_dir, f"finewebedu_{split}_{shard_index:06d}.bin")
        write_datafile(filename, toks)
        shard_index += 1

    for path in bucket_files:
        with open(path, "rb") as f:
            while True:
                chunk = np.fromfile(f, dtype=np.uint16, count=read_chunk_tokens)
                if chunk.size == 0:
                    break
                i = 0
                while i < chunk.size:
                    space = shard_size - token_count
                    take = min(space, chunk.size - i)
                    all_tokens_np[token_count : token_count + take] = chunk[i : i + take]
                    token_count += take
                    i += take
                    pbar.update(take)
                    if token_count == shard_size:
                        _write_shard(all_tokens_np)
                        token_count = 0

        if delete_spool:
            os.remove(path)

    # write last partial shard
    if token_count != 0:
        _write_shard(all_tokens_np[:token_count])

    pbar.close()

    if delete_spool:
        # keep metadata for provenance
        for fname in ["spool_meta.json"]:
            src = os.path.join(spool_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, fname))


def main() -> None:
    parser = argparse.ArgumentParser(description="FineWeb-EDU preprocessing (score-prioritized)")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "fineweb-edu-100B"),
        help="Local directory containing FineWeb-EDU parquet/jsonl files",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="parquet",
        choices=["parquet", "json", "jsonl"],
        help="Local file format for FineWeb-EDU data",
    )
    parser.add_argument(
        "--data_glob",
        type=str,
        default="**/*.parquet",
        help="Recursive glob under --data_dir to find data files",
    )
    parser.add_argument("--text_key", type=str, default="text", help="Key for document text")
    parser.add_argument("--score_key", type=str, default="score", help="Key for document score")

    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "finewebedu100B"),
        help="Output directory for final .bin shards",
    )
    parser.add_argument(
        "--spool_dir",
        type=str,
        default=None,
        help="Directory for intermediate bucket spools (default: <out_dir>_spool)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["spool", "finalize", "all"],
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing spool/out dirs before running the selected stage(s)",
    )
    parser.add_argument(
        "--delete_spool",
        action="store_true",
        help="If set, delete bucket spool files after finalizing (saves disk space)",
    )

    parser.add_argument("--shard_size", type=int, default=10**8, help="Tokens per output shard")
    parser.add_argument("--max_score", type=float, default=5.0, help="Maximum possible score")
    parser.add_argument(
        "--score_bin_width",
        type=float,
        default=0.1,
        help="Score bin width for bucketing (smaller => closer to exact sort)",
    )

    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument(
        "--worker_type",
        type=str,
        default="batched",
        choices=["batched", "threads", "processes"],
        help="Tokenization backend (batched is much faster for tiktoken)",
    )
    parser.add_argument(
        "--batch_docs",
        type=int,
        default=1024,
        help="(threads) docs per batch submitted to the thread pool",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="(processes) chunksize for multiprocessing imap_unordered",
    )
    parser.add_argument(
        "--max_open_spool_files",
        type=int,
        default=64,
        help="Max simultaneously open bucket spool file handles",
    )
    parser.add_argument(
        "--spool_buffer_mb",
        type=int,
        default=256,
        help="Total in-memory buffer budget (MB) across buckets during spooling",
    )

    parser.add_argument(
        "--read_chunk_tokens",
        type=int,
        default=10_000_000,
        help="(finalize) how many tokens to read per chunk from bucket spools",
    )

    args = parser.parse_args()

    # adjust defaults that depend on other args
    if args.data_format in ("json", "jsonl") and args.data_glob == "**/*.parquet":
        args.data_glob = "**/*.jsonl" if args.data_format == "jsonl" else "**/*.json"
    if args.spool_dir is None:
        args.spool_dir = args.out_dir.rstrip("/\\") + "_spool"

    if args.overwrite:
        if os.path.isdir(args.spool_dir):
            shutil.rmtree(args.spool_dir)
        if os.path.isdir(args.out_dir) and args.stage in ("finalize", "all"):
            shutil.rmtree(args.out_dir)
    else:
        # prevent accidental appends/mixes
        if args.stage in ("spool", "all") and os.path.isdir(args.spool_dir):
            existing = glob.glob(os.path.join(args.spool_dir, "bucket_*.u16"))
            if existing:
                raise FileExistsError(
                    f"{args.spool_dir} already contains bucket spools; "
                    f"remove it or pass --overwrite"
                )
        if args.stage in ("finalize", "all") and os.path.isdir(args.out_dir):
            existing = glob.glob(os.path.join(args.out_dir, "finewebedu_*_*.bin"))
            if existing:
                raise FileExistsError(
                    f"{args.out_dir} already contains output shards; "
                    f"remove it or pass --overwrite"
                )

    bucketer = ScoreBucketer.from_width(max_score=args.max_score, bin_width=args.score_bin_width)

    if args.stage in ("spool", "all"):
        dataset = _load_local_dataset(
            data_dir=args.data_dir, data_format=args.data_format, data_glob=args.data_glob
        )
        _spool_stage(
            dataset=dataset,
            spool_dir=args.spool_dir,
            text_key=args.text_key,
            score_key=args.score_key,
            bucketer=bucketer,
            num_workers=args.num_workers,
            worker_type=args.worker_type,
            batch_docs=args.batch_docs,
            chunksize=args.chunksize,
            max_open_spool_files=args.max_open_spool_files,
            spool_buffer_mb=args.spool_buffer_mb,
        )

    if args.stage in ("finalize", "all"):
        _finalize_stage(
            spool_dir=args.spool_dir,
            out_dir=args.out_dir,
            shard_size=args.shard_size,
            delete_spool=args.delete_spool,
            read_chunk_tokens=args.read_chunk_tokens,
        )


if __name__ == "__main__":
    main()

