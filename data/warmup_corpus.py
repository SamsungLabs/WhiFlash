"""Long-text corpus for component-benchmark warmup prompts.

Component benchmarks need to evaluate at large ``base_kv_len`` (e.g. 4096+).
Reaching that via a short prompt + decode-loop warmup is expensive (thousands
of decode rounds, plus acceptance collapses past natural EOS). The cheap
alternative is to use a long prompt — prefill is one forward pass over N
tokens, vs N/avg_accept rounds.

This module caches a ~1.1M-token corpus at ``assets/warmup_corpus.txt`` (repo root),
auto-built on first use from the public-domain ``wikitext-103-raw-v1`` dataset.
Subsequent calls just read the cache.

Usage:
    corpus_ids = load_warmup_corpus_tokens(tokenizer)        # 1-D int tensor
    prompt_ids = slice_warmup_prompt(corpus_ids, 4096)       # [4086] tokens
    # Then feed prompt_ids.unsqueeze(0) into a prefill call.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Cached corpus path: <repo_root>/assets/warmup_corpus.txt
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "assets" / "warmup_corpus.txt"

# Default size: 1.1M tokens — covers any base_kv_len up to ~1M with margin.
_DEFAULT_TARGET_TOKENS = 1_100_000


def _build_corpus(min_tokens: int, cache_path: Path) -> None:
    """Stream ``wikitext-103-raw-v1`` and write a concatenated text file."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Building the warmup corpus requires the `datasets` package. "
            "Install with `pip install datasets`, or generate the file manually "
            f"at {cache_path}."
        ) from e

    # ~1.3 tokens per English word; 10% margin on top of caller's request.
    target_words = int(min_tokens / 1.3 * 1.1)

    print(f"[warmup_corpus] Building {cache_path} (target ~{min_tokens} tokens)…")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    chunks: list[str] = []
    total_words = 0
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        # Skip section headings ("= Title =", "== Subsection ==").
        if text.startswith("=") and text.endswith("="):
            continue
        chunks.append(text)
        total_words += len(text.split())
        if total_words >= target_words:
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"[warmup_corpus] wrote {len(chunks)} chunks (~{total_words} words)")


def load_warmup_corpus_tokens(
    tokenizer,
    min_tokens: int = _DEFAULT_TARGET_TOKENS,
    cache_path: Path | str = _DEFAULT_CACHE,
) -> torch.Tensor:
    """Return a 1-D ``LongTensor`` of token IDs of length ≥ ``min_tokens``.

    Builds and caches the corpus on first call. If the existing cache is too
    small for the requested ``min_tokens``, rebuilds it.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        _build_corpus(min_tokens, cache_path)

    text = cache_path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    if ids.shape[0] < min_tokens:
        # Cached corpus is too small for the requested size — rebuild larger.
        print(f"[warmup_corpus] cache has {ids.shape[0]} tokens, " f"rebuilding for {min_tokens}…")
        _build_corpus(min_tokens, cache_path)
        text = cache_path.read_text(encoding="utf-8")
        ids = tokenizer(text, return_tensors="pt").input_ids[0]

    return ids


def slice_warmup_prompt(
    corpus_ids: torch.Tensor,
    target_tokens: int,
    margin: int = 10,
) -> torch.Tensor:
    """Slice the corpus to ``target_tokens - margin`` tokens.

    ``margin`` accounts for the fact that prefill produces ``pos = N + 1`` and
    leaves the first generated token at index ``N``; a small extra cushion is
    helpful so the first benched round has headroom.
    """
    n = max(target_tokens - margin, 1)
    if n > corpus_ids.shape[0]:
        raise ValueError(
            f"requested {n} tokens but corpus has only {corpus_ids.shape[0]}. " f"Rebuild with a larger min_tokens."
        )
    return corpus_ids[:n]
