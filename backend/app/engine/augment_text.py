"""Rule-based text augmentation. No WordNet, NLTK, or model calls.

Randomness for a row comes only from SHA-256(seed, source_row_id, op_index).
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Iterable, Sequence

import pandas as pd

MAX_VARIANTS = 10
MAX_USES_PER_ROW = 5
MIN_WORDS_FOR_EDIT = 4
OPS = ("swap", "deletion", "insertion", "sentence_shuffle")

_SENTENCE = re.compile(r"[^.!?]+[.!?]*", flags=re.UNICODE)


class AugmentError(ValueError):
    """A dataset or config that augmentation cannot run on."""


def derive_rng(seed: int, source_row_id: int, op_index: int) -> random.Random:
    payload = f"{int(seed)}|{int(source_row_id)}|{int(op_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def derive_seed_int(seed: int, source_row_id: int, op_index: int) -> int:
    payload = f"{int(seed)}|{int(source_row_id)}|{int(op_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def op_index(variant: int, attempt: int, slot: int) -> int:
    return int(variant) * 1000 + int(attempt) * 10 + int(slot)


def _as_text(value) -> str | None:
    """Return a usable string, or None for NaN, empty, and non-strings."""
    if value is None or isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return None


def _swap(text: str, alpha: float, rng: random.Random) -> str:
    words = text.split()
    if len(words) < MIN_WORDS_FOR_EDIT or alpha <= 0:
        return text
    out = words[:]
    swaps = max(1, int(round(alpha * len(words))))
    for _ in range(swaps):
        i = rng.randrange(len(out))
        j = rng.randrange(len(out))
        out[i], out[j] = out[j], out[i]
    if out == words:
        return text
    return " ".join(out)


def _delete(text: str, alpha: float, rng: random.Random) -> str:
    words = text.split()
    if len(words) < MIN_WORDS_FOR_EDIT or alpha <= 0:
        return text
    kept = [word for word in words if rng.random() > alpha]
    if len(kept) == len(words):
        return text
    if not kept:
        kept = [words[rng.randrange(len(words))]]
    return " ".join(kept)


def _insert(text: str, alpha: float, rng: random.Random) -> str:
    words = text.split()
    if not words or alpha <= 0:
        return text
    out = words[:]
    inserts = max(1, int(round(alpha * len(words))))
    for _ in range(inserts):
        word = out[rng.randrange(len(out))]
        out.insert(rng.randrange(len(out) + 1), word)
    if out == words:
        return text
    return " ".join(out)


def _shuffle_sentences(text: str, alpha: float, rng: random.Random) -> str:
    if alpha <= 0 or rng.random() > alpha:
        return text
    parts = [part.strip() for part in _SENTENCE.findall(text) if part.strip()]
    if len(parts) < 2:
        return text
    order = parts[:]
    rng.shuffle(order)
    if order == parts:
        return text
    return " ".join(order)


_APPLY = {
    "swap": _swap,
    "deletion": _delete,
    "insertion": _insert,
    "sentence_shuffle": _shuffle_sentences,
}


def apply_ops(text: str, ops: Sequence[tuple[str, float]], seed: int, source_row_id: int, index: int) -> tuple[str, str]:
    current = text
    applied: list[str] = []
    for slot, (name, alpha) in enumerate(ops):
        fn = _APPLY.get(name)
        if fn is None:
            raise AugmentError(f"Unknown text operation '{name}'.")
        updated = fn(current, float(alpha), derive_rng(seed, source_row_id, index + slot))
        if updated != current:
            applied.append(name)
            current = updated
    return current, ",".join(applied)


def variants_for_text(
    value,
    ops: Sequence[tuple[str, float]],
    n_variants: int,
    seed: int,
    source_row_id: int,
) -> list[tuple[str, str]]:
    """Up to n_variants strings that differ from the source and from each other."""
    text = _as_text(value)
    n_variants = max(0, min(MAX_VARIANTS, int(n_variants)))
    if text is None or n_variants == 0 or not ops:
        return []
    seen = {text}
    produced: list[tuple[str, str]] = []
    attempt = 0
    limit = n_variants * 6
    while len(produced) < n_variants and attempt < limit:
        updated, names = apply_ops(text, ops, seed, source_row_id, op_index(len(produced), attempt, 0))
        attempt += 1
        if not names or updated in seen:
            continue
        seen.add(updated)
        produced.append((updated, names))
    return produced


def variant_budgets(labels: Iterable, strategy: str, n_variants: int) -> list[int]:
    """How many synthetic rows to draw from each source row.

    balance: minority classes are filled toward the majority count, and each source
    row contributes at most 5 synthetic rows.
    multiply: every row contributes the same capped count.
    """
    labels = [str(label) for label in labels]
    n_variants = max(0, min(MAX_VARIANTS, int(n_variants)))
    if strategy == "multiply":
        return [n_variants] * len(labels)
    if strategy != "balance":
        raise AugmentError("strategy must be 'balance' or 'multiply'.")
    counts: dict[str, int] = {}
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        counts[label] = counts.get(label, 0) + 1
        grouped.setdefault(label, []).append(index)
    majority = max(counts.values()) if counts else 0
    budgets = [0] * len(labels)
    for label, indexes in grouped.items():
        deficit = majority - len(indexes)
        if deficit <= 0:
            continue
        per = min(MAX_USES_PER_ROW, deficit // len(indexes))
        remainder = deficit - per * len(indexes)
        for offset, index in enumerate(indexes):
            extra = per + (1 if offset < remainder else 0)
            budgets[index] = min(MAX_USES_PER_ROW, extra)
    return budgets
