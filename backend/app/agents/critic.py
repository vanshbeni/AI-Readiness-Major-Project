"""
Critic: deterministic verification of agent output. Numbers in answers must match computed facts;
report text may only contain numbers through {{F#}} placeholders.
"""
import re
from typing import List, Tuple

from app.agents.facts import FactStore

_CITATION = re.compile(r"\[(?:[FTC]\d+)(?:\s*,\s*[FTC]\d+)*\]|\{\{\s*F\d+\s*\}\}")
_NUMBER = re.compile(r"(?<![\w.])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-−]?\d+(?:\.\d+)?")


def _numbers(text: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for m in _NUMBER.finditer(_CITATION.sub(" ", text)):
        raw = m.group(0)
        try:
            out.append((raw, float(raw.replace(",", "").replace("−", "-"))))
        except ValueError:
            continue
    return out


def _matches(value: float, raw: str, known: List[float]) -> bool:
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    tol_round = 0.5 * 10 ** (-decimals)
    for k in known:
        for candidate in (k, k * 100, abs(k), abs(k) * 100):
            if abs(candidate - value) <= max(tol_round, abs(candidate) * 0.005):
                return True
    return False


def verify_numbers(text: str, facts: FactStore, ignore_small_ints: int = 12) -> Tuple[int, List[str]]:
    """Returns (verified_count, unverified_number_strings). Small integers (counts like 'top 3') are ignored."""
    known = facts.numbers()
    verified, unverified = 0, []
    for raw, value in _numbers(text):
        if value == int(value) and abs(value) <= ignore_small_ints and "." not in raw:
            continue
        if 1900 <= value <= 2100 and value == int(value) and "," not in raw:
            continue  # years
        if _matches(value, raw, known):
            verified += 1
        elif raw not in unverified:
            unverified.append(raw)
    return verified, unverified


def stray_numbers(template: str) -> List[str]:
    """Numbers written directly in a {{F#}} template (allowed: list markers and years)."""
    stray = []
    for line in template.splitlines():
        body = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line)
        for raw, value in _numbers(body):
            if value == int(value) and (abs(value) <= 12 or 1900 <= value <= 2100) and "." not in raw:
                continue
            stray.append(raw)
    return stray
