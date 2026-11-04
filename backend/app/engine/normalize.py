"""
Automatic, non-destructive normalisation that runs before profiling, scoring and cleaning.

Real exports hide problems that make a column look cleaner than it is: placeholder text (' NAN', '?', '   ') that
is really a blank, and numbers written as text ('$1,200', '12,5', 'thirty', 'SIXTY THOUSAND') that turn a numeric
column into a fake category. Without this pass the profiler counts none of those as missing, types the column as
categorical, and the health score and target suggestion are wrong. Every step is recorded as a prep op so the
exported script reproduces it, and reported as an issue so the score reflects what was found.
"""
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List

import pandas as pd

from app.engine.transforms import is_number_words, to_number

PLACEHOLDER_TOKENS = {
    "?", "-", "--", "---", ".", "na", "n/a", "n.a.", "none", "null", "nil", "nan", "unknown", "missing", "undefined",
    "#n/a", "#na", "tbd", "pending", "not available", "not applicable", "no data", "blank", "#value!", "#ref!",
    "#div/0!", "#null!", "#num!", "nat", "(blank)", "(null)", "empty",
}
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{9,18}$")
FORMATTED_NUM_RE = re.compile(r"^\s*\(?\s*([-+]?)\s*([^\d\s,()+-]*?)\s*([-+]?\s*\d[\d.,\s]*)\s*([^\d\s.,()]*)\s*\)?\s*$")
# Letters around a number only count as formatting when they are a currency code / word, a percent sign or a
# magnitude word; anything else ('C85', 'A-12', 'Block7', '12kg') is a code or a unit and must stay text.
CURRENCY_WORDS = {
    "rs", "rs.", "inr", "usd", "us$", "eur", "gbp", "jpy", "cny", "rmb", "aud", "a$", "cad", "c$", "chf", "nzd",
    "sgd", "hkd", "sek", "nok", "dkk", "kr", "zar", "brl", "r$", "mxn", "aed", "sar", "pkr", "bdt", "lkr", "npr",
    "idr", "rp", "myr", "rm", "php", "thb", "vnd", "krw", "rub", "try", "pln", "zl", "czk", "huf", "ils", "egp",
    "ngn", "kes", "%",
}
MAGNITUDE_WORDS = {"k", "thousand", "million", "mn", "billion", "bn", "lakh", "lakhs", "crore", "crores"}
# "1.234,56" or "12,5" / "12,50" are decimal commas; "12,345" (exactly three digits) is a thousands separator.
DECIMAL_COMMA_RE = re.compile(r"^-?\d{1,3}(?:\.\d{3})+,\d+$|^-?\d+,\d{1,2}$")
NUMERIC_TEXT_SHARE = 0.9


def _is_number_affix(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    if t in CURRENCY_WORDS or t in MAGNITUDE_WORDS:
        return True
    parts = t.split()
    if len(parts) > 1:
        return all(_is_number_affix(p) for p in parts)
    return all(unicodedata.category(ch) == "Sc" or ch == "%" for ch in t)


def looks_formatted_number(value: Any) -> bool:
    """'$1,234', '12,50 €', 'Rs. 900', '45%', '(40)', '60k' -> True; 'C85', 'B 12', 'SKU-123', '12kg' -> False."""
    m = FORMATTED_NUM_RE.match(str(value))
    return bool(m) and _is_number_affix(m.group(2)) and _is_number_affix(m.group(4))


def looks_numeric_text(value: Any) -> bool:
    return looks_formatted_number(value) or is_number_words(value)


def is_placeholder(value: Any) -> bool:
    s = str(value).strip().lower()
    return s == "" or s in PLACEHOLDER_TOKENS


def is_email(value: Any) -> bool:
    return bool(EMAIL_RE.match(str(value).strip()))


def is_phone(value: Any) -> bool:
    s = str(value).strip()
    return bool(PHONE_RE.match(s)) and sum(c.isdigit() for c in s) >= 9


def category_mapping(series: pd.Series) -> Dict[str, str]:
    """Maps spelling variants (case / surrounding whitespace) to the most frequent written form."""
    values = series.dropna().astype(str)
    groups: Dict[str, Counter] = {}
    for v in values:
        groups.setdefault(" ".join(v.split()).casefold(), Counter())[v] += 1
    mapping: Dict[str, str] = {}
    for counter in groups.values():
        if len(counter) < 2:
            continue
        top = max(counter.values())
        # Among frequent spellings (at least half as common as the top one), prefer a clean, non-shouting form.
        frequent = [v for v, c in counter.items() if c >= 0.5 * top]
        canonical = max(frequent, key=lambda v: (v == v.strip(), not v.strip().isupper(), not v.strip().islower(), counter[v])).strip()
        for variant in counter:
            if variant != canonical:
                mapping[variant] = canonical
    return mapping


def plan(df: pd.DataFrame, protect: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Normalisation steps for this frame: {column, kind, func, kwargs, description, count, pct, examples}."""
    protect = set(protect)
    n = max(len(df), 1)
    steps: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        if col in protect or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        nonnull = s.dropna()
        if nonnull.empty:
            continue
        strs = nonnull.astype(str)
        placeholder = strs.map(is_placeholder)
        padded = (strs != strs.str.strip()) & ~placeholder
        if padded.any():
            count = int(padded.sum())
            examples = [str(v) for v in strs[padded].drop_duplicates().head(4)]
            steps.append({
                "column": col, "kind": "padded_text", "func": "strip_text", "kwargs": {"column": col},
                "description": f"Prep (auto): trimmed leading/trailing spaces from {count} value(s) in '{col}' "
                               f"(e.g. {examples[0]!r}).",
                "count": count, "pct": round(count / n * 100, 2), "examples": examples,
            })
        if placeholder.any():
            tokens = strs[placeholder].value_counts()
            count = int(placeholder.sum())
            shown = ", ".join(repr(t) for t in tokens.index[:4])
            steps.append({
                "column": col, "kind": "disguised_missing", "func": "replace_with_nan",
                "kwargs": {"column": col, "values": [str(t) for t in tokens.index[:30]]},
                "description": f"Prep (auto): treated {count} placeholder cell(s) in '{col}' ({shown}) as missing.",
                "count": count, "pct": round(count / n * 100, 2), "examples": [str(t) for t in tokens.index[:4]],
            })
        rest = strs[~placeholder].str.strip()
        if len(rest) < 5 or rest.str.fullmatch(r"-?\d+(\.\d+)?").all():
            continue
        numeric_like = rest.map(looks_numeric_text)
        if numeric_like.mean() < NUMERIC_TEXT_SHARE:
            continue
        digits = rest[rest.str.contains(r"\d")].map(lambda v: re.sub(r"[^\d.,-]", "", v))
        decimal = "," if len(digits) and digits.map(lambda v: bool(DECIMAL_COMMA_RE.match(v))).mean() > 0.5 else "."
        parsed = rest.map(lambda v: to_number(v, decimal))
        if parsed.notna().mean() < NUMERIC_TEXT_SHARE:
            continue
        odd = rest[~rest.str.fullmatch(r"-?\d+(\.\d+)?")]
        lost = int(parsed.isna().sum())
        steps.append({
            "column": col, "kind": "numeric_stored_as_text", "func": "parse_numeric",
            "kwargs": {"column": col, "decimal": decimal},
            "description": f"Prep (auto): read '{col}' as numbers ({int(len(odd))} value(s) written as text, e.g. "
                           f"{', '.join(repr(v) for v in odd.drop_duplicates().head(3))})"
                           + (f"; {lost} unreadable value(s) became missing." if lost else "."),
            "count": int(len(odd)), "pct": round(len(odd) / n * 100, 2),
            "examples": [str(v) for v in odd.drop_duplicates().head(4)], "unreadable": lost, "decimal": decimal,
        })
    return steps


def issues_from(steps: List[Dict[str, Any]], target_col: str = None) -> List[Dict[str, Any]]:
    """Issue dicts (detector format) for what the normalisation found, so the health score reflects it."""
    out = []
    for st in steps:
        col, pct = st["column"], st["pct"]
        role = " (the target)" if col == target_col else ""
        if st["kind"] == "padded_text":
            out.append({
                "issue_type": "padded_text", "column": col,
                "severity": "medium" if pct > 20 else "low",
                "title": f"Untrimmed Whitespace in '{col}'{role}: {st['count']} value(s) (e.g. {st['examples'][0]!r})",
                "details": {"column": col, "count": st["count"], "pct": pct, "examples": st["examples"]},
            })
        elif st["kind"] == "disguised_missing":
            out.append({
                "issue_type": "disguised_missing", "column": col,
                "severity": "high" if pct > 10 else ("medium" if pct > 2 else "low"),
                "title": f"Hidden Missing Values in '{col}'{role}: {st['count']} placeholder cell(s) ({', '.join(repr(e) for e in st['examples'][:3])})",
                "details": {"column": col, "count": st["count"], "pct": pct, "examples": st["examples"]},
            })
        else:
            out.append({
                "issue_type": "numeric_stored_as_text", "column": col,
                "severity": "high" if pct > 5 else "medium",
                "title": f"Numbers Stored as Text in '{col}'{role} (e.g. {', '.join(repr(e) for e in st['examples'][:2])})",
                "details": {"column": col, "count": st["count"], "pct": pct, "examples": st["examples"],
                            "unreadable": st.get("unreadable", 0)},
            })
    return out
