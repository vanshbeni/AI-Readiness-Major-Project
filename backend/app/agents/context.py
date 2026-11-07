"""
Compact, privacy-aware dataset context for agents. Everything here is computed by code; agents only reason over it.
"""
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.agents.policy import category_mapping
from app.core.config import settings
from app.engine.normalize import (  # noqa: F401  (re-exported for the agents)
    DECIMAL_COMMA_RE, EMAIL_RE, PHONE_RE, PLACEHOLDER_TOKENS, looks_formatted_number, looks_numeric_text,
)
from app.engine.stats import min_support, pool_rare, target_association  # noqa: F401
from app.engine.transforms import to_number

SENTINEL_VALUES = {-1, -9, -99, -999, -9999, -99999, 99, 999, 9999, 99999, 999999}
PLAIN_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
DATE_TRIPLE_RE = re.compile(r"^\s*(\d{1,4})[/.-](\d{1,2})[/.-](\d{1,4})")


def redact(value: Any) -> str:
    s = str(value)[:40]
    if EMAIL_RE.match(s.strip()):
        return "<email>"
    if PHONE_RE.match(s.strip()) and sum(c.isdigit() for c in s) >= 9:
        return "<phone/long number>"
    return s


def _strings(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip()


def column_signals(df: pd.DataFrame, col: str, inferred: str) -> Dict[str, Any]:
    s = df[col]
    n = max(len(s), 1)
    nonnull = s.dropna()
    sig: Dict[str, Any] = {
        "column": col,
        "type": inferred,
        "missing_pct": round(float(s.isna().mean() * 100), 2),
        "unique": int(nonnull.nunique()),
    }
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

    if numeric and len(nonnull):
        clean = nonnull.astype(float)
        sig["min"], sig["max"] = round(float(clean.min()), 4), round(float(clean.max()), 4)
        sig["mean"], sig["median"] = round(float(clean.mean()), 4), round(float(clean.median()), 4)
        sig["share_non_negative"] = round(float((clean >= 0).mean()), 4)
        counts = clean.value_counts()
        q01, q99 = clean.quantile(0.01), clean.quantile(0.99)
        sentinels = [
            {"value": float(v), "count": int(c)}
            for v, c in counts.items()
            if v in SENTINEL_VALUES and c >= max(2, 0.005 * n) and (v < q01 or v > q99 or c / n > 0.02)
            and (v < clean.median() - 5 * (clean.std() or 1) or v > clean.median() + 5 * (clean.std() or 1) or v < 0 <= clean.median())
        ]
        if sentinels:
            sig["sentinel_candidates"] = sentinels
        if pd.api.types.is_integer_dtype(s) or bool((clean % 1 == 0).all()):
            uniq = np.sort(clean.unique())
            sig["integer"] = True
            if len(uniq) == len(clean) and len(uniq) > 2 and bool(np.all(np.diff(uniq) == 1)):
                sig["consecutive_unique_integers"] = True
            if 1800 <= uniq.min() and uniq.max() <= 2100:
                sig["year_like_range"] = True
        elif nonnull.nunique() == len(nonnull):
            sig["all_unique_floats"] = True
        top_share = float(counts.iloc[0] / len(clean)) if len(counts) else 0.0
        if top_share >= 0.95:
            sig["dominant_value_share"] = round(top_share, 4)
    elif len(nonnull):
        strs = _strings(s)
        low = strs.str.lower()
        placeholders = low[low.isin(PLACEHOLDER_TOKENS) | (low == "")].value_counts()
        if len(placeholders):
            sig["placeholder_tokens"] = {str(k): int(v) for k, v in placeholders.items()}
        rest = strs[~(low.isin(PLACEHOLDER_TOKENS) | (low == ""))]
        if len(rest):
            looks_numeric = rest.map(looks_numeric_text)
            sig["numeric_parse_ratio"] = round(float(looks_numeric.mean()), 3)
            sig["plain_number_ratio"] = round(float(rest.map(lambda v: bool(PLAIN_NUM_RE.match(v))).mean()), 3)
            if sig["numeric_parse_ratio"] >= 0.8:
                comma = rest.map(lambda v: bool(DECIMAL_COMMA_RE.match(re.sub(r"[^\d.,-]", "", v))))
                sig["decimal_comma_ratio"] = round(float(comma.mean()), 3)
                parsed = rest.map(lambda v: to_number(v, "," if sig["decimal_comma_ratio"] > 0.5 else "."))
                sig["parsed_numeric_ratio"] = round(float(parsed.notna().mean()), 3)
            triples = rest.map(lambda v: DATE_TRIPLE_RE.match(v))
            matched = triples[triples.notna()]
            if len(matched) >= 0.5 * len(rest):
                first = matched.map(lambda m: int(m.group(1)))
                second = matched.map(lambda m: int(m.group(2)))
                sig["date_triples"] = {
                    "share": round(len(matched) / len(rest), 3),
                    "first_over_12": int(((first > 12) & (first <= 31)).sum()),
                    "second_over_12": int((second > 12).sum()),
                    "year_first": int((first > 31).sum()),
                }
            if inferred not in ("datetime",):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample = rest.sample(min(500, len(rest)), random_state=42)
                    parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed", utc=True)
                if parsed_dates.notna().mean() >= 0.8 and sig.get("numeric_parse_ratio", 0) < 0.8:
                    sig["parses_as_date_ratio"] = round(float(parsed_dates.notna().mean()), 3)
            sig["email_ratio"] = round(float(rest.map(lambda v: bool(EMAIL_RE.match(v))).mean()), 3)
            sig["phone_ratio"] = round(float(rest.map(lambda v: bool(PHONE_RE.match(v)) and sum(c.isdigit() for c in v) >= 9).mean()), 3)
            if sig["email_ratio"] < 0.05:
                sig.pop("email_ratio")
            if sig["phone_ratio"] < 0.05:
                sig.pop("phone_ratio")
            if nonnull.nunique() <= 500:
                mapping = category_mapping(nonnull)
                if mapping:
                    sig["spelling_variants"] = {"groups": len(set(mapping.values())), "mapping": dict(list(mapping.items())[:12])}
            delim = rest.map(lambda v: "," in v or ";" in v or "|" in v or "/" in v)
            if delim.mean() >= 0.4 and nonnull.nunique() > 10 and sig.get("numeric_parse_ratio", 0) < 0.8 and "date_triples" not in sig:
                sig["delimited_list_ratio"] = round(float(delim.mean()), 3)

    if settings.AI_PRIVACY_MODE == "samples" and len(nonnull):
        sig["samples"] = [redact(v) for v in nonnull.drop_duplicates().head(6).tolist()]
    if settings.AI_PRIVACY_MODE == "schema":
        sig = {k: v for k, v in sig.items() if k in ("column", "type", "missing_pct", "unique")}
    return sig


def dataset_brief(df: pd.DataFrame, types: Dict[str, str], target: Optional[str], problem_type: Optional[str]) -> Dict[str, Any]:
    brief: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "problem_type": problem_type,
        "column_signals": [column_signals(df, c, types.get(c, "text")) for c in df.columns[:80]],
    }
    if df.shape[1] > 80:
        brief["note"] = f"Only the first 80 of {df.shape[1]} columns are summarised."
    return brief
