"""Deterministic target-leakage screening: features that predict the target (almost) perfectly on their own."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEAK_THRESHOLD = 0.98
MAX_ROWS = 20000
MAX_GROUP_SHARE = 0.5   # columns with more distinct values than this share of rows behave like IDs: skipped


def _purity(x: pd.Series, y: pd.Series) -> float:
    """Share of rows whose target equals the majority target of their x-group (1.0 = x determines y)."""
    counts = pd.crosstab(x.astype(str), y.astype(str))
    return float(counts.max(axis=1).sum() / counts.values.sum())


def _eta_squared(x: pd.Series, y: pd.Series) -> float:
    """Share of target variance explained by the group means of x."""
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    means = y.groupby(x.astype(str)).transform("mean")
    return float(1 - ((y - means) ** 2).sum() / total)


def leakage_candidates(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    skip: Optional[set] = None,
    threshold: float = LEAK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Returns [{column, score, method}] for features whose single-feature score >= threshold."""
    if target not in df.columns:
        return []
    data = df[df[target].notna()]
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=42)
    if len(data) < 10:
        return []
    y = data[target]
    classification = problem_type == "classification"
    if classification and y.nunique() < 2:
        return []
    baseline = float(y.astype(str).value_counts(normalize=True).iloc[0]) if classification else 0.0

    found: List[Dict[str, Any]] = []
    for col in data.columns:
        if col == target or (skip and col in skip):
            continue
        x = data[col]
        mask = x.notna()
        if mask.sum() < max(10, 0.5 * len(data)):
            continue
        xs, ys = x[mask], y[mask]
        n_unique = xs.nunique()
        if n_unique < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(xs) and not pd.api.types.is_bool_dtype(xs)
        score, method = 0.0, ""
        try:
            if classification:
                if not numeric or n_unique <= 50:
                    if n_unique > MAX_GROUP_SHARE * len(xs) or baseline > 0.95:
                        continue
                    score, method = _purity(xs, ys), "group purity"
                elif ys.nunique() == 2:
                    auc = roc_auc_score((ys.astype(str) == ys.astype(str).iloc[0]).astype(int), xs.astype(float))
                    score, method = float(max(auc, 1 - auc)), "single-feature ROC AUC"
            else:
                yn = pd.to_numeric(ys, errors="coerce")
                if numeric and n_unique > 50:
                    ok = yn.notna()
                    if ok.sum() > 10 and xs[ok].std() > 0 and yn[ok].std() > 0:
                        score, method = float(abs(np.corrcoef(xs[ok].astype(float), yn[ok])[0, 1])), "|Pearson r|"
                elif n_unique <= MAX_GROUP_SHARE * len(xs):
                    ok = yn.notna()
                    score, method = _eta_squared(xs[ok], yn[ok]), "eta squared"
        except (ValueError, TypeError):
            continue
        if score >= threshold:
            found.append({"column": str(col), "score": round(score, 4), "method": method})
    return sorted(found, key=lambda f: -f["score"])
