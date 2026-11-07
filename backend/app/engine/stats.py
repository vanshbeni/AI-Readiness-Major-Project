"""Association statistics shared by the detector, the scorer and the agents."""
import math
from typing import Optional

import numpy as np
import pandas as pd


def min_support(n_rows: int) -> int:
    """Smallest number of rows a category needs before it can be trusted as a feature value."""
    return max(5, int(math.ceil(0.01 * n_rows)))


def pool_rare(x: pd.Series, support: int, label: str = "__rare__") -> pd.Series:
    """Categories seen fewer than `support` times are pooled, so one-off values (names, IDs) cannot look predictive."""
    s = x.astype(str)
    counts = s.value_counts()
    rare = set(counts[counts < support].index)
    return s.where(~s.isin(rare), label) if rare else s


def _adjusted_eta(values: pd.Series, groups: pd.Series) -> float:
    """Correlation ratio corrected for the number of groups (chance level ~0 instead of ~sqrt(k/n))."""
    n, k = len(values), int(groups.nunique())
    if k < 2 or n <= k:
        return 0.0
    total = float(((values - values.mean()) ** 2).sum())
    if not total:
        return 0.0
    means = values.groupby(groups).transform("mean")
    eta2 = 1 - float(((values - means) ** 2).sum()) / total
    return float(np.sqrt(max(0.0, 1 - (1 - eta2) * (n - 1) / (n - k))))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    table = pd.crosstab(x, y).to_numpy(dtype=float)
    n = table.sum()
    r, k = table.shape
    if n <= 1 or min(r, k) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def association_pvalue(df: pd.DataFrame, col: str, target: str, problem_type: str) -> Optional[float]:
    """p-value that the column and the target are unrelated (chi-square / ANOVA / Pearson, rare categories pooled)."""
    from scipy import stats as st
    if col not in df.columns or target not in df.columns or col == target:
        return None
    data = df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    if len(data) < 10 or data[col].nunique() < 2 or data[target].nunique() < 2:
        return None
    x, y = data[col], data[target]
    x_num = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x) and x.nunique() > 20
    try:
        if problem_type == "classification":
            if x_num:
                groups = [g.astype(float).to_numpy() for _, g in x.groupby(y.astype(str)) if len(g) > 1]
                return float(st.f_oneway(*groups).pvalue) if len(groups) >= 2 else None
            xs = pool_rare(x, min_support(len(x)))
            if xs.nunique() < 2:
                return 1.0
            return float(st.chi2_contingency(pd.crosstab(xs, y.astype(str)))[1])
        yn = pd.to_numeric(y, errors="coerce")
        ok = yn.notna()
        if x_num:
            return float(st.pearsonr(x[ok].astype(float), yn[ok]).pvalue)
        xs = pool_rare(x[ok], min_support(int(ok.sum())))
        groups = [g.to_numpy(dtype=float) for _, g in yn[ok].groupby(xs) if len(g) > 1]
        return float(st.f_oneway(*groups).pvalue) if len(groups) >= 2 else 1.0
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def target_association(df: pd.DataFrame, col: str, target: str, problem_type: str) -> Optional[float]:
    """
    0..1 strength of the column's relationship with the target that generalises beyond the rows seen:
    bias-corrected Cramér's V / adjusted correlation ratio over categories with enough support, or |r| for numbers.
    A column whose values are (nearly) all unique scores ~0 instead of a spurious 1.
    """
    if col not in df.columns or target not in df.columns or col == target:
        return None
    data = df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    if len(data) < 10 or data[col].nunique() < 2:
        return None
    x, y = data[col], data[target]
    x_num = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x) and x.nunique() > 20
    try:
        if problem_type == "classification":
            if x_num:
                return round(_adjusted_eta(x.astype(float), y.astype(str)), 4)
            xs = pool_rare(x, min_support(len(x)))
            if xs.nunique() < 2:
                return 0.0
            return round(cramers_v(xs, y.astype(str)), 4)
        yn = pd.to_numeric(y, errors="coerce")
        ok = yn.notna()
        if x_num:
            r = np.corrcoef(x[ok].astype(float), yn[ok])[0, 1]
            return round(float(abs(r)), 4) if np.isfinite(r) else None
        xs = pool_rare(x[ok], min_support(int(ok.sum())))
        return round(_adjusted_eta(yn[ok].astype(float), xs), 4)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
