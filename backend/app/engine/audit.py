"""
Post-cleaning audit: inspects the model-ready output the way a reviewer would and reports every defect that is
still present (blanks, text left unencoded, near-empty indicators, constant / duplicate / identifier columns,
whole numbers stored as floats) plus advisory findings (small sample, imputation and capping footprints,
one-hot baselines, class balance). Each defect carries a `fix` code that the executor applies and re-audits,
so the exported data is checked against its own result instead of trusting the plan.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.engine.profiler import infer_column_type


def min_support(n_rows: int) -> int:
    return max(5, int(np.ceil(0.01 * n_rows)))


def _is_binary(s: pd.Series) -> bool:
    vals = set(pd.unique(s.dropna()))
    return bool(vals) and vals.issubset({0, 1, 0.0, 1.0, True, False})


def _finding(check: str, severity: str, detail: str, column: Optional[str] = None, fix: Optional[str] = None, **extra) -> Dict[str, Any]:
    return {"check": check, "severity": severity, "column": column, "detail": detail, "fix": fix, **extra}


def detect_defects(df: pd.DataFrame, target_col: Optional[str], encode_allowed: bool) -> List[Dict[str, Any]]:
    """Defects in the model-ready frame. Every finding with a `fix` is something the executor can repair."""
    out: List[Dict[str, Any]] = []
    n = len(df)
    if n == 0:
        return [_finding("empty_output", "critical", "The cleaned dataset has no rows left.")]
    support = min_support(n)
    features = [c for c in df.columns if c != target_col]

    seen_hashes: Dict[Any, str] = {}
    for c in features:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if not numeric:
            out.append(_finding("non_numeric_feature", "high" if encode_allowed else "medium",
                                f"'{c}' is still text ({s.nunique()} distinct values); models need numbers.",
                                c, "encode" if encode_allowed else None))
            continue
        if np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
            out.append(_finding("infinite_values", "high", f"'{c}' contains infinite values.", c, "fill_numeric"))
            continue
        n_missing = int(s.isna().sum())
        if n_missing:
            out.append(_finding("missing_values_left", "high", f"'{c}' still has {n_missing} blank cell(s).", c, "fill_numeric",
                                count=n_missing))
            continue
        nunique = int(s.nunique())
        if nunique <= 1:
            out.append(_finding("constant_feature", "medium", f"'{c}' has a single value in every row; it carries no signal.", c, "drop"))
            continue
        if _is_binary(s):
            ones = int((s == 1).sum())
            minority = min(ones, n - ones)
            if minority < support:
                out.append(_finding("sparse_indicator", "high",
                                    f"'{c}' is 1 for only {minority} of {n} rows (needs at least {support}); a model would "
                                    f"memorise those rows instead of learning a pattern.", c, "drop", count=minority))
                continue
            if not pd.api.types.is_integer_dtype(s):
                out.append(_finding("binary_as_float", "low", f"'{c}' holds only 0/1 but is stored as decimals.", c, "cast_int"))
        else:
            top_share = int(s.value_counts().iloc[0])
            if n - top_share < support:
                out.append(_finding("near_constant_feature", "medium",
                                    f"'{c}' has the same value in all but {n - top_share} of {n} rows (needs {support}+ to vary); "
                                    f"it carries almost no signal.", c, "drop"))
                continue
            if not pd.api.types.is_integer_dtype(s) and bool((s % 1 == 0).all()) and float(s.abs().max()) < 2 ** 53:
                out.append(_finding("integer_as_float", "low", f"'{c}' holds whole numbers but is stored as decimals.", c, "cast_int"))
        if infer_column_type(s, str(c)) == "id":
            out.append(_finding("identifier_feature", "high", f"'{c}' is a running identifier (unique consecutive integers).", c, "drop"))
            continue
        key = pd.util.hash_pandas_object(s, index=False).sum()
        twin = seen_hashes.get(key)
        if twin is not None and df[twin].equals(s):
            out.append(_finding("duplicate_column", "medium", f"'{c}' is an exact copy of '{twin}'.", c, "drop", twin=twin))
            continue
        seen_hashes[key] = c

    if target_col and target_col in df.columns and df[target_col].isna().any():
        out.append(_finding("missing_target", "critical", f"The target '{target_col}' has blank values.", target_col))
    return out


def advisories(df: pd.DataFrame, target_col: Optional[str], problem_type: str, lineage: Dict[str, Dict[str, Any]],
               fill_notes: List[Dict[str, Any]], cap_notes: List[Dict[str, Any]], baselines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Things a reviewer should know about the output that are not defects to auto-fix."""
    out: List[Dict[str, Any]] = []
    n = len(df)
    features = [c for c in df.columns if c != target_col]
    if n < 1000:
        out.append(_finding("small_sample", "medium" if n < 500 else "low",
                            f"Only {n} rows: scores can swing between splits. Use stratified cross-validation and report the "
                            f"mean and spread, not a single split.", count=n))
    if features and len(features) > max(10, n / 10):
        out.append(_finding("many_features_per_row", "medium",
                            f"{len(features)} features for {n} rows; prefer regularised models or feature selection inside CV."))
    for note in fill_notes:
        flag = note.get("flag")
        how = note["how"]
        out.append(_finding("imputation_footprint", "info",
                            f"{note['count']} blank value(s) in '{note['column']}' were filled with {how}"
                            + (f"; '{flag}' marks exactly which rows were filled." if flag else
                               "; too few to need a separate flag."), note["column"], count=note["count"]))
    for note in cap_notes:
        out.append(_finding("capping_footprint", "info",
                            f"{note['count']} extreme value(s) in '{note['column']}' were capped to [{note['lower']}, {note['upper']}]"
                            + (f"; '{note['flag']}' marks them, so the repeated boundary value is explainable." if note.get("flag") else "."),
                            note["column"], count=note["count"]))
    for b in baselines:
        out.append(_finding("one_hot_baseline", "info",
                            f"'{b['column']}' became {len(b['indicators'])} indicator(s); rows where all are 0 mean '{b['baseline']}' "
                            f"(the most common value, left out on purpose).", b["column"], baseline=b["baseline"]))
    if target_col and target_col in df.columns and problem_type == "classification":
        counts = df[target_col].value_counts(normalize=True)
        if len(counts) >= 2:
            out.append(_finding("class_balance", "info" if counts.iloc[0] < 0.75 else "medium",
                                f"Most common class '{counts.index[0]}' is {round(float(counts.iloc[0]) * 100, 1)}% of rows: always "
                                f"predicting it scores that accuracy, so judge models on balanced accuracy / F1 / ROC-AUC against "
                                f"that baseline."))
    out.append(_finding("fit_scope", "info",
                        "Fill values, caps and category lists were learned from this whole file. For an unbiased final score, "
                        "re-fit them on the training folds only (the exported script lists every fitted value)."))
    return out
