import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.engine import transforms as T
from app.engine.profiler import duplicate_subset

MAX_ONE_HOT_CATEGORIES = 50
MAX_MULTI_HOT_TOKENS = 15


@dataclass
class PipelineResult:
    df_cleaned: pd.DataFrame          # cleaned, human-readable (before encoding) - used for re-scoring
    df_model_ready: pd.DataFrame      # final exported dataset (encoded if approved)
    applied_steps: List[str] = field(default_factory=list)
    script_code: str = ""
    dropped_informative_columns: List[str] = field(default_factory=list)


def _py(value: Any) -> Any:
    """Converts numpy scalars/containers to plain Python so they repr() cleanly into the script."""
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_or(value: Any, fallback: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if np.isnan(f) or np.isinf(f) else f


def _build_script(ops: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = [
        "# ==========================================================",
        "# Auto-Generated Pre-ML Preprocessing Pipeline",
        "# AI Data Readiness Platform",
        "# Reproduces the exported cleaned dataset using the exact fitted parameters.",
        "# Feature scaling and class rebalancing are intentionally NOT applied here:",
        "# fit them inside your model's cross-validation / training pipeline.",
        "# ==========================================================",
        inspect.getsource(T),
        "",
        "def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:",
        "    df = df_raw.copy()",
        "    df.columns = [str(c) for c in df.columns]",
    ]
    for name, kwargs in ops:
        args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        lines.append(f"    df = {name}(df, {args})")
    lines += [
        "    return df",
        "",
        "",
        "if __name__ == '__main__':",
        "    import sys",
        "    if len(sys.argv) < 2:",
        "        print('Usage: python pipeline.py raw_dataset.csv [output.csv]')",
        "        sys.exit(1)",
        "    cleaned = clean_data(pd.read_csv(sys.argv[1]))",
        "    out_path = sys.argv[2] if len(sys.argv) > 2 else 'cleaned_dataset.csv'",
        "    cleaned.to_csv(out_path, index=False)",
        "    print(f'Saved {cleaned.shape[0]} rows x {cleaned.shape[1]} columns to {out_path}')",
        "",
    ]
    return "\n".join(lines)


def execute_pipeline(
    df_raw: pd.DataFrame,
    recommendations: List[Dict[str, Any]],
    target_col: Optional[str] = None,
    problem_type: str = "classification",
) -> PipelineResult:
    """
    Executes the approved remediation steps in a safe order:
    missing-target rows -> duplicates -> identifiers -> datetime -> units -> multi-label ->
    invalid negatives -> column drops -> outlier capping -> imputation -> rare grouping -> encoding.
    Only approved recommendations are executed.
    """
    df = df_raw.copy()
    df.columns = [str(c) for c in df.columns]
    steps: List[str] = []
    ops: List[Tuple[str, Dict[str, Any]]] = []

    def apply(func_name: str, description: str, **kwargs):
        nonlocal df
        kwargs = _py(kwargs)
        df = getattr(T, func_name)(df, **kwargs)
        ops.append((func_name, kwargs))
        steps.append(description)

    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recommendations:
        if r.get("is_approved"):
            key = (r.get("params") or {}).get("method_key")
            if key:
                by_key[key].append(r)

    def is_feature(col: Optional[str]) -> bool:
        return bool(col) and col in df.columns and col != target_col

    # 1. Rows without a target label
    if by_key.get("drop_missing_target") and target_col and target_col in df.columns:
        n_missing = int(df[target_col].isna().sum())
        if n_missing:
            apply("drop_rows_missing", f"Target Labels: Dropped {n_missing} rows with a missing '{target_col}'.", column=target_col)

    # 2. Duplicate records (identifier columns ignored)
    if by_key.get("drop_duplicates"):
        subset = duplicate_subset(df)
        before = len(df)
        n_dups = int(df.duplicated(subset=subset).sum())
        if n_dups:
            apply("drop_duplicate_rows", f"Deduplication: Removed {n_dups} duplicate records ({before} -> {before - n_dups} rows).", subset=subset)

    # 3. Identifier columns
    id_cols = [r["column"] for r in by_key.get("drop_identifier", []) if is_feature(r["column"])]
    if id_cols:
        apply("drop_columns", f"Identifiers: Dropped non-predictive ID column(s): {', '.join(id_cols)}.", columns=id_cols)

    # 4. Datetime feature extraction
    for r in by_key.get("parse_datetime", []):
        col = r["column"]
        if not is_feature(col):
            continue
        parsed = T.parse_dates(df[col])
        fill_year = _finite_or(parsed.dt.year.median(), 2000.0)
        fill_month = _finite_or(parsed.dt.month.median(), 6.0)
        fill_day = _finite_or(parsed.dt.day.median(), 15.0)
        apply("extract_datetime",
              f"Datetime Engineering: Parsed '{col}' into '{col}_year', '{col}_month' and '{col}_day'.",
              column=col, fill_year=round(fill_year), fill_month=round(fill_month), fill_day=round(fill_day))

    # 5. Mixed value + unit strings
    for r in by_key.get("split_units", []):
        col = r["column"]
        if not is_feature(col):
            continue
        values = pd.Series([T.parse_unit(v)[0] for v in df[col]], dtype=float)
        fill_value = _finite_or(values.median(), 0.0)
        apply("split_units",
              f"Unit Disentanglement: Parsed '{col}' into numeric '{col}_value' and categorical '{col}_unit'.",
              column=col, fill_value=fill_value)

    # 6. Delimited multi-label lists
    for r in by_key.get("multi_hot", []):
        col = r["column"]
        if not is_feature(col):
            continue
        all_tokens = [t for cell in df[col] for t in T.split_tokens(cell)]
        tokens = pd.Series(all_tokens, dtype=object).value_counts().head(MAX_MULTI_HOT_TOKENS).index.tolist()
        apply("multi_hot", f"Multi-Label Encoding: Tokenized '{col}' into {len(tokens)} binary tag indicators.",
              column=col, tokens=[str(t) for t in tokens])

    # 7. Invalid negative values -> median of valid values
    for r in by_key.get("replace_negatives_median", []):
        col = r["column"]
        if not is_feature(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        valid = df[col][df[col] >= 0]
        fill_value = _finite_or(valid.median(), 0.0)
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            apply("replace_negatives",
                  f"Domain Validity: Replaced {n_neg} negative values in '{col}' with the valid median ({round(fill_value, 3)}).",
                  column=col, fill_value=fill_value)

    # 8. Column drops
    drop_reasons = {
        "drop_constant": "zero variance",
        "drop_collinear": "redundant / collinear",
        "drop_column_missing": "over 70% missing",
        "drop_high_cardinality": "high-cardinality free text",
    }
    informative_keys = {"drop_collinear", "drop_column_missing"}
    dropped_informative: List[str] = []
    for key, reason in drop_reasons.items():
        cols = [r["column"] for r in by_key.get(key, []) if is_feature(r["column"])]
        cols = list(dict.fromkeys(cols))
        if cols:
            apply("drop_columns", f"Dimensionality: Dropped {reason} column(s): {', '.join(cols)}.", columns=cols)
            if key in informative_keys:
                dropped_informative.extend(cols)

    # 9. Outlier capping (before imputation so fill statistics are not skewed by extremes)
    for r in by_key.get("cap_outliers", []):
        col = r["column"]
        params = r.get("params") or {}
        if not is_feature(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lb = _finite_or(params.get("lower_bound"), float(df[col].quantile(0.01)))
        ub = _finite_or(params.get("upper_bound"), float(df[col].quantile(0.99)))
        apply("clip_values", f"Outlier Capping: Winsorized '{col}' to [{lb}, {ub}].", column=col, lower=lb, upper=ub)

    # 10. Missing value imputation (method chosen by column type in the rule matrix)
    for key in ("impute_median", "impute_mean", "impute_mode", "impute_unknown"):
        for r in by_key.get(key, []):
            col = r["column"]
            if not is_feature(col) or not df[col].isna().any():
                continue
            series = df[col]
            numeric = pd.api.types.is_numeric_dtype(series)
            if key in ("impute_median", "impute_mean") and numeric:
                stat = series.median() if key == "impute_median" else series.mean()
                label = "Median" if key == "impute_median" else "Mean"
                if pd.isna(stat):
                    continue
                apply("fill_missing", f"Imputation: Replaced missing in '{col}' with {label} ({round(float(stat), 3)}).",
                      column=col, value=float(stat))
            elif key == "impute_unknown" and not numeric:
                apply("fill_missing", f"Imputation: Filled missing in '{col}' with 'Unknown' & added indicator 'has_{col}'.",
                      column=col, value="Unknown", add_flag=True)
            else:
                mode = series.mode()
                if mode.empty:
                    continue
                value = mode.iloc[0]
                apply("fill_missing", f"Imputation: Replaced missing in '{col}' with Mode ({value!r}).", column=col, value=value)

    # 11. Rare category grouping
    for r in by_key.get("group_rare", []):
        col = r["column"]
        if not is_feature(col):
            continue
        keep_n = int((r.get("params") or {}).get("keep_top", 10))
        keep = [str(v) for v in df[col].dropna().astype(str).value_counts().head(keep_n).index.tolist()]
        apply("group_rare", f"Encoding: Grouped categories in '{col}' into Top {len(keep)} + 'Other'.", column=col, keep=keep)

    df_cleaned = df.copy()

    # 12. One-hot encoding of remaining categorical features
    if by_key.get("one_hot_encode"):
        cat_cols = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
        encoded, skipped = [], []
        for col in cat_cols:
            counts = df[col].dropna().astype(str).value_counts()
            if len(counts) > MAX_ONE_HOT_CATEGORIES:
                skipped.append(col)
                continue
            categories = sorted(counts.index.tolist())
            baseline = counts.index[0] if len(counts) else None
            indicator_cats = [c for c in categories if c != baseline]
            apply("one_hot", f"Encoding: One-hot encoded '{col}' ({len(indicator_cats)} indicators, baseline '{baseline}').",
                  column=col, categories=indicator_cats)
            encoded.append(col)
        if skipped:
            steps.append(f"Encoding: Left {', '.join(skipped)} as text (more than {MAX_ONE_HOT_CATEGORIES} categories).")

    return PipelineResult(
        df_cleaned=df_cleaned,
        df_model_ready=df,
        applied_steps=steps,
        script_code=_build_script(ops),
        dropped_informative_columns=dropped_informative,
    )
