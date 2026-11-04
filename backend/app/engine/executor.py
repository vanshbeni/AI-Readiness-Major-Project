import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.engine import audit as A
from app.engine import normalize as N
from app.engine import transforms as T
from app.engine.profiler import duplicate_subset

MAX_ONE_HOT_CATEGORIES = 50
MAX_MULTI_HOT_TOKENS = 15
MAX_GROUP_LEVELS = 12
MIN_GROUP_EFFECT = 0.05
MAX_AUDIT_ROUNDS = 4
MAX_FILL_SHARE = 0.7


@dataclass
class PipelineResult:
    df_cleaned: pd.DataFrame          # cleaned, human-readable (before encoding) - used for re-scoring
    df_model_ready: pd.DataFrame      # final exported dataset (encoded if approved)
    applied_steps: List[str] = field(default_factory=list)
    script_code: str = ""
    dropped_informative_columns: List[str] = field(default_factory=list)
    audit: List[Dict[str, Any]] = field(default_factory=list)
    data_dictionary: List[Dict[str, Any]] = field(default_factory=list)


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


def _new_column_meanings(func: str, kwargs: Dict[str, Any], before: pd.DataFrame, new_cols: List[str]) -> Dict[str, str]:
    """Plain-language meaning of every column an operation creates (feeds the data dictionary)."""
    col = kwargs.get("column")
    if func == "one_hot":
        names = T.indicator_names(before, col, kwargs["categories"])
        return {n: f"1 if '{col}' was '{cat}', else 0" for n, cat in zip(names, kwargs["categories"])}
    if func == "multi_hot":
        names = T.indicator_names(before, col, kwargs["tokens"])
        return {n: f"1 if the list in '{col}' contains '{tok}', else 0" for n, tok in zip(names, kwargs["tokens"])}
    if func == "word_flags":
        names = T.indicator_names(before, f"{col}_word", kwargs["words"])
        return {n: f"1 if the word '{w}' appears in '{col}', else 0" for n, w in zip(names, kwargs["words"])}
    meanings: Dict[str, str] = {}
    for n in new_cols:
        if func in ("fill_missing", "fill_by_group") and n.endswith("_was_missing"):
            meanings[n] = f"1 if '{col}' was blank in the raw data (and was filled), else 0"
        elif func in ("fill_missing", "presence_flag") and n.startswith("has_"):
            meanings[n] = f"1 if '{col}' had a recorded value, 0 if it was blank"
        elif func == "clip_values":
            meanings[n] = f"1 if '{col}' was an extreme value capped to [{kwargs['lower']}, {kwargs['upper']}], else 0"
        elif func == "extract_prefix":
            meanings[n] = f"Letter prefix of the code in '{col}' (rare prefixes = 'Other', blanks = 'Missing')"
        elif func == "extract_datetime":
            meanings[n] = f"{n.rsplit('_', 1)[-1].capitalize()} part of the date in '{col}'"
        elif func == "split_units":
            meanings[n] = (f"Numeric amount parsed from '{col}'" if n.endswith("_value") else f"Unit label parsed from '{col}'")
        else:
            meanings[n] = f"Derived from '{col}'"
    return meanings


def _best_group(df: pd.DataFrame, col: str, target_col: Optional[str], support: int) -> Optional[Tuple[str, float]]:
    """The low-cardinality feature that best explains `col` (adjusted eta squared), for group-wise imputation."""
    y = df[col]
    ok = y.notna()
    best: Optional[Tuple[str, float]] = None
    for g in df.columns:
        if g in (col, target_col):
            continue
        s = df[g]
        nun = int(s.nunique(dropna=True))
        if nun < 2 or nun > MAX_GROUP_LEVELS:
            continue
        sub = pd.DataFrame({"k": s[ok].map(lambda v: None if pd.isna(v) else str(v)), "v": y[ok].astype(float)}).dropna()
        if len(sub) <= nun + 1 or (sub["k"].value_counts() >= support).sum() < 2:
            continue
        total = float(((sub["v"] - sub["v"].mean()) ** 2).sum())
        if not total:
            continue
        means = sub.groupby("k")["v"].transform("mean")
        eta2 = 1 - float(((sub["v"] - means) ** 2).sum()) / total
        adj = 1 - (1 - eta2) * (len(sub) - 1) / (len(sub) - nun)
        if adj >= MIN_GROUP_EFFECT and (best is None or adj > best[1]):
            best = (str(g), round(float(adj), 3))
    return best


def execute_pipeline(
    df_raw: pd.DataFrame,
    recommendations: List[Dict[str, Any]],
    target_col: Optional[str] = None,
    problem_type: str = "classification",
    prep_ops: Optional[List[Tuple[str, Dict[str, Any], str]]] = None,
) -> PipelineResult:
    """
    Executes the approved remediation steps in a safe order:
    approved data preparation (agent/user column decisions) -> missing-target rows -> duplicates ->
    identifiers -> datetime -> units -> multi-label -> code prefixes / word flags -> invalid negatives ->
    column drops / presence flags -> extreme-value capping -> imputation -> rare grouping -> encoding ->
    self-audit of the output (repairs + re-audit until clean).
    Only approved recommendations are executed; the audit repairs are recorded in the script like any other step.
    prep_ops: (transform name, kwargs, description) triples produced by app.agents.policy.
    """
    df = df_raw.copy()
    df.columns = [str(c) for c in df.columns]
    steps: List[str] = []
    ops: List[Tuple[str, Dict[str, Any]]] = []
    lineage: Dict[str, Dict[str, Any]] = {c: {"source": c, "origin": "original", "meaning": "", "steps": []} for c in df.columns}
    fill_notes: List[Dict[str, Any]] = []
    cap_notes: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []

    def apply(func_name: str, description: str, **kwargs):
        nonlocal df
        kwargs = _py(kwargs)
        before = df
        df = getattr(T, func_name)(df, **kwargs)
        ops.append((func_name, kwargs))
        steps.append(description)
        before_cols = set(before.columns)
        new_cols = [c for c in df.columns if c not in before_cols]
        meanings = _new_column_meanings(func_name, kwargs, before, new_cols) if new_cols else {}
        src = kwargs.get("column")
        touched = set(kwargs.get("columns") or []) if func_name == "cast_integer" else {src}
        for c in df.columns:
            if c in new_cols:
                lineage[c] = {"source": src or "", "origin": "derived", "meaning": meanings.get(c, ""), "steps": [description]}
            elif c in touched and c in lineage:
                lineage[c]["steps"].append(description)
        for c in before_cols - set(df.columns):
            lineage.pop(c, None)

    # 0. Infinite values are never valid measurements. Always part of the script: CSV readers differ on 'inf'.
    negatives_blanked: Dict[str, int] = {}
    for c in df.columns:
        if c != target_col and pd.api.types.is_float_dtype(df[c]):
            n_inf = int(np.isinf(df[c].to_numpy(dtype=float, na_value=np.nan)).sum())
            if n_inf:
                negatives_blanked[c] = n_inf
    df = T.infinite_to_missing(df)
    ops.append(("infinite_to_missing", {}))
    if negatives_blanked:
        steps.append("Domain Validity: Treated infinite values as blanks in "
                     + ", ".join(f"'{c}' ({n})" for c, n in negatives_blanked.items()) + "; they are filled with the other gaps.")

    # 0b. Approved data preparation, recorded so the exported script reproduces it
    for func_name, kwargs, description in prep_ops or []:
        apply(func_name, description, **kwargs)

    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_keys = set()
    for r in recommendations:
        key = (r.get("params") or {}).get("method_key")
        if key:
            all_keys.add(key)
            if r.get("is_approved"):
                by_key[key].append(r)

    def is_feature(col: Optional[str]) -> bool:
        return bool(col) and col in df.columns and col != target_col

    def support() -> int:
        return A.min_support(len(df))

    # 0c. One spelling per category value ('DELHI' / 'delhi' -> 'Delhi'), before counting duplicates or categories
    for r in by_key.get("merge_variants", []):
        col = r["column"]
        if not col or col not in df.columns:
            continue
        mapping = N.category_mapping(df[col])
        if mapping:
            n_cells = int(df[col].astype(str).isin(set(mapping)).sum())
            apply("map_values", f"Consistency: Merged {len(mapping)} spelling/case variant(s) in '{col}' ({n_cells} cells).",
                  column=col, mapping=mapping)

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

    # 6. Delimited multi-label lists (only tokens with enough rows to learn from)
    for r in by_key.get("multi_hot", []):
        col = r["column"]
        if not is_feature(col):
            continue
        all_tokens = [t for cell in df[col] for t in set(T.split_tokens(cell))]
        counts = pd.Series(all_tokens, dtype=object).value_counts()
        tokens = [str(t) for t, c in counts.items() if c >= support()][:MAX_MULTI_HOT_TOKENS]
        if not tokens:
            continue
        apply("multi_hot", f"Multi-Label Encoding: Tokenized '{col}' into {len(tokens)} binary tag indicators.",
              column=col, tokens=tokens)

    # 6b. Keep the reusable part of high-cardinality text before it is dropped
    for r in by_key.get("extract_prefix", []):
        col = r["column"]
        if not is_feature(col):
            continue
        prefixes = df[col].map(T.code_prefix).dropna().value_counts()
        keep = [str(p) for p, c in prefixes.items() if c >= support()][:15]
        if len(keep) >= 2:
            apply("extract_prefix", f"Feature Extraction: Created '{col}_prefix' from the letter prefix of '{col}' ({', '.join(keep[:6])}).",
                  column=col, keep=keep)
    for r in by_key.get("word_flags", []):
        col = r["column"]
        if not is_feature(col):
            continue
        sets = df[col].map(T.word_set)
        words = [w for w in (r.get("params") or {}).get("words", []) if support() <= int(sets.map(lambda s, w=w: w in s).sum()) <= len(df) - support()]
        if words:
            apply("word_flags", f"Feature Extraction: Created {len(words)} word flag(s) from '{col}' ({', '.join(words[:6])}).",
                  column=col, words=words)

    # 7. Impossible negatives -> blanks, imputed and flagged with the column's other gaps in step 10
    for r in by_key.get("replace_negatives_median", []):
        col = r["column"]
        if not is_feature(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            apply("negatives_to_missing",
                  f"Domain Validity: Treated {n_neg} impossible negative value(s) in '{col}' as blanks (filled with the other gaps).",
                  column=col)
            negatives_blanked[col] = negatives_blanked.get(col, 0) + n_neg

    # 8. Column drops and presence flags for mostly-empty columns
    for r in by_key.get("missing_indicator", []):
        col = r["column"]
        if not is_feature(col):
            continue
        pct = round(float(df[col].isna().mean() * 100), 1)
        apply("presence_flag", f"Sparse Column: Replaced '{col}' ({pct}% blank) with 'has_{col}' (1 = value recorded).", column=col)
    drop_reasons = {
        "drop_leakage": "target leakage",
        "drop_constant": "zero variance",
        "drop_collinear": "redundant / collinear",
        "drop_column_missing": "over 70% missing",
        "drop_high_cardinality": "high-cardinality free text",
        "drop_personal_data": "personal data (privacy risk, no generalisable signal)",
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

    # 9. Extreme-value capping (before imputation so fill statistics are not skewed by extremes)
    for r in by_key.get("cap_outliers", []):
        col = r["column"]
        params = r.get("params") or {}
        if not is_feature(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lb = _finite_or(params.get("lower_bound"), float(df[col].quantile(0.01)))
        ub = _finite_or(params.get("upper_bound"), float(df[col].quantile(0.99)))
        lb, ub = max(lb, float(df[col].min())), min(ub, float(df[col].max()))
        n_capped = int(((df[col] < lb) | (df[col] > ub)).sum())
        if not n_capped:
            continue
        flag = bool(params.get("add_flag", False)) and n_capped >= support()
        apply("clip_values", f"Outlier Capping: Capped {n_capped} value(s) in '{col}' to [{round(lb, 3)}, {round(ub, 3)}]"
              + (f" and flagged them in '{col}_was_capped'." if flag else "."),
              column=col, lower=lb, upper=ub, add_flag=flag)
        cap_notes.append({"column": col, "count": n_capped, "lower": round(lb, 3), "upper": round(ub, 3),
                          "flag": f"{col}_was_capped" if flag else None})

    # 10. Missing value imputation (method chosen by column type in the rule matrix)
    imputed_cols = {r["column"] for k in ("impute_median", "impute_mean", "impute_mode", "impute_unknown") for r in by_key.get(k, [])}
    for col in negatives_blanked:
        if col not in imputed_cols:
            by_key["impute_median"].append({"column": col, "params": {"add_flag": True, "group_by": "auto"}})
    for key in ("impute_median", "impute_mean", "impute_mode", "impute_unknown"):
        for r in by_key.get(key, []):
            col = r["column"]
            if not is_feature(col) or not df[col].isna().any():
                continue
            params = r.get("params") or {}
            series = df[col]
            n_blank = int(series.isna().sum())
            numeric = pd.api.types.is_numeric_dtype(series)
            if n_blank > MAX_FILL_SHARE * len(df):
                apply("presence_flag", f"Sparse Column: '{col}' is {round(n_blank / len(df) * 100, 1)}% blank, too sparse to fill; "
                      f"replaced it with 'has_{col}' (1 = value recorded).", column=col)
                continue
            if key in ("impute_median", "impute_mean") and numeric:
                flag = params.get("add_flag", True) is not False and n_blank >= support()
                flag_name = T.fresh_name(df, f"{col}_was_missing") if flag else None
                stat = series.median() if key == "impute_median" else series.mean()
                if pd.isna(stat):
                    continue
                group = _best_group(df, col, target_col, support()) if key == "impute_median" and params.get("group_by") == "auto" else None
                if group:
                    g, effect = group
                    keys = df[g].map(lambda v: None if pd.isna(v) else str(v))
                    sub = pd.DataFrame({"k": keys, "v": series}).dropna()
                    sizes = sub["k"].value_counts()
                    medians = {str(k): round(float(m), 4) for k, m in sub.groupby("k")["v"].median().items() if sizes[k] >= support()}
                    apply("fill_by_group",
                          f"Imputation: Filled {n_blank} blank(s) in '{col}' with the median per '{g}' "
                          f"({', '.join(f'{k}: {v}' for k, v in list(medians.items())[:6])}; others {round(float(stat), 3)})"
                          + (f" and flagged them in '{flag_name}'." if flag else "."),
                          column=col, group_column=g, values=medians, fallback=float(stat), add_flag=flag)
                    how = f"the median of their '{g}' group"
                else:
                    label = "Median" if key == "impute_median" else "Mean"
                    apply("fill_missing", f"Imputation: Replaced {n_blank} blank(s) in '{col}' with the {label} ({round(float(stat), 3)})"
                          + (f" and flagged them in '{flag_name}'." if flag else "."),
                          column=col, value=float(stat), add_flag=flag, flag_style="missing")
                    how = f"the {label.lower()} {round(float(stat), 3)}"
                fill_notes.append({"column": col, "count": n_blank, "how": how, "flag": flag_name})
            elif key == "impute_unknown" and not numeric:
                flag = bool(params.get("add_flag", True))
                apply("fill_missing", f"Imputation: Filled {n_blank} blank(s) in '{col}' with 'Unknown'"
                      + (f" & added indicator 'has_{col}'." if flag else " (kept as its own category)."),
                      column=col, value="Unknown", add_flag=flag)
                fill_notes.append({"column": col, "count": n_blank, "how": "the category 'Unknown'", "flag": f"has_{col}" if flag else None})
            else:
                mode = series.mode()
                if mode.empty:
                    continue
                value = _py(mode.iloc[0])
                # Numeric codes cannot hold an 'Unknown' category, so sizeable gaps get a flag instead.
                flag = numeric and n_blank >= support()
                flag_name = T.fresh_name(df, f"{col}_was_missing") if flag else None
                apply("fill_missing", f"Imputation: Replaced {n_blank} blank(s) in '{col}' with the most common value ({value!r})"
                      + (f" and flagged them in '{flag_name}'." if flag else "."),
                      column=col, value=value, add_flag=flag, flag_style="missing")
                fill_notes.append({"column": col, "count": n_blank, "how": f"the most common value {value!r}", "flag": flag_name})

    # 11. Rare category grouping: only values with enough rows are kept as categories
    for r in by_key.get("group_rare", []):
        col = r["column"]
        if not is_feature(col):
            continue
        keep_n = int((r.get("params") or {}).get("keep_top", 10))
        counts = df[col].dropna().astype(str).value_counts()
        keep = [str(v) for v, c in counts.items() if c >= support()][:keep_n]
        if not keep:
            apply("drop_columns", f"Encoding: Dropped '{col}': no value occurs in at least {support()} rows, so it cannot generalise.",
                  columns=[col])
            continue
        if len(keep) < len(counts):
            apply("group_rare", f"Encoding: Grouped categories in '{col}' into the {len(keep)} value(s) with at least {support()} rows + 'Other'.",
                  column=col, keep=keep)

    df_cleaned = df.copy()

    # 12. Encoding of remaining text features (one-hot over well-supported categories)
    encode_allowed = bool(by_key.get("one_hot_encode")) or "one_hot_encode" not in all_keys

    def encode(col: str) -> None:
        if df[col].isna().any():
            apply("fill_missing", f"Encoding: Filled {int(df[col].isna().sum())} blank(s) in '{col}' with 'Missing' before encoding.",
                  column=col, value="Missing")
        counts = df[col].astype(str).value_counts()
        sup = support()
        supported = [str(v) for v, c in counts.items() if c >= sup][:MAX_ONE_HOT_CATEGORIES]
        rest = int(counts.sum() - counts[supported].sum()) if supported else int(counts.sum())
        if not supported or (len(supported) == 1 and rest < sup):
            why = ("it has a single value" if len(counts) == 1 else
                   f"{len(counts)} distinct values and fewer than 2 groups of {sup}+ rows; it would only let a model memorise rows")
            apply("drop_columns", f"Encoding: Dropped '{col}': {why}.", columns=[col])
            return
        if rest:
            if rest >= sup:
                apply("group_rare", f"Encoding: Grouped {len(counts) - len(supported)} rare value(s) of '{col}' (each under {sup} rows) into 'Other'.",
                      column=col, keep=supported)
            else:
                apply("group_rare", f"Encoding: Folded {rest} row(s) with rare values of '{col}' into its most common value '{supported[0]}'.",
                      column=col, keep=supported, other_label=supported[0])
            counts = df[col].astype(str).value_counts()
        baseline = str(counts.index[0])
        indicators = sorted(c for c in counts.index.astype(str) if c != baseline)
        apply("one_hot", f"Encoding: One-hot encoded '{col}' ({len(indicators)} indicator(s); all zeros = '{baseline}').",
              column=col, categories=indicators)
        baselines.append({"column": col, "baseline": baseline, "indicators": indicators})

    if encode_allowed:
        for col in [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]:
            encode(col)

    # 13. Self-audit: repair whatever is still wrong in the output, then audit again
    repaired: List[Dict[str, Any]] = []
    for _ in range(MAX_AUDIT_ROUNDS):
        defects = [d for d in A.detect_defects(df, target_col, encode_allowed) if d["fix"]]
        if not defects:
            break
        to_cast = []
        for d in defects:
            col, fix = d["column"], d["fix"]
            if col not in df.columns:
                continue
            if fix == "encode":
                encode(col)
            elif fix == "drop":
                apply("drop_columns", f"Quality Audit: Dropped '{col}' - {d['detail']}", columns=[col])
            elif fix == "fill_numeric":
                values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if d["check"] == "infinite_values":
                    apply("replace_with_nan", f"Quality Audit: Treated infinite values in '{col}' as blank.", column=col, values=["inf", "-inf"])
                stat = _finite_or(values.median(), 0.0)
                n_blank = int(values.isna().sum())
                flag = n_blank >= support()
                apply("fill_missing", f"Quality Audit: Filled {n_blank} remaining blank(s) in '{col}' with the median ({round(stat, 3)})"
                      + (f" and flagged them in '{col}_was_missing'." if flag else "."),
                      column=col, value=stat, add_flag=flag, flag_style="missing")
                fill_notes.append({"column": col, "count": n_blank, "how": f"the median {round(stat, 3)}",
                                   "flag": f"{col}_was_missing" if flag else None})
            elif fix == "cast_int":
                to_cast.append(col)
                continue
            repaired.append(d)
        if to_cast:
            apply("cast_integer", f"Quality Audit: Stored {len(to_cast)} whole-number column(s) as integers ({', '.join(to_cast[:8])}"
                  + ("..." if len(to_cast) > 8 else "") + ").", columns=to_cast)
            repaired.extend(d for d in defects if d["fix"] == "cast_int")

    remaining = A.detect_defects(df, target_col, encode_allowed)
    audit = (
        [{**d, "status": "fixed"} for d in repaired]
        + [{**d, "status": "open"} for d in remaining]
        + [{**d, "status": "advice"} for d in A.advisories(df, target_col, problem_type, lineage, fill_notes, cap_notes, baselines)]
    )

    return PipelineResult(
        df_cleaned=df_cleaned,
        df_model_ready=df,
        applied_steps=steps,
        script_code=_build_script(ops),
        dropped_informative_columns=dropped_informative,
        audit=audit,
        data_dictionary=_data_dictionary(df, target_col, lineage, baselines),
    )


def _data_dictionary(df: pd.DataFrame, target_col: Optional[str], lineage: Dict[str, Dict[str, Any]],
                     baselines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_col = {b["column"]: b["baseline"] for b in baselines}
    rows = []
    for c in df.columns:
        s = df[c]
        info = lineage.get(c, {"source": "", "origin": "derived", "meaning": "", "steps": []})
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        nunique = int(s.nunique())
        if c == target_col:
            kind = "target"
        elif numeric and set(pd.unique(s.dropna())).issubset({0, 1}):
            kind = "binary flag (0/1)"
        elif numeric and pd.api.types.is_integer_dtype(s) and nunique <= 10:
            kind = "low-cardinality integer (ordinal level or count)"
        elif numeric:
            kind = "integer" if pd.api.types.is_integer_dtype(s) else "continuous number"
        else:
            kind = "text"
        meaning = info.get("meaning") or ("Original column" + (" (prediction target)" if c == target_col else ""))
        src = info.get("source")
        if src in baseline_by_col and info.get("origin") == "derived" and kind == "binary flag (0/1)":
            meaning += f"; all '{src}_*' indicators 0 = '{baseline_by_col[src]}'"
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "kind": kind,
            "source_column": src,
            "origin": info.get("origin"),
            "meaning": meaning,
            "transformations": info.get("steps", []),
            "missing": int(s.isna().sum()),
            "unique": nunique,
            "min": float(s.min()) if numeric and len(s.dropna()) else None,
            "max": float(s.max()) if numeric and len(s.dropna()) else None,
        })
    return rows
