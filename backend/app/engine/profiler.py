import re
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

ID_TOKENS = {"id", "uuid", "guid", "pk", "index", "idx", "key", "code", "number", "no"}
STRONG_ID_TOKENS = {"id", "uuid", "guid", "pk"}
BOOLEAN_STRINGS = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
DATE_NAME_TOKENS = {"date", "time", "timestamp", "datetime", "added", "created", "updated", "dob", "dt", "day", "month"}
YEAR_NAME_TOKENS = {"year", "yr", "yyyy"}
DATE_VALUE_PATTERN = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


def name_tokens(col_name: str) -> List[str]:
    """Splits a column name into lowercase word tokens (snake_case, kebab-case, camelCase, spaces)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col_name))
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def name_has_token(col_name: str, keywords) -> bool:
    return any(_singular(t) in keywords or t in keywords for t in name_tokens(col_name))


def infer_column_type(series: pd.Series, col_name: str) -> str:
    """
    Infers fine-grained semantic data type of a pandas Series:
    - id
    - boolean
    - year_range
    - datetime
    - unit_mixed_numeric (e.g. '90 min', '2 Seasons')
    - multilabel_text (e.g. 'Comedies, Dramas, International Movies')
    - person_name_or_text (e.g. 'Rajiv Chilaka', high cardinality entity names)
    - numeric
    - categorical (low-cardinality nominal)
    - text (free-form prose)
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return "text"

    col_lower = col_name.lower()
    tokens = name_tokens(col_name)
    n_unique = clean_series.nunique()
    all_unique = n_unique == len(clean_series) and len(clean_series) > 1

    # 1. Unique ID column (by name token, or an integer 1..N style sequence)
    if all_unique:
        if any(t in STRONG_ID_TOKENS for t in tokens) or (tokens and tokens[-1] in ID_TOKENS):
            return "id"
        if pd.api.types.is_integer_dtype(series) and len(clean_series) > 20:
            sorted_vals = np.sort(clean_series.to_numpy())
            if np.all(np.diff(sorted_vals) == 1):
                return "id"

    # 2. Boolean
    if series.dtype == bool:
        return "boolean"
    if n_unique <= 2:
        normalized = {str(v).strip().lower() for v in clean_series.unique()}
        normalized = {"1" if v == "1.0" else "0" if v == "0.0" else v for v in normalized}
        if normalized.issubset(BOOLEAN_STRINGS):
            return "boolean"

    # 3. Year columns (requires a year-like column name, e.g. release_year, YearBuilt)
    if "year" in col_lower or any(t in YEAR_NAME_TOKENS for t in tokens):
        num_clean = pd.to_numeric(clean_series, errors="coerce").dropna()
        if len(num_clean) > 5 and len(num_clean) >= 0.9 * len(clean_series):
            min_val, max_val = num_clean.min(), num_clean.max()
            if 1800 <= min_val <= 2100 and 1800 <= max_val <= 2100:
                return "year_range"

    # 4. Standard Numeric
    if pd.api.types.is_numeric_dtype(series):
        if clean_series.nunique() <= 5 and clean_series.nunique() < len(clean_series) * 0.05:
            return "categorical"
        return "numeric"

    # 5. Native Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # String Analysis on Non-Numeric Columns
    sample_strs = clean_series.astype(str).head(60).tolist()

    # 6. Check if Chronological Date String (e.g. "January 1, 2020", "2019-11-01", "01/05/2021")
    date_like_ratio = sum(bool(DATE_VALUE_PATTERN.search(s)) for s in sample_strs) / max(len(sample_strs), 1)
    if date_like_ratio >= 0.8 or (any(t in DATE_NAME_TOKENS for t in tokens) and date_like_ratio >= 0.5):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample_strs, format="mixed", errors="coerce")
                if parsed.notna().mean() >= 0.75:
                    return "datetime"
        except Exception:
            pass

    # 7. Check if Mixed Unit Numeric String (e.g. "90 min", "1 Season", "2 Seasons", "50 MB", "$120")
    unit_mixed_pattern = re.compile(r"^\s*\$?\d+(\.\d+)?\s*[a-zA-Z%]+\s*$")
    matching_units = sum(bool(unit_mixed_pattern.match(s)) for s in sample_strs)
    if len(sample_strs) > 0 and (matching_units / len(sample_strs)) >= 0.70:
        return "unit_mixed_numeric"

    # 8. Check if Delimited Multi-Label List (e.g. "Action, Drama, Thriller" or "Tom Hanks, Tim Allen")
    delimiters_found = sum(bool(',' in s or ';' in s or '|' in s) for s in sample_strs)
    multilabel_names = {"cast", "genre", "listed", "tag", "categorie", "category", "keyword", "actor", "skill", "label"}
    if len(sample_strs) > 0 and (delimiters_found / len(sample_strs)) >= 0.40 and name_has_token(col_name, multilabel_names):
        return "multilabel_text"

    # 9. Check if Person Name / Entity Free-Text (e.g. Director, Author, Title, City)
    unique_count = n_unique
    unique_ratio = unique_count / len(clean_series) if len(clean_series) > 0 else 1.0

    person_names = {"director", "actor", "author", "artist", "creator", "writer", "name", "title", "user", "person", "username"}
    if name_has_token(col_name, person_names):
        if unique_count > 15:
            return "person_name_or_text"

    # 10. Low Cardinality Nominal Categorical vs Free Text
    if unique_count <= 25 or unique_ratio < 0.05:
        return "categorical"

    return "text"


def identifier_columns(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns if infer_column_type(df[c], str(c)) == "id"]


def duplicate_subset(df: pd.DataFrame) -> List[str]:
    """Columns used for duplicate detection: everything except unique identifiers."""
    ids = set(identifier_columns(df))
    subset = [str(c) for c in df.columns if str(c) not in ids]
    return subset or [str(c) for c in df.columns]


def count_duplicate_records(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.duplicated(subset=duplicate_subset(df)).sum())


def profile_dataset(df: pd.DataFrame, target_col: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Profiles a pandas DataFrame, computing column-level and dataset-level statistics.
    Returns (summary_stats_dict, list_of_column_profiles).
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    
    duplicate_rows = count_duplicate_records(df)
    duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    type_counts = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "boolean": 0,
        "id": 0,
        "text": 0,
        "year_range": 0,
        "unit_mixed_numeric": 0,
        "multilabel_text": 0,
        "person_name_or_text": 0,
    }
    column_profiles = []

    for col in df.columns:
        series = df[col]
        col_type = infer_column_type(series, str(col))
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

        missing_cnt = int(series.isna().sum())
        missing_pct = round((missing_cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_cnt = int(series.nunique())
        unique_pct = round((unique_cnt / total_rows * 100), 2) if total_rows > 0 else 0.0

        col_profile = {
            "name": str(col),
            "inferred_type": col_type,
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "unique_count": unique_cnt,
            "unique_pct": unique_pct,
            "is_target": (str(col) == target_col),
            "numeric_stats": None,
            "categorical_stats": None,
        }

        if col_type in ["numeric", "year_range"]:
            clean_num = pd.to_numeric(series, errors="coerce").dropna()
            if not clean_num.empty:
                q25 = float(clean_num.quantile(0.25))
                q75 = float(clean_num.quantile(0.75))
                iqr = float(q75 - q25)
                skew = float(clean_num.skew()) if len(clean_num) > 2 else 0.0
                kurt = float(clean_num.kurt()) if len(clean_num) > 3 else 0.0
                
                col_profile["numeric_stats"] = {
                    "min": float(clean_num.min()),
                    "max": float(clean_num.max()),
                    "mean": round(float(clean_num.mean()), 3),
                    "median": round(float(clean_num.median()), 3),
                    "std": round(float(clean_num.std()), 3) if len(clean_num) > 1 else 0.0,
                    "skewness": round(skew, 3) if not np.isnan(skew) else 0.0,
                    "kurtosis": round(kurt, 3) if not np.isnan(kurt) else 0.0,
                    "iqr": round(iqr, 3),
                    "q25": round(q25, 3),
                    "q75": round(q75, 3),
                }

        elif col_type in ["categorical", "boolean", "person_name_or_text", "unit_mixed_numeric", "multilabel_text"]:
            val_counts = series.value_counts(dropna=False).head(10)
            top_cats = []
            for val, cnt in val_counts.items():
                val_str = "null" if pd.isna(val) else str(val)
                pct = round(cnt / total_rows * 100, 1) if total_rows > 0 else 0.0
                top_cats.append({"value": val_str, "count": int(cnt), "percentage": pct})
            col_profile["categorical_stats"] = {
                "top_categories": top_cats,
                "cardinality": unique_cnt,
            }

        column_profiles.append(col_profile)

    summary_stats = {
        "row_count": total_rows,
        "col_count": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "overall_missing_pct": overall_missing_pct,
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_pct": duplicate_pct,
        "memory_usage_mb": memory_mb,
        "type_breakdown": type_counts,
    }

    return summary_stats, column_profiles
