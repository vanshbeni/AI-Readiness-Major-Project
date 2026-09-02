import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


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

    # 1. Unique ID column
    if any(k in col_lower for k in ["id", "uuid", "guid", "index", "pk"]) and clean_series.nunique() == len(clean_series):
        return "id"

    # 2. Boolean
    if series.dtype == bool or set(clean_series.unique()).issubset({0, 1, "0", "1", "true", "false", "True", "False", "yes", "no"}):
        if clean_series.nunique() <= 2:
            return "boolean"

    # 3. Year Range Check (e.g. release_year, birth_year)
    if pd.api.types.is_numeric_dtype(series) or "year" in col_lower:
        num_clean = pd.to_numeric(clean_series, errors="coerce").dropna()
        if not num_clean.empty and len(num_clean) > 5:
            min_val = num_clean.min()
            max_val = num_clean.max()
            if 1800 <= min_val <= 2050 and 1800 <= max_val <= 2050 and (max_val - min_val) < 150:
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
    if any(k in col_lower for k in ["date", "added", "created", "timestamp", "time", "dob"]) or len(sample_strs) > 5:
        try:
            import warnings
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
    if len(sample_strs) > 0 and (delimiters_found / len(sample_strs)) >= 0.40 and any(k in col_lower for k in ["cast", "genre", "listed_in", "tags", "categories", "keywords", "actors"]):
        return "multilabel_text"

    # 9. Check if Person Name / Entity Free-Text (e.g. Director, Author, Title, City)
    unique_count = clean_series.nunique()
    unique_ratio = unique_count / len(clean_series) if len(clean_series) > 0 else 1.0

    if any(k in col_lower for k in ["director", "actor", "author", "artist", "creator", "writer", "name", "title", "user", "person"]):
        if unique_count > 15:
            return "person_name_or_text"

    # 10. Low Cardinality Nominal Categorical vs Free Text
    if unique_count <= 25 or unique_ratio < 0.05:
        return "categorical"

    return "text"


def profile_dataset(df: pd.DataFrame, target_col: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Profiles a pandas DataFrame, computing column-level and dataset-level statistics.
    Returns (summary_stats_dict, list_of_column_profiles).
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    
    duplicate_rows = int(df.duplicated().sum())
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
