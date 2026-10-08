import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.engine.profiler import infer_column_type, name_has_token, count_duplicate_records

NON_NEGATIVE_KEYWORDS = {
    "age", "salary", "income", "price", "cost", "revenue", "count", "quantity", "qty",
    "amount", "hour", "fare", "distance", "tenure", "charge", "fee", "weight", "height",
    "duration", "population", "area", "sqft", "room", "bedroom", "bathroom",
}

TRANSFORMED_TYPES = {"datetime", "unit_mixed_numeric", "multilabel_text"}
TEXT_TYPES = {"categorical", "text", "person_name_or_text"}
HIGH_CARDINALITY_LIMIT = 50
RARE_GROUPING_LIMIT = 10


def is_high_cardinality(nunique: int, total_rows: int) -> bool:
    return nunique > HIGH_CARDINALITY_LIMIT or (total_rows > 50 and nunique / total_rows > 0.40)


def detect_issues(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scans the DataFrame for data quality issues and semantic feature engineering opportunities.
    Returns a list of structured issue dictionaries.
    """
    issues: List[Dict[str, Any]] = []
    total_rows, _ = df.shape
    if total_rows == 0:
        return issues

    col_types = {str(c): infer_column_type(df[c], str(c)) for c in df.columns}

    # 1. Duplicate records (ignoring unique identifier columns)
    dup_count = count_duplicate_records(df)
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows * 100), 2)
        severity = "high" if dup_pct > 5.0 else ("medium" if dup_pct > 1.0 else "low")
        issues.append({
            "issue_type": "duplicate_rows",
            "column": None,
            "severity": severity,
            "title": f"{dup_count} Duplicate Rows ({dup_pct}%)",
            "details": {"duplicate_count": dup_count, "duplicate_pct": dup_pct, "total_rows": total_rows},
        })

    # 2. Missing target labels
    if target_col and target_col in df.columns:
        missing_target = int(df[target_col].isna().sum())
        if missing_target > 0:
            pct = round(missing_target / total_rows * 100, 2)
            issues.append({
                "issue_type": "missing_target",
                "column": target_col,
                "severity": "critical" if pct > 20 else "high",
                "title": f"{missing_target} Rows Missing the Target '{target_col}' ({pct}%)",
                "details": {"column": target_col, "missing_count": missing_target, "missing_pct": pct},
            })

    categorical_to_encode: List[str] = []

    for col in df.columns:
        col = str(col)
        series = df[col]
        col_type = col_types[col]
        is_target = col == target_col
        clean = series.dropna()
        nunique = int(clean.nunique())

        # 3. Identifier columns carry no predictive signal
        if col_type == "id" and not is_target:
            issues.append({
                "issue_type": "identifier_column",
                "column": col,
                "severity": "low",
                "title": f"Unique Identifier Column '{col}'",
                "details": {"column": col, "unique_count": nunique},
            })
            continue

        # 4. Missing values (features only; target handled above)
        missing_cnt = int(series.isna().sum())
        if missing_cnt > 0 and not is_target:
            missing_pct = round((missing_cnt / total_rows * 100), 2)
            if missing_pct > 50.0:
                severity = "critical"
            elif missing_pct > 20.0:
                severity = "high"
            elif missing_pct > 5.0:
                severity = "medium"
            else:
                severity = "low"
            issues.append({
                "issue_type": "missing_values",
                "column": col,
                "severity": severity,
                "title": f"Missing Values in '{col}' ({missing_pct}%)",
                "details": {
                    "column": col,
                    "missing_count": missing_cnt,
                    "missing_pct": missing_pct,
                    "inferred_type": col_type,
                    "cardinality": nunique,
                },
            })

        if is_target:
            continue

        # 5. Constant / zero-variance features (any dtype)
        if nunique == 1:
            issues.append({
                "issue_type": "constant_feature",
                "column": col,
                "severity": "medium",
                "title": f"Constant / Zero Variance Feature '{col}'",
                "details": {"column": col, "unique_values": nunique},
            })
            continue

        # 6. Unparsed datetime strings
        if col_type == "datetime" and series.dtype == object:
            issues.append({
                "issue_type": "unparsed_datetime_feature",
                "column": col,
                "severity": "medium",
                "title": f"Raw Date/Timestamp String in '{col}'",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(3).tolist()]},
            })

        # 7. Mixed unit strings ('90 min', '2 Seasons')
        if col_type == "unit_mixed_numeric":
            issues.append({
                "issue_type": "unit_mixed_feature",
                "column": col,
                "severity": "high",
                "title": f"Mixed Value & Units in '{col}' (e.g. '90 min', '1 Season')",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(4).tolist()]},
            })

        # 8. Delimited multi-label lists
        if col_type == "multilabel_text":
            issues.append({
                "issue_type": "multilabel_delimited_text",
                "column": col,
                "severity": "medium",
                "title": f"Delimited Multi-Label List in '{col}' (e.g. genres, tags)",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(3).tolist()]},
            })

        # 9. Text / categorical cardinality handling
        if col_type in TEXT_TYPES and not pd.api.types.is_numeric_dtype(series):
            if is_high_cardinality(nunique, total_rows):
                issues.append({
                    "issue_type": "high_cardinality_text",
                    "column": col,
                    "severity": "low",
                    "title": f"High-Cardinality Free Text in '{col}' ({nunique} unique values)",
                    "details": {"column": col, "unique_count": nunique, "unique_pct": round(nunique / total_rows * 100, 2)},
                })
                continue
            if nunique > RARE_GROUPING_LIMIT:
                issues.append({
                    "issue_type": "rare_categories",
                    "column": col,
                    "severity": "low",
                    "title": f"Many Rare Categories in '{col}' ({nunique} unique values)",
                    "details": {"column": col, "unique_count": nunique, "keep_top": RARE_GROUPING_LIMIT},
                })
            categorical_to_encode.append(col)
        elif col_type == "boolean" and not pd.api.types.is_numeric_dtype(series):
            categorical_to_encode.append(col)

        if col_type in TRANSFORMED_TYPES or not pd.api.types.is_numeric_dtype(series) or col_type == "boolean":
            continue

        # 10. Domain validity (negative values where non-negative expected)
        if name_has_token(col, NON_NEGATIVE_KEYWORDS):
            neg_count = int((clean < 0).sum())
            if neg_count > 0:
                issues.append({
                    "issue_type": "invalid_domain_values",
                    "column": col,
                    "severity": "high",
                    "title": f"Invalid Negative Values in '{col}' ({neg_count} rows)",
                    "details": {
                        "column": col,
                        "negative_count": neg_count,
                        "negative_pct": round(neg_count / len(clean) * 100, 2),
                        "min_value": float(clean.min()),
                    },
                })

        # 11. Statistical outliers (not for year columns or low-cardinality codes)
        if col_type == "numeric" and len(clean) > 10:
            q25, q75 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outlier_cnt = int(((clean < lower_bound) | (clean > upper_bound)).sum())
                outlier_pct = round(outlier_cnt / len(clean) * 100, 2)
                if outlier_cnt > 0 and outlier_pct > 1.0:
                    severity = "high" if outlier_pct > 10.0 else ("medium" if outlier_pct > 3.0 else "low")
                    issues.append({
                        "issue_type": "statistical_outliers",
                        "column": col,
                        "severity": severity,
                        "title": f"Outliers in '{col}' ({outlier_cnt} rows, {outlier_pct}%)",
                        "details": {
                            "column": col,
                            "outlier_count": outlier_cnt,
                            "outlier_pct": outlier_pct,
                            "lower_bound": round(float(lower_bound), 3),
                            "upper_bound": round(float(upper_bound), 3),
                            "skewness": round(float(clean.skew()), 3) if len(clean) > 2 else 0.0,
                        },
                    })

    # 12. Categorical encoding requirement
    if categorical_to_encode:
        issues.append({
            "issue_type": "categorical_encoding",
            "column": None,
            "severity": "info",
            "title": f"{len(categorical_to_encode)} Categorical Feature(s) Need Numeric Encoding",
            "details": {"columns": categorical_to_encode},
        })

    # 13. Multicollinearity among numeric features
    numeric_cols = [
        str(c) for c in df.select_dtypes(include=[np.number]).columns
        if col_types[str(c)] in ("numeric", "categorical") and str(c) != target_col and df[c].nunique() > 1
    ]
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                r_val = float(corr_matrix.loc[col1, col2])
                if not np.isnan(r_val) and r_val > 0.85:
                    high_corr_pairs.append({"col1": col1, "col2": col2, "correlation": round(r_val, 3)})
        high_corr_pairs.sort(key=lambda p: p["correlation"], reverse=True)
        already_dropped = set()
        for pair in high_corr_pairs[:5]:
            if pair["col1"] in already_dropped or pair["col2"] in already_dropped:
                continue
            already_dropped.add(pair["col2"])
            issues.append({
                "issue_type": "multicollinear_features",
                "column": f"{pair['col1']} & {pair['col2']}",
                "severity": "medium",
                "title": f"High Correlation ({pair['correlation']}) between '{pair['col1']}' and '{pair['col2']}'",
                "details": pair,
            })

    # 14. Target class imbalance (binary and multi-class)
    if target_col and target_col in df.columns and problem_type == "classification":
        counts = df[target_col].dropna().value_counts()
        if len(counts) >= 2:
            majority_ratio = float(counts.iloc[0] / counts.sum())
            minority_ratio = float(counts.iloc[-1] / counts.sum())
            imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])
            ratio_str = f"{round(majority_ratio * 100, 1)}% vs {round(minority_ratio * 100, 1)}%"
            if imbalance_ratio >= 3.0:
                if majority_ratio > 0.90 or imbalance_ratio >= 10:
                    severity = "critical"
                elif imbalance_ratio >= 5:
                    severity = "high"
                else:
                    severity = "medium"
                issues.append({
                    "issue_type": "class_imbalance",
                    "column": target_col,
                    "severity": severity,
                    "title": f"Target Class Imbalance on '{target_col}' ({ratio_str})",
                    "details": {
                        "column": target_col,
                        "majority_class": str(counts.index[0]),
                        "majority_ratio": round(majority_ratio, 3),
                        "minority_class": str(counts.index[-1]),
                        "minority_ratio": round(minority_ratio, 3),
                        "imbalance_ratio": round(imbalance_ratio, 2),
                        "n_classes": int(len(counts)),
                        "ratio_display": ratio_str,
                    },
                })

    return issues
