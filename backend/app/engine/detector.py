import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.engine.profiler import infer_column_type

NON_NEGATIVE_KEYWORDS = [
    "age", "salary", "income", "price", "cost", "revenue", "count",
    "quantity", "amount", "hours", "fare", "rate", "distance", "tenure"
]


def detect_issues(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scans the DataFrame for data quality issues and semantic feature engineering opportunities.
    Returns a list of structured issue dictionaries.
    """
    issues = []
    total_rows, total_cols = df.shape
    if total_rows == 0:
        return issues

    # 1. Exact Duplicate Rows Detection
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows * 100), 2)
        severity = "high" if dup_pct > 5.0 else ("medium" if dup_pct > 1.0 else "low")
        issues.append({
            "issue_type": "duplicate_rows",
            "column": None,
            "severity": severity,
            "title": f"{dup_count} Duplicate Rows ({dup_pct}%)",
            "details": {
                "duplicate_count": dup_count,
                "duplicate_pct": dup_pct,
                "total_rows": total_rows,
            },
        })

    # Column-by-column semantic scanning
    for col in df.columns:
        series = df[col]
        col_type = infer_column_type(series, str(col))
        col_lower = str(col).lower()

        # 2. Missing Values Detection
        missing_cnt = int(series.isna().sum())
        if missing_cnt > 0:
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
                "column": str(col),
                "severity": severity,
                "title": f"Missing Values in '{col}' ({missing_pct}%)",
                "details": {
                    "column": str(col),
                    "missing_count": missing_cnt,
                    "missing_pct": missing_pct,
                    "inferred_type": col_type,
                    "cardinality": int(series.nunique()),
                },
            })

        # 3. Unparsed Datetime Strings
        if col_type == "datetime" and series.dtype == object and str(col) != target_col:
            issues.append({
                "issue_type": "unparsed_datetime_feature",
                "column": str(col),
                "severity": "medium",
                "title": f"Raw Date/Timestamp String in '{col}'",
                "details": {
                    "column": str(col),
                    "sample_values": series.dropna().head(3).tolist(),
                },
            })

        # 4. Mixed Unit Strings (e.g. Duration: '90 min' vs '2 Seasons')
        if col_type == "unit_mixed_numeric" and str(col) != target_col:
            issues.append({
                "issue_type": "unit_mixed_feature",
                "column": str(col),
                "severity": "high",
                "title": f"Mixed Value & Units in '{col}' (e.g. '90 min', '1 Season')",
                "details": {
                    "column": str(col),
                    "sample_values": series.dropna().head(4).tolist(),
                },
            })

        # 5. Delimited Multi-Label Lists (e.g. 'Comedies, Dramas, International Movies')
        if col_type == "multilabel_text" and str(col) != target_col:
            issues.append({
                "issue_type": "multilabel_delimited_text",
                "column": str(col),
                "severity": "medium",
                "title": f"Delimited Multi-Label List in '{col}' (e.g. genres, tags)",
                "details": {
                    "column": str(col),
                    "sample_values": series.dropna().head(3).tolist(),
                },
            })

        # 6. Domain validity (negative values where non-negative expected)
        if pd.api.types.is_numeric_dtype(series):
            clean_num = series.dropna()
            if any(kw in col_lower for kw in NON_NEGATIVE_KEYWORDS):
                neg_count = int((clean_num < 0).sum())
                if neg_count > 0:
                    neg_pct = round(neg_count / len(clean_num) * 100, 2)
                    issues.append({
                        "issue_type": "invalid_domain_values",
                        "column": str(col),
                        "severity": "high",
                        "title": f"Invalid Negative Values in '{col}' ({neg_count} rows)",
                        "details": {
                            "column": str(col),
                            "negative_count": neg_count,
                            "negative_pct": neg_pct,
                            "min_value": float(clean_num.min()),
                        },
                    })

            # Statistical Outliers (DO NOT FLAG year_range columns like release_year)
            if col_type != "year_range" and "year" not in col_lower and len(clean_num) > 10:
                q25 = clean_num.quantile(0.25)
                q75 = clean_num.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - (1.5 * iqr)
                    upper_bound = q75 + (1.5 * iqr)
                    outliers = clean_num[(clean_num < lower_bound) | (clean_num > upper_bound)]
                    outlier_cnt = int(len(outliers))
                    if outlier_cnt > 0:
                        outlier_pct = round(outlier_cnt / len(clean_num) * 100, 2)
                        if outlier_pct > 1.0 and str(col) != target_col:
                            severity = "high" if outlier_pct > 10.0 else ("medium" if outlier_pct > 3.0 else "low")
                            issues.append({
                                "issue_type": "statistical_outliers",
                                "column": str(col),
                                "severity": severity,
                                "title": f"Outliers in '{col}' ({outlier_cnt} rows, {outlier_pct}%)",
                                "details": {
                                    "column": str(col),
                                    "outlier_count": outlier_cnt,
                                    "outlier_pct": outlier_pct,
                                    "lower_bound": round(float(lower_bound), 3),
                                    "upper_bound": round(float(upper_bound), 3),
                                    "skewness": round(float(clean_num.skew()), 3) if len(clean_num) > 2 else 0.0,
                                },
                            })

            # Constant or near-constant column
            if clean_num.nunique() <= 1 and str(col) != target_col:
                issues.append({
                    "issue_type": "constant_feature",
                    "column": str(col),
                    "severity": "medium",
                    "title": f"Constant / Zero Variance Feature '{col}'",
                    "details": {"column": str(col), "unique_values": int(clean_num.nunique())},
                })

    # 7. Multicollinearity / High Correlation
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if infer_column_type(df[c], str(c)) != "year_range"]
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    if col1 == target_col or col2 == target_col:
                        continue
                    r_val = float(corr_matrix.loc[col1, col2])
                    if not np.isnan(r_val) and r_val > 0.85:
                        high_corr_pairs.append({"col1": col1, "col2": col2, "correlation": round(r_val, 3)})

            if high_corr_pairs:
                for pair in high_corr_pairs[:5]:
                    issues.append({
                        "issue_type": "multicollinear_features",
                        "column": f"{pair['col1']} & {pair['col2']}",
                        "severity": "medium",
                        "title": f"High Correlation ({pair['correlation']}) between '{pair['col1']}' and '{pair['col2']}'",
                        "details": pair,
                    })
        except Exception:
            pass

    # 8. Target Class Imbalance (Classification)
    if target_col and target_col in df.columns and problem_type == "classification":
        target_series = df[target_col].dropna()
        val_counts = target_series.value_counts(normalize=True)
        if len(val_counts) >= 2:
            majority_ratio = float(val_counts.iloc[0])
            minority_ratio = float(val_counts.iloc[-1])
            ratio_str = f"{round(majority_ratio * 100, 1)}% vs {round(minority_ratio * 100, 1)}%"
            if majority_ratio > 0.70:
                severity = "critical" if majority_ratio > 0.90 else "high"
                issues.append({
                    "issue_type": "class_imbalance",
                    "column": target_col,
                    "severity": severity,
                    "title": f"Target Class Imbalance on '{target_col}' ({ratio_str})",
                    "details": {
                        "column": target_col,
                        "majority_class": str(val_counts.index[0]),
                        "majority_ratio": round(majority_ratio, 3),
                        "minority_class": str(val_counts.index[-1]),
                        "minority_ratio": round(minority_ratio, 3),
                        "ratio_display": ratio_str,
                    },
                })

    return issues
