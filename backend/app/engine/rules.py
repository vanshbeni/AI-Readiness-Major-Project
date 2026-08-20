from typing import List, Dict, Any


def select_remediation_methods(
    issues: List[Dict[str, Any]],
    column_profiles: List[Dict[str, Any]],
    problem_type: str = "classification",
) -> List[Dict[str, Any]]:
    """
    Stage 2: Deterministic Rule Matrix (Domain-Aware Data Science Rules)
    Maps detected data issues and semantic characteristics to precise remediation methods.
    """
    col_profile_map = {cp["name"]: cp for cp in column_profiles}
    recommendations = []

    for issue in issues:
        issue_type = issue["issue_type"]
        col_name = issue.get("column")
        details = issue.get("details", {})
        col_profile = col_profile_map.get(col_name) if col_name else None
        inferred_type = col_profile.get("inferred_type", "text") if col_profile else "text"

        if issue_type == "duplicate_rows":
            dup_cnt = details.get("duplicate_count", 0)
            recommendations.append({
                "issue_type": issue_type,
                "column": None,
                "method": "Drop Exact Duplicate Rows",
                "reason_title": f"Remove {dup_cnt} duplicate rows",
                "severity": issue["severity"],
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "duplicate_count": dup_cnt,
                    "duplicate_pct": details.get("duplicate_pct", 0.0),
                },
            })

        elif issue_type == "missing_values":
            missing_pct = details.get("missing_pct", 0.0)
            cardinality = details.get("cardinality", col_profile.get("unique_count", 0) if col_profile else 0)

            if missing_pct > 70.0:
                method = "Drop High-Missingness Column (>70%)"
                reason_title = f"Drop '{col_name}' ({missing_pct}% missing)"
                is_destructive = True
                is_approved = False
            elif inferred_type in ["person_name_or_text", "text"] or cardinality > 20:
                # High-cardinality entity / text (e.g. director, cast, author, notes)
                method = "Impute with 'Unknown' & Binary Presence Flag"
                reason_title = f"Impute '{col_name}' with 'Unknown' (Preserves Entity Semantics)"
                is_destructive = False
                is_approved = True
            elif inferred_type in ["numeric", "year_range"]:
                num_stats = col_profile.get("numeric_stats") if col_profile else None
                skewness = num_stats.get("skewness", 0.0) if num_stats else 0.0
                if abs(skewness) > 1.0:
                    method = "Median Imputation"
                    reason_title = f"Impute '{col_name}' using Median (Skewness: {skewness})"
                else:
                    method = "Mean Imputation"
                    reason_title = f"Impute '{col_name}' using Mean (Normal Distribution)"
                is_destructive = False
                is_approved = True
            else:
                # Low-cardinality nominal category with low missingness (e.g. rating, gender, status)
                method = "Mode (Most Frequent) Imputation"
                reason_title = f"Impute '{col_name}' with Mode (Low-Cardinality Category)"
                is_destructive = False
                is_approved = True

            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": method,
                "reason_title": reason_title,
                "severity": issue["severity"],
                "is_destructive": is_destructive,
                "is_approved": is_approved,
                "stats_context": {
                    "column": col_name,
                    "missing_pct": missing_pct,
                    "inferred_type": inferred_type,
                    "cardinality": cardinality,
                    "skewness": col_profile.get("numeric_stats", {}).get("skewness", 0.0) if col_profile and col_profile.get("numeric_stats") else 0.0,
                },
            })

        elif issue_type == "unparsed_datetime_feature":
            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "Parse Datetime & Extract Temporal Signals (Year, Month, Day)",
                "reason_title": f"Engineer Temporal Features from '{col_name}'",
                "severity": "medium",
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "column": col_name,
                    "sample_values": details.get("sample_values", []),
                },
            })

        elif issue_type == "unit_mixed_feature":
            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "Split into Numeric Value & Categorical Unit Columns",
                "reason_title": f"Disentangle '{col_name}' into Value & Unit (e.g. runtime vs seasons)",
                "severity": "high",
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "column": col_name,
                    "sample_values": details.get("sample_values", []),
                },
            })

        elif issue_type == "multilabel_delimited_text":
            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "Multi-Hot Binary Token Encoding (Delimited List)",
                "reason_title": f"Tokenize & Multi-Hot Encode individual categories in '{col_name}'",
                "severity": "medium",
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "column": col_name,
                    "sample_values": details.get("sample_values", []),
                },
            })

        elif issue_type == "invalid_domain_values":
            neg_cnt = details.get("negative_count", 0)
            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "Clip Negative Values to Zero (0.0)",
                "reason_title": f"Clip {neg_cnt} invalid negative values in '{col_name}'",
                "severity": "high",
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "column": col_name,
                    "negative_count": neg_cnt,
                    "min_value": details.get("min_value", 0.0),
                },
            })

        elif issue_type == "statistical_outliers":
            outlier_pct = details.get("outlier_pct", 0.0)
            skew = details.get("skewness", 0.0)
            lower_b = details.get("lower_bound", 0.0)
            upper_b = details.get("upper_bound", 0.0)

            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "IQR Winsorization / Capping (1.5x IQR)",
                "reason_title": f"Cap outliers in '{col_name}' between [{lower_b}, {upper_b}]",
                "severity": issue["severity"],
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "column": col_name,
                    "outlier_count": details.get("outlier_count", 0),
                    "outlier_pct": outlier_pct,
                    "lower_bound": lower_b,
                    "upper_bound": upper_b,
                    "skewness": skew,
                },
            })

        elif issue_type == "multicollinear_features":
            col1 = details.get("col1")
            col2 = details.get("col2")
            corr = details.get("correlation", 0.0)
            drop_col = col2
            recommendations.append({
                "issue_type": issue_type,
                "column": drop_col,
                "method": f"Drop Redundant Collinear Feature '{drop_col}'",
                "reason_title": f"Drop '{drop_col}' (Correlation {corr} with '{col1}')",
                "severity": "medium",
                "is_destructive": True,
                "is_approved": False,
                "stats_context": {
                    "col1": col1,
                    "col2": col2,
                    "correlation": corr,
                    "drop_candidate": drop_col,
                },
            })

        elif issue_type == "constant_feature":
            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": "Drop Zero-Variance Constant Column",
                "reason_title": f"Drop '{col_name}' (Zero Variance)",
                "severity": "medium",
                "is_destructive": True,
                "is_approved": True,
                "stats_context": {"column": col_name},
            })

        elif issue_type == "class_imbalance":
            ratio_disp = details.get("ratio_display", "")
            maj_ratio = details.get("majority_ratio", 0.5)
            if maj_ratio > 0.85:
                method = "SMOTE Oversampling + Balanced Class Weights"
                reason_title = f"Apply SMOTE + Balanced Weights ({ratio_disp})"
            else:
                method = "Balanced Class Weighting (Cost-Sensitive)"
                reason_title = f"Apply Cost-Sensitive Class Weights ({ratio_disp})"

            recommendations.append({
                "issue_type": issue_type,
                "column": col_name,
                "method": method,
                "reason_title": reason_title,
                "severity": issue["severity"],
                "is_destructive": False,
                "is_approved": True,
                "stats_context": {
                    "target_column": col_name,
                    "majority_class": details.get("majority_class"),
                    "majority_ratio": maj_ratio,
                    "ratio_display": ratio_disp,
                },
            })

    return recommendations
