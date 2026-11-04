from typing import List, Dict, Any

NUMERIC_TYPES = {"numeric", "year_range"}
CATEGORY_TYPES = {"categorical", "boolean"}
FLAG_MISSING_PCT = 5.0
INDICATOR_ONLY_PCT = 70.0


def _rec(issue: Dict[str, Any], method_key: str, method: str, reason_title: str, *,
         column=None, is_destructive=False, is_approved=True, severity=None, **context) -> Dict[str, Any]:
    return {
        "issue_type": issue["issue_type"],
        "column": column if column is not None else issue.get("column"),
        "method": method,
        "reason_title": reason_title,
        "severity": severity or issue["severity"],
        "is_destructive": is_destructive,
        "is_approved": is_approved,
        "stats_context": {"method_key": method_key, **context},
    }


def select_remediation_methods(
    issues: List[Dict[str, Any]],
    column_profiles: List[Dict[str, Any]],
    problem_type: str = "classification",
) -> List[Dict[str, Any]]:
    """
    Stage 2: Deterministic Rule Matrix.
    Maps detected issues to remediation methods. `stats_context.method_key` is the machine-readable
    instruction the executor follows; `method` is the human-readable label.
    """
    col_profile_map = {cp["name"]: cp for cp in column_profiles}
    recommendations: List[Dict[str, Any]] = []

    for issue in issues:
        issue_type = issue["issue_type"]
        col_name = issue.get("column")
        details = issue.get("details", {})
        col_profile = col_profile_map.get(col_name) if col_name else None
        inferred_type = col_profile.get("inferred_type", "text") if col_profile else details.get("inferred_type", "text")

        if issue_type == "duplicate_rows":
            cnt = details.get("duplicate_count", 0)
            recommendations.append(_rec(
                issue, "drop_duplicates", "Drop Duplicate Records", f"Remove {cnt} duplicate rows",
                column=None, duplicate_count=cnt, duplicate_pct=details.get("duplicate_pct", 0.0),
            ))

        elif issue_type == "missing_target":
            cnt = details.get("missing_count", 0)
            recommendations.append(_rec(
                issue, "drop_missing_target", "Drop Rows with Missing Target",
                f"Drop {cnt} rows without a '{col_name}' label",
                is_destructive=True, missing_count=cnt, missing_pct=details.get("missing_pct", 0.0), column_name=col_name,
            ))

        elif issue_type == "identifier_column":
            recommendations.append(_rec(
                issue, "drop_identifier", "Drop Identifier Column", f"Drop ID column '{col_name}'",
                is_destructive=True, unique_count=details.get("unique_count", 0), column_name=col_name,
            ))

        elif issue_type == "missing_values":
            missing_pct = details.get("missing_pct", 0.0)
            cardinality = details.get("cardinality", 0)
            num_stats = (col_profile or {}).get("numeric_stats") or {}
            skewness = num_stats.get("skewness", 0.0)
            ctx = dict(column_name=col_name, missing_pct=missing_pct, inferred_type=inferred_type,
                       cardinality=cardinality, skewness=skewness)

            # Whether a value was missing is often informative itself, so any gap large enough to learn from
            # (min_support rows) keeps a 0/1 record; otherwise filled values would pose as real measurements.
            missing_cnt = int(details.get("missing_count", 0))
            total = int(round(missing_cnt * 100 / missing_pct)) if missing_pct else 0
            flag = missing_pct >= FLAG_MISSING_PCT or missing_cnt >= max(5, -(-total // 100))
            if missing_pct > INDICATOR_ONLY_PCT:
                recommendations.append(_rec(
                    issue, "missing_indicator", f"Replace with Presence Indicator 'has_{col_name}'",
                    f"Keep only whether '{col_name}' was recorded ({missing_pct}% missing is too sparse to fill)", **ctx,
                ))
            elif inferred_type in NUMERIC_TYPES:
                recommendations.append(_rec(
                    issue, "impute_median", "Group-Aware Median Imputation" + (" + Missing Flag" if flag else ""),
                    f"Impute '{col_name}' with the median (per related group when one exists)"
                    + (f" and add '{col_name}_was_missing'" if flag else ""),
                    add_flag=flag, group_by="auto", **ctx,
                ))
            elif inferred_type in CATEGORY_TYPES and not flag:
                recommendations.append(_rec(
                    issue, "impute_mode", "Mode (Most Frequent) Imputation",
                    f"Impute '{col_name}' with Mode ({missing_pct}% missing, low-cardinality category)", **ctx,
                ))
            else:
                text_like = inferred_type not in CATEGORY_TYPES
                recommendations.append(_rec(
                    issue, "impute_unknown", "Impute with 'Unknown' Category" + (" & Presence Flag" if text_like else ""),
                    f"Fill '{col_name}' with an explicit 'Unknown' category ({missing_pct}% missing)", add_flag=text_like, **ctx,
                ))

        elif issue_type in ("disguised_missing", "numeric_stored_as_text", "padded_text"):
            what = {"disguised_missing": "treat placeholder text as missing",
                    "numeric_stored_as_text": "read numbers written as text as numbers",
                    "padded_text": "trim leading/trailing spaces"}[issue_type]
            recommendations.append(_rec(
                issue, "auto_prep", "Normalised Automatically Before Profiling",
                f"Already applied: {what} in '{col_name}' ({details.get('count', 0)} cell(s))",
                column_name=col_name, count=details.get("count", 0), pct=details.get("pct", 0.0),
                examples=details.get("examples", []), kind=issue_type,
            ))

        elif issue_type == "inconsistent_categories":
            recommendations.append(_rec(
                issue, "merge_variants", "Merge Spelling / Case Variants",
                f"Write each value of '{col_name}' one way ({details.get('count', 0)} cell(s) use a variant spelling)",
                column_name=col_name, count=details.get("count", 0), mapping=details.get("mapping", {}),
            ))

        elif issue_type == "personal_data":
            recommendations.append(_rec(
                issue, "drop_personal_data", "Drop Personal-Data Column",
                f"Drop '{col_name}' ({details.get('kind', 'personal data').replace('_', ' ')}: privacy risk, no generalisable signal)",
                is_destructive=True, column_name=col_name, kind=details.get("kind"), unique_count=details.get("unique_count", 0),
            ))

        elif issue_type == "code_prefix_feature":
            keep = details.get("keep", [])
            recommendations.append(_rec(
                issue, "extract_prefix", "Extract Code Prefix as a Category",
                f"Create '{col_name}_prefix' from the letter prefix of '{col_name}' ({', '.join(keep[:5])})",
                column_name=col_name, keep=keep, coverage=details.get("coverage"), sample_values=details.get("sample_values", []),
            ))

        elif issue_type == "text_token_features":
            words = details.get("words", [])
            recommendations.append(_rec(
                issue, "word_flags", "Extract Frequent Words as 0/1 Features",
                f"Create {len(words)} word flags from '{col_name}' ({', '.join(words[:5])})",
                column_name=col_name, words=words, coverage=details.get("coverage"), sample_values=details.get("sample_values", []),
            ))

        elif issue_type == "unparsed_datetime_feature":
            recommendations.append(_rec(
                issue, "parse_datetime", "Parse Datetime & Extract Temporal Signals (Year, Month, Day)",
                f"Engineer Temporal Features from '{col_name}'",
                column_name=col_name, sample_values=details.get("sample_values", []),
            ))

        elif issue_type == "unit_mixed_feature":
            recommendations.append(_rec(
                issue, "split_units", "Split into Numeric Value & Categorical Unit Columns",
                f"Disentangle '{col_name}' into Value & Unit",
                column_name=col_name, sample_values=details.get("sample_values", []),
            ))

        elif issue_type == "multilabel_delimited_text":
            recommendations.append(_rec(
                issue, "multi_hot", "Multi-Hot Binary Token Encoding (Delimited List)",
                f"Tokenize & Multi-Hot Encode individual categories in '{col_name}'",
                column_name=col_name, sample_values=details.get("sample_values", []),
            ))

        elif issue_type == "invalid_domain_values":
            neg_cnt = details.get("negative_count", 0)
            recommendations.append(_rec(
                issue, "replace_negatives_median", "Treat Invalid Negatives as Missing (then impute + flag)",
                f"Treat {neg_cnt} invalid negative values in '{col_name}' as blanks and impute them with the other gaps",
                column_name=col_name, negative_count=neg_cnt, min_value=details.get("min_value", 0.0),
            ))

        elif issue_type == "statistical_outliers":
            extreme = int(details.get("extreme_count", 0))
            common = dict(column_name=col_name, outlier_count=details.get("outlier_count", 0),
                          outlier_pct=details.get("outlier_pct", 0.0), extreme_count=extreme,
                          skewness=details.get("skewness", 0.0))
            if extreme:
                lower_b, upper_b = details.get("extreme_lower", 0.0), details.get("extreme_upper", 0.0)
                recommendations.append(_rec(
                    issue, "cap_outliers", "Cap Extreme Values (3x IQR) + Capped Flag",
                    f"Cap {extreme} extreme value(s) in '{col_name}' to [{lower_b}, {upper_b}] and flag them",
                    lower_bound=lower_b, upper_bound=upper_b, add_flag=True, fence="3x IQR", **common,
                ))
            else:
                # Values beyond 1.5x IQR but within 3x IQR are normal tails (fares, incomes): keep them by default.
                lower_b, upper_b = details.get("lower_bound", 0.0), details.get("upper_bound", 0.0)
                recommendations.append(_rec(
                    issue, "cap_outliers", "Optional: Cap Tail Values (1.5x IQR)",
                    f"Keep {details.get('outlier_count', 0)} tail value(s) in '{col_name}' (no extreme values); capping is optional",
                    lower_bound=lower_b, upper_bound=upper_b, add_flag=True, fence="1.5x IQR", is_approved=False,
                    severity="low", **common,
                ))

        elif issue_type == "multicollinear_features":
            col1, col2 = details.get("col1"), details.get("col2")
            corr = details.get("correlation", 0.0)
            recommendations.append(_rec(
                issue, "drop_collinear", f"Drop Redundant Collinear Feature '{col2}'",
                f"Drop '{col2}' (Correlation {corr} with '{col1}')",
                column=col2, is_destructive=True, is_approved=False, severity="medium",
                col1=col1, col2=col2, correlation=corr, column_name=col2,
            ))

        elif issue_type == "target_leakage":
            recommendations.append(_rec(
                issue, "drop_leakage", "Drop Leaking Feature",
                f"Drop '{col_name}' (predicts the target almost perfectly)",
                is_destructive=True, column_name=col_name, score=details.get("score"),
                leak_method=details.get("method"), target_column=details.get("target"),
            ))

        elif issue_type == "constant_feature":
            recommendations.append(_rec(
                issue, "drop_constant", "Drop Zero-Variance Constant Column", f"Drop '{col_name}' (Zero Variance)",
                is_destructive=True, column_name=col_name,
            ))

        elif issue_type == "high_cardinality_text":
            uniq = details.get("unique_count", 0)
            recommendations.append(_rec(
                issue, "drop_high_cardinality", "Drop High-Cardinality Free-Text Column",
                f"Drop '{col_name}' ({uniq} unique values, not usable as a categorical feature)",
                is_destructive=True, column_name=col_name, unique_count=uniq, unique_pct=details.get("unique_pct", 0.0),
                supported_categories=details.get("supported_categories"), min_support=details.get("min_support"),
            ))

        elif issue_type == "rare_categories":
            keep = details.get("keep_top", 10)
            recommendations.append(_rec(
                issue, "group_rare", f"Group Rare Categories (Top {keep} + 'Other')",
                f"Keep the {keep} most frequent values of '{col_name}' and group the rest as 'Other'",
                column_name=col_name, unique_count=details.get("unique_count", 0), keep_top=keep,
            ))

        elif issue_type == "categorical_encoding":
            cols = details.get("columns", [])
            recommendations.append(_rec(
                issue, "one_hot_encode", "One-Hot Encode Categorical Features",
                f"Convert {len(cols)} categorical column(s) into numeric indicator columns",
                column=None, columns=cols,
            ))

        elif issue_type == "class_imbalance":
            ratio_disp = details.get("ratio_display", "")
            ratio = details.get("imbalance_ratio", 1.0)
            ctx = dict(target_column=col_name, majority_class=details.get("majority_class"),
                       majority_ratio=details.get("majority_ratio", 0.5), imbalance_ratio=ratio, ratio_display=ratio_disp)
            if ratio >= 6.0:
                recommendations.append(_rec(
                    issue, "smote_class_weights", "SMOTE Oversampling + Balanced Class Weights (in training folds)",
                    f"Apply SMOTE + Balanced Weights during model training ({ratio_disp})", **ctx,
                ))
            else:
                recommendations.append(_rec(
                    issue, "class_weights", "Balanced Class Weighting (Cost-Sensitive)",
                    f"Apply Cost-Sensitive Class Weights during model training ({ratio_disp})", **ctx,
                ))

    return recommendations
