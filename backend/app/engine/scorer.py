from typing import Dict, Any, List, Tuple


def calculate_health_score(
    summary_stats: Dict[str, Any],
    column_profiles: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    problem_type: str = "classification",
) -> Tuple[float, Dict[str, float], str, str, List[str]]:
    """
    Computes a mathematical composite 0-100 Data Health Score with weighted sub-scores.
    Returns (composite_score, sub_scores_dict, letter_grade, summary_text, top_negative_drivers).
    """
    # 1. Missingness Sub-score (Weight: 25%)
    overall_missing_pct = summary_stats.get("overall_missing_pct", 0.0)
    # Penalty scales with missing percentage
    missing_penalty = min(100.0, overall_missing_pct * 2.5)
    missingness_score = max(0.0, round(100.0 - missing_penalty, 1))

    # 2. Duplication Sub-score (Weight: 15%)
    dup_pct = summary_stats.get("duplicate_rows_pct", 0.0)
    dup_penalty = min(100.0, dup_pct * 5.0)
    duplicate_score = max(0.0, round(100.0 - dup_penalty, 1))

    # 3. Outlier Sub-score (Weight: 15%)
    outlier_issues = [i for i in issues if i["issue_type"] == "statistical_outliers"]
    if outlier_issues:
        avg_outlier_pct = sum(i["details"].get("outlier_pct", 0.0) for i in outlier_issues) / len(outlier_issues)
        outlier_penalty = min(100.0, avg_outlier_pct * 3.0 + len(outlier_issues) * 4.0)
    else:
        outlier_penalty = 0.0
    outlier_score = max(0.0, round(100.0 - outlier_penalty, 1))

    # 4. Validity Sub-score (Weight: 15%)
    validity_issues = [i for i in issues if i["issue_type"] == "invalid_domain_values"]
    if validity_issues:
        val_penalty = min(100.0, len(validity_issues) * 20.0)
    else:
        val_penalty = 0.0
    validity_score = max(0.0, round(100.0 - val_penalty, 1))

    # 5. Target Balance Sub-score (Weight: 15%)
    imbalance_issues = [i for i in issues if i["issue_type"] == "class_imbalance"]
    if imbalance_issues and problem_type == "classification":
        maj_ratio = imbalance_issues[0]["details"].get("majority_ratio", 0.5)
        # 50:50 is 100, 95:5 is ~20
        imbalance_penalty = max(0.0, (maj_ratio - 0.5) * 160.0)
        target_balance_score = max(0.0, round(100.0 - imbalance_penalty, 1))
    else:
        target_balance_score = 100.0

    # 6. Feature Quality / Multicollinearity Sub-score (Weight: 15%)
    corr_issues = [i for i in issues if i["issue_type"] == "multicollinear_features"]
    const_issues = [i for i in issues if i["issue_type"] == "constant_feature"]
    feat_penalty = min(100.0, len(corr_issues) * 10.0 + len(const_issues) * 15.0)
    feature_quality_score = max(0.0, round(100.0 - feat_penalty, 1))

    # Weighted Composite Score
    weights = {
        "missingness": 0.25,
        "duplicate": 0.15,
        "outlier": 0.15,
        "validity": 0.15,
        "target_balance": 0.15,
        "feature_quality": 0.15,
    }

    composite_score = round(
        (missingness_score * weights["missingness"])
        + (duplicate_score * weights["duplicate"])
        + (outlier_score * weights["outlier"])
        + (validity_score * weights["validity"])
        + (target_balance_score * weights["target_balance"])
        + (feature_quality_score * weights["feature_quality"]),
        1,
    )
    composite_score = max(0.0, min(100.0, composite_score))

    # Determine Letter Grade
    if composite_score >= 90:
        grade = "A (Excellent Readiness)"
    elif composite_score >= 80:
        grade = "B (Good Readiness)"
    elif composite_score >= 70:
        grade = "C (Fair - Needs Cleaning)"
    elif composite_score >= 60:
        grade = "D (Poor - High Risk)"
    else:
        grade = "F (Critical Quality Defects)"

    # Identify Top Negative Drivers
    drivers = []
    if missingness_score < 80:
        drivers.append(f"High missingness ({overall_missing_pct}% total missing values)")
    if duplicate_score < 90:
        drivers.append(f"{summary_stats.get('duplicate_rows_count', 0)} exact duplicate records")
    if outlier_score < 85:
        drivers.append(f"Extreme statistical outliers in {len(outlier_issues)} numeric feature(s)")
    if validity_score < 90:
        drivers.append(f"Domain invalidity detected in {len(validity_issues)} column(s)")
    if target_balance_score < 80:
        drivers.append("Significant class imbalance in the target column")
    if feature_quality_score < 85:
        drivers.append("Multicollinear or constant redundant features")

    if not drivers:
        drivers.append("Dataset is clean with minimal structural issues")

    # Generate Summary Sentence
    if composite_score >= 85:
        summary_text = f"Dataset is in high health with a score of {composite_score}/100. Minor polish will optimize ML readiness."
    elif composite_score >= 70:
        summary_text = f"Dataset scored {composite_score}/100. Key remediation steps required for {', '.join(drivers[:2])}."
    else:
        summary_text = f"Dataset readiness is compromised at {composite_score}/100 due to severe data defects: {', '.join(drivers[:3])}."

    sub_scores = {
        "missingness_score": missingness_score,
        "duplicate_score": duplicate_score,
        "outlier_score": outlier_score,
        "validity_score": validity_score,
        "target_balance_score": target_balance_score,
        "feature_quality_score": feature_quality_score,
    }

    return composite_score, sub_scores, grade, summary_text, drivers
