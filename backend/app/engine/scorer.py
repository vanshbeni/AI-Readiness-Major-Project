from typing import Dict, Any, List, Optional, Tuple

WEIGHTS = {
    "missingness": 0.25,
    "duplicate": 0.15,
    "outlier": 0.15,
    "validity": 0.15,
    "target_balance": 0.15,
    "feature_quality": 0.15,
}


def grade_for_score(score: float) -> str:
    if score >= 90:
        return "A (Excellent Readiness)"
    if score >= 80:
        return "B (Good Readiness)"
    if score >= 70:
        return "C (Fair - Needs Cleaning)"
    if score >= 60:
        return "D (Poor - High Risk)"
    return "F (Critical Quality Defects)"


def calculate_health_score(
    summary_stats: Dict[str, Any],
    column_profiles: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    problem_type: str = "classification",
    dropped_informative_ratio: Optional[float] = None,
) -> Tuple[float, Dict[str, float], str, str, List[str]]:
    """
    Computes a composite 0-100 Data Health Score with weighted sub-scores.
    dropped_informative_ratio: share of informative (non-ID, non-constant) feature columns removed by
    cleaning; penalised so the score cannot be raised simply by deleting problematic columns.
    Returns (composite_score, sub_scores_dict, letter_grade, summary_text, top_negative_drivers).
    """
    # 1. Missingness: overall cell share plus the worst single column (incl. missing target labels)
    overall_missing_pct = summary_stats.get("overall_missing_pct", 0.0)
    worst_col_missing = max((cp.get("missing_pct", 0.0) for cp in column_profiles), default=0.0)
    missing_penalty = min(100.0, overall_missing_pct * 2.5 + worst_col_missing * 0.25)
    missingness_score = max(0.0, round(100.0 - missing_penalty, 1))

    # 2. Duplication
    dup_pct = summary_stats.get("duplicate_rows_pct", 0.0)
    duplicate_score = max(0.0, round(100.0 - min(100.0, dup_pct * 5.0), 1))

    # 3. Outliers
    outlier_issues = [i for i in issues if i["issue_type"] == "statistical_outliers"]
    if outlier_issues:
        avg_outlier_pct = sum(i["details"].get("outlier_pct", 0.0) for i in outlier_issues) / len(outlier_issues)
        outlier_penalty = min(100.0, avg_outlier_pct * 3.0 + len(outlier_issues) * 4.0)
    else:
        outlier_penalty = 0.0
    outlier_score = max(0.0, round(100.0 - outlier_penalty, 1))

    # 4. Validity
    validity_issues = [i for i in issues if i["issue_type"] == "invalid_domain_values"]
    validity_score = max(0.0, round(100.0 - min(100.0, len(validity_issues) * 20.0), 1))

    # 5. Target balance (majority/minority ratio; 1:1 = 100, 1:9 or worse ~ 0)
    imbalance_issues = [i for i in issues if i["issue_type"] == "class_imbalance"]
    if imbalance_issues and problem_type == "classification":
        ratio = imbalance_issues[0]["details"].get("imbalance_ratio", 1.0)
        target_balance_score = max(0.0, round(100.0 - min(100.0, (ratio - 1.0) * 12.0), 1))
    else:
        target_balance_score = 100.0

    # 6. Feature quality
    corr_issues = [i for i in issues if i["issue_type"] == "multicollinear_features"]
    const_issues = [i for i in issues if i["issue_type"] == "constant_feature"]
    id_issues = [i for i in issues if i["issue_type"] == "identifier_column"]
    feat_penalty = min(100.0, len(corr_issues) * 10.0 + len(const_issues) * 15.0 + len(id_issues) * 5.0)
    feature_quality_score = max(0.0, round(100.0 - feat_penalty, 1))

    composite_score = (
        missingness_score * WEIGHTS["missingness"]
        + duplicate_score * WEIGHTS["duplicate"]
        + outlier_score * WEIGHTS["outlier"]
        + validity_score * WEIGHTS["validity"]
        + target_balance_score * WEIGHTS["target_balance"]
        + feature_quality_score * WEIGHTS["feature_quality"]
    )

    retention_penalty = 0.0
    if dropped_informative_ratio:
        retention_penalty = round(min(20.0, dropped_informative_ratio * 20.0), 1)
        composite_score -= retention_penalty

    composite_score = round(max(0.0, min(100.0, composite_score)), 1)
    grade = grade_for_score(composite_score)

    drivers = []
    if missingness_score < 80:
        drivers.append(f"High missingness ({overall_missing_pct}% of cells, worst column {worst_col_missing}%)")
    if duplicate_score < 90:
        drivers.append(f"{summary_stats.get('duplicate_rows_count', 0)} duplicate records")
    if outlier_score < 85:
        drivers.append(f"Extreme statistical outliers in {len(outlier_issues)} numeric feature(s)")
    if validity_score < 90:
        drivers.append(f"Domain invalidity detected in {len(validity_issues)} column(s)")
    if target_balance_score < 80:
        drivers.append("Significant class imbalance in the target column")
    if feature_quality_score < 85:
        drivers.append("Multicollinear, constant or identifier features")
    if retention_penalty > 0:
        drivers.append(f"Information loss from dropped feature columns (-{retention_penalty} pts)")

    if not drivers:
        drivers.append("Dataset is clean with minimal structural issues")

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
