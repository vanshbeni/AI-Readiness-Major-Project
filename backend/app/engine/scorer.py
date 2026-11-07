"""
Data Health Score: how ready a dataset is for machine learning, 0-100.

Seven dimensions are measured from the *amount* of data affected, not from the number of issues raised:

  * Each column-level dimension combines BREADTH (share of columns with the defect) and DEPTH (average share of
    their cells affected, weighted by how serious the defect is):  100 * (1 - 0.5*breadth - 0.5*depth).
    One mildly affected column barely moves it; a defect that touches many columns deeply drives it to 0.
  * Dimensions are combined with a weighted GEOMETRIC mean, so one failing dimension pulls the score down
    instead of being averaged away by six good ones (a weighted arithmetic mean let messy data score 'A').
  * Features that carry no relationship with the target cap the score at grade D: cleaning cannot fix that.
"""
from math import exp, log
from typing import Any, Dict, List, Optional, Tuple

WEIGHTS = {
    "completeness": 0.20,
    "validity": 0.20,
    "signal": 0.20,
    "usefulness": 0.15,
    "target": 0.10,
    "uniqueness": 0.10,
    "outliers": 0.05,
}
# How much a defective cell counts towards validity (1 = the value is wrong or unusable as stored)
VALIDITY_SEVERITY = {
    "invalid_domain_values": 1.0,
    "disguised_missing": 1.0,
    "numeric_stored_as_text": 1.0,
    "mixed_date_formats": 0.6,
    "inconsistent_categories": 0.5,
    "padded_text": 0.3,
}
# How much an unusable column counts towards feature usefulness
USEFULNESS_WEIGHT = {
    "target_leakage": 2.0,
    "personal_data": 1.0,
    "constant_feature": 1.0,
    "identifier_column": 0.5,
    "high_cardinality_text": 0.5,
    "multicollinear_features": 0.3,
}
NO_SIGNAL_SCORE_CAP = 64.0
TARGET_MISSING_WEIGHT = 2.0


def task_weights(problem_type: Optional[str]) -> Dict[str, float]:
    """Dimension weights for the task, renormalised to 1. Unsupervised tasks have no target; anomaly detection
    has no signal to measure up front (unusual rows are what it looks for)."""
    drop = {"clustering": {"target"}, "anomaly_detection": {"target", "signal"}}.get(problem_type or "", set())
    kept = {k: w for k, w in WEIGHTS.items() if k not in drop}
    total = sum(kept.values())
    return {k: w / total for k, w in kept.items()}


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


def breadth_depth(shares: Dict[str, float], n_columns: int) -> float:
    """100 * (1 - 0.5*breadth - 0.5*depth) over the columns with a non-zero affected share."""
    affected = [min(1.0, s) for s in shares.values() if s > 0]
    if not affected or n_columns <= 0:
        return 100.0
    breadth = min(1.0, len(affected) / n_columns)
    depth = sum(affected) / len(affected)
    return round(max(0.0, 100.0 * (1 - 0.5 * breadth - 0.5 * depth)), 1)


def _cell_share(issue: Dict[str, Any], rows: int) -> float:
    d = issue.get("details") or {}
    if issue["issue_type"] == "invalid_domain_values":
        return float(d.get("negative_count", 0)) / max(rows, 1)
    if "count" in d:
        return float(d.get("count") or 0) / max(rows, 1)
    return float(d.get("pct", 0.0) or 0.0) / 100.0


def calculate_health_score(
    summary_stats: Dict[str, Any],
    column_profiles: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    problem_type: str = "classification",
    dropped_informative_ratio: Optional[float] = None,
) -> Tuple[float, Dict[str, float], str, str, List[str]]:
    """
    Returns (composite_score, sub_scores_dict, letter_grade, summary_text, top_negative_drivers).
    dropped_informative_ratio: share of informative feature columns removed by cleaning; penalised so the score
    cannot be raised simply by deleting problematic columns.
    """
    rows = int(summary_stats.get("row_count") or 0)
    n_cols = int(summary_stats.get("col_count") or len(column_profiles) or 1)
    n_features = max(1, n_cols - 1)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for i in issues:
        by_type.setdefault(i["issue_type"], []).append(i)

    # 1. Completeness: blank cells per column (placeholders already count as blank); the target counts double
    missing_shares = {}
    target_missing = 0.0
    for cp in column_profiles:
        share = float(cp.get("missing_pct", 0.0)) / 100.0
        if cp.get("is_target"):
            target_missing = share
            share = min(1.0, share * TARGET_MISSING_WEIGHT)
        missing_shares[cp["name"]] = share
    completeness = breadth_depth(missing_shares, n_cols)

    # 2. Validity & consistency: cells stored wrongly (placeholders, numbers as text, impossible values,
    #    mixed date formats, spelling variants, padded text), weighted by severity
    validity_shares: Dict[str, float] = {}
    for kind, severity in VALIDITY_SEVERITY.items():
        for i in by_type.get(kind, []):
            col = i.get("column") or ""
            validity_shares[col] = validity_shares.get(col, 0.0) + severity * _cell_share(i, rows)
    validity = breadth_depth(validity_shares, n_cols)

    # 3. Feature usefulness: share of feature columns that cannot (or must not) be used as they are
    unusable = 0.0
    for kind, w in USEFULNESS_WEIGHT.items():
        unusable += w * len(by_type.get(kind, []))
    usefulness = round(max(0.0, 100.0 * (1 - unusable / n_features)), 1)

    # 4. Signal. Supervised: strongest statistically real association of any usable feature with the target.
    #    Clustering: silhouette lift of real groups over the same data with columns shuffled (no relationships).
    signal_issue = (by_type.get("weak_target_signal") or by_type.get("weak_cluster_structure") or [None])[0]
    if signal_issue and signal_issue["issue_type"] == "weak_cluster_structure":
        gap = float(signal_issue["details"].get("gap", 0.0))
        signal = round(max(0.0, min(100.0, 20.0 + 80.0 * max(gap, 0.0) / 0.10)), 1)
    elif signal_issue:
        best = float(signal_issue["details"].get("max_association", 0.0))
        signal = round(max(0.0, min(100.0, 20.0 + 80.0 * best / 0.10)), 1)
    else:
        signal = 100.0
    no_signal = bool(signal_issue and signal_issue["details"].get("no_signal"))

    # 5. Target health: labelled share x class balance (1:1 = 1, 1:9 or worse ~ 0)
    balance = 1.0
    imbalance = (by_type.get("class_imbalance") or [None])[0]
    if imbalance and problem_type == "classification":
        ratio = float(imbalance["details"].get("imbalance_ratio", 1.0))
        balance = max(0.0, 1 - min(1.0, (ratio - 1.0) * 0.12))
    target = round(100.0 * (1 - target_missing) * balance, 1)

    # 6. Uniqueness: duplicate records
    dup_pct = float(summary_stats.get("duplicate_rows_pct", 0.0))
    uniqueness = round(max(0.0, 100.0 - min(100.0, dup_pct * 5.0)), 1)

    # 7. Outliers: only extreme values count (3x IQR); normal long tails are not defects
    outlier_shares = {}
    for i in by_type.get("statistical_outliers", []):
        d = i["details"]
        extreme = float(d.get("extreme_count", d.get("outlier_count", 0)))
        outlier_shares[i["column"]] = min(1.0, 5.0 * extreme / max(rows, 1))
    outliers = breadth_depth(outlier_shares, n_cols)

    dims = {
        "completeness": completeness, "validity": validity, "signal": signal, "usefulness": usefulness,
        "target": target, "uniqueness": uniqueness, "outliers": outliers,
    }
    weights = task_weights(problem_type)
    composite = 100.0 * exp(sum(w * log(max(dims[k], 1.0) / 100.0) for k, w in weights.items()))

    retention_penalty = 0.0
    if dropped_informative_ratio:
        retention_penalty = round(min(20.0, dropped_informative_ratio * 20.0), 1)
        composite -= retention_penalty
    if no_signal:
        composite = min(composite, NO_SIGNAL_SCORE_CAP)
    composite = round(max(0.0, min(100.0, composite)), 1)
    grade = grade_for_score(composite)

    # Drivers: dimensions ordered by how many points they cost
    labels = {
        "completeness": lambda: f"Missing values (incl. placeholders) in {sum(1 for s in missing_shares.values() if s > 0)} column(s)",
        "validity": lambda: "Wrongly stored values in {} column(s) ({})".format(
            sum(1 for s in validity_shares.values() if s > 0),
            ", ".join(sorted({k.replace('_', ' ') for k in VALIDITY_SEVERITY if by_type.get(k)}))),
        "signal": lambda: (("No natural groups: clusters would be arbitrary" if no_signal else "Only weak group structure")
                           if problem_type == "clustering" else
                           ("No feature is related to the target: models can only reach chance level" if no_signal
                            else "Weak predictive signal for the target")),
        "usefulness": lambda: "Unusable feature columns ({})".format(
            ", ".join(sorted({k.replace('_', ' ') for k in USEFULNESS_WEIGHT if by_type.get(k)}))),
        "target": lambda: f"Target health ({round(target_missing * 100, 1)}% unlabelled"
                          + (", class imbalance)" if balance < 1 else ")"),
        "uniqueness": lambda: f"{summary_stats.get('duplicate_rows_count', 0)} duplicate records",
        "outliers": lambda: f"Extreme values in {len(outlier_shares)} numeric column(s)",
    }
    cost = {k: -weights[k] * log(max(dims[k], 1.0) / 100.0) for k in weights}
    drivers = [labels[k]() for k in sorted(cost, key=cost.get, reverse=True) if dims[k] < 90]
    if retention_penalty > 0:
        drivers.append(f"Information loss from dropped feature columns (-{retention_penalty} pts)")
    if not drivers:
        drivers.append("Dataset is clean with minimal structural issues")

    if no_signal and problem_type == "clustering":
        summary_text = (f"Dataset scored {composite}/100: the rows form no natural groups (they cluster no better than the "
                        f"same data with its columns shuffled), so any segmentation would be arbitrary.")
    elif no_signal:
        summary_text = (f"Dataset scored {composite}/100: none of its features is related to the target, so cleaning "
                        f"alone cannot make it ML-ready. Add informative features or reconsider the target.")
    elif composite >= 85:
        summary_text = f"Dataset is in high health with a score of {composite}/100. Minor polish will optimize ML readiness."
    elif composite >= 70:
        summary_text = f"Dataset scored {composite}/100. Key remediation steps required for {'; '.join(drivers[:2])}."
    else:
        summary_text = f"Dataset readiness is compromised at {composite}/100: {'; '.join(drivers[:3])}."

    sub_scores = {
        "missingness_score": completeness,
        "duplicate_score": uniqueness,
        "outlier_score": outliers,
        "validity_score": validity,
        "target_balance_score": target,
        "feature_quality_score": usefulness,
        "signal_score": signal,
    }
    return composite, sub_scores, grade, summary_text, drivers
