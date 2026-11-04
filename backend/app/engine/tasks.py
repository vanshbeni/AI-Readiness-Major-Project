"""The machine-learning tasks the platform supports and what each one needs."""
from typing import Optional

SUPERVISED = ("classification", "regression")
UNSUPERVISED = ("clustering", "anomaly_detection")
ALL_TASKS = SUPERVISED + UNSUPERVISED

LABELS = {
    "classification": "classification",
    "regression": "regression",
    "clustering": "clustering",
    "anomaly_detection": "anomaly detection",
}


def needs_target(problem_type: Optional[str]) -> bool:
    return problem_type in SUPERVISED


def is_unsupervised(problem_type: Optional[str]) -> bool:
    return problem_type in UNSUPERVISED


def target_or_none(problem_type: Optional[str], target_column: Optional[str]) -> Optional[str]:
    """Unsupervised objectives store an empty target; every engine call should see None instead."""
    return target_column or None if needs_target(problem_type) else None
