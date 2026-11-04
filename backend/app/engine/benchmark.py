import logging
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

MAX_BENCHMARK_ROWS = 2000
MIN_BENCHMARK_ROWS = 10
MAX_TEXT_CATEGORIES = 50


class BenchmarkError(ValueError):
    """Raised when the dataset/objective combination cannot be benchmarked; message is user-facing."""


def prepare_target(y: pd.Series, problem_type: str) -> pd.Series:
    if problem_type == "classification":
        n_classes = y.nunique()
        if n_classes < 2:
            raise BenchmarkError("The target has only one class after cleaning. Classification needs at least two classes.")
        if pd.api.types.is_numeric_dtype(y):
            non_integer = (y % 1 != 0).mean()
            if non_integer > 0.05 or n_classes > max(20, 0.2 * len(y)):
                raise BenchmarkError(
                    f"The target looks continuous ({n_classes} distinct numeric values). Choose 'regression' instead."
                )
        elif n_classes > max(20, 0.5 * len(y)):
            raise BenchmarkError(f"The target has {n_classes} distinct values, which looks like an identifier, not a class label.")
        return pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index, name=y.name)

    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.isna().mean() > 0.05:
        raise BenchmarkError("The target is not numeric. Regression needs a numeric target; choose 'classification' instead.")
    if numeric.nunique() <= 2:
        raise BenchmarkError("The target has two or fewer distinct values. Choose 'classification' instead.")
    return numeric


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    text_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    too_wide = [c for c in text_cols if X[c].nunique() > MAX_TEXT_CATEGORIES]
    X = X.drop(columns=too_wide)
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    if X.shape[1] == 0:
        raise BenchmarkError("No usable feature columns remain after cleaning. Approve fewer column drops or add features.")
    return X


def _make_pipeline(model, use_smote: bool, smote_k: int):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps + [("smote", SMOTE(k_neighbors=smote_k, random_state=42)), ("model", model)])
    from sklearn.pipeline import Pipeline
    return Pipeline(steps + [("model", model)])


# Metric registry: key -> (label, sklearn scoring, kind). kind: "pct" (0-1 shown as %), "ratio" (0-1), "error" (lower is better)
METRICS: Dict[str, Dict[str, Tuple[str, Any, str]]] = {
    "classification": {
        "f1_macro": ("F1-Score (Macro)", make_scorer(f1_score, average="macro", zero_division=0), "pct"),
        "balanced_accuracy": ("Balanced Accuracy", "balanced_accuracy", "pct"),
        "accuracy": ("Accuracy", "accuracy", "pct"),
        "roc_auc": ("ROC AUC", "roc_auc", "ratio"),
        "average_precision": ("PR AUC (Average Precision)", "average_precision", "ratio"),
    },
    "regression": {
        "r2": ("R² Score", make_scorer(r2_score), "ratio"),
        "mae": ("MAE (Mean Absolute Error)", "neg_mean_absolute_error", "error"),
        "rmse": ("RMSE (Root Mean Squared Error)", "neg_root_mean_squared_error", "error"),
    },
}
BINARY_ONLY_METRICS = {"roc_auc", "average_precision"}
CV_STRATEGIES = {"stratified", "kfold", "time"}
MAX_PLAN_ROWS = 50000
SLOW_MODEL_ROW_LIMIT = 5000   # SVC / SVR scale ~quadratically; skipped above this many rows
# Verdict thresholds: gain over the best no-skill baseline (metric units, relative for error metrics) and in
# standard errors of the cross-validated mean.
MIN_LIFT, USEFUL_LIFT = 0.02, 0.05
NO_SIGNAL_Z, USEFUL_Z = 3.0, 6.0


def _time_order(df: pd.DataFrame, time_column: Optional[str]) -> Optional[List[str]]:
    """Columns to sort by for a time-ordered split (raw date columns become <col>_year/_month/_day after cleaning)."""
    if not time_column:
        return None
    parts = [f"{time_column}_{p}" for p in ("year", "month", "day")]
    if all(p in df.columns for p in parts):
        return parts
    if time_column in df.columns and pd.api.types.is_numeric_dtype(df[time_column]):
        return [time_column]
    return None


def _display(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{round(value * 100, 2)}%"
    if kind == "error":
        return f"{value:,.4g}"
    return f"{round(value, 4)}"


def benchmark_models(
    df_cleaned: pd.DataFrame,
    target_col: str,
    problem_type: str = "classification",
    imbalance_strategy: Optional[str] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], str, str, int, Dict[str, Any]]:
    """
    Cross-validates candidate models. Imputation, scaling and SMOTE are fitted inside each training
    fold so validation folds never leak into preprocessing.
    imbalance_strategy: None | "class_weights" | "smote_class_weights" (from approved recommendations).
    plan: optional evaluation plan {metric, cv, time_column, max_rows, source, reason} (Evaluate agent / user).
    Returns (leaderboard, primary_metric_label, summary_text, cv_folds, evaluation).
    """
    plan = dict(plan or {})
    if target_col not in df_cleaned.columns:
        raise BenchmarkError(f"Target column '{target_col}' is not present in the cleaned dataset.")

    df = df_cleaned[df_cleaned[target_col].notna()]
    if len(df) < MIN_BENCHMARK_ROWS:
        raise BenchmarkError(f"Only {len(df)} labelled rows are available; at least {MIN_BENCHMARK_ROWS} are needed to benchmark models.")

    registry = METRICS[problem_type]
    default_metric = "f1_macro" if problem_type == "classification" else "r2"
    metric_key = plan.get("metric") if plan.get("metric") in registry else default_metric
    cv_kind = plan.get("cv") if plan.get("cv") in CV_STRATEGIES else ("stratified" if problem_type == "classification" else "kfold")
    order_cols = _time_order(df, plan.get("time_column")) if cv_kind == "time" else None
    if cv_kind == "time" and not order_cols:
        cv_kind = "stratified" if problem_type == "classification" else "kfold"
    max_rows = int(min(max(int(plan.get("max_rows") or MAX_BENCHMARK_ROWS), 200), MAX_PLAN_ROWS))
    rows_available = len(df)

    if order_cols:
        df = df.sort_values(order_cols, kind="stable")
        if len(df) > max_rows:
            df = df.iloc[-max_rows:]  # most recent rows, order preserved

    y = prepare_target(df[target_col], problem_type)
    X = _prepare_features(df.drop(columns=[target_col]))

    binary = problem_type == "classification" and y.nunique() == 2
    if metric_key in BINARY_ONLY_METRICS and not binary:
        metric_key = default_metric
    if binary:
        minority = y.value_counts().idxmin()
        y = (y == minority).astype(int)  # positive class = minority, so PR AUC measures what matters

    if not order_cols and len(X) > max_rows:
        stratify = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=stratify, random_state=42)

    primary_metric, scorer, metric_kind = registry[metric_key]
    use_weights = problem_type == "classification" and imbalance_strategy in ("class_weights", "smote_class_weights")
    use_smote = False
    smote_k = 5

    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        if min_class < 2:
            raise BenchmarkError("At least one target class has fewer than 2 rows, so cross-validation is impossible.")
        n_splits = min(5, min_class)
        if cv_kind == "time":
            cv = TimeSeriesSplit(n_splits=n_splits)
        elif cv_kind == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if imbalance_strategy == "smote_class_weights":
            min_train_class = int(np.floor(min_class * (n_splits - 1) / n_splits))
            smote_k = min(5, min_train_class - 1)
            use_smote = smote_k >= 1
        cw = "balanced" if use_weights else None
        candidates = [
            ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight=cw, random_state=42, n_jobs=-1),
             "Ensemble of decision trees. Excellent balance of non-linear accuracy, robustness, and feature interpretability.", "High"),
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight=cw, random_state=42),
             "Linear decision boundary baseline. Highly explainable, fast inference, optimal for linearly separable data.", "Moderate"),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
             "Sequential boosting algorithm. High predictive power for complex tabular interactions.", "High"),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, class_weight=cw, random_state=42),
             "Single rule tree. Fully white-box transparent, instantaneous training, prone to slight variance.", "Moderate"),
            ("Support Vector Machine (SVC)", SVC(kernel="rbf", class_weight=cw, random_state=42),
             "Kernel-based max-margin classifier. Effective in high-dimensional transformed spaces.", "Moderate"),
        ]
        try:
            from xgboost import XGBClassifier
            candidates.insert(1, ("XGBoost", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1),
                                  "Regularized gradient boosted trees with strong benchmark accuracy.", "High"))
        except Exception:
            pass
        try:
            from lightgbm import LGBMClassifier
            candidates.insert(2, ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=3, class_weight=cw, verbose=-1, random_state=42, n_jobs=-1),
                                  "Leaf-wise gradient boosting. Extremely fast training on large tabular datasets.", "High"))
        except Exception:
            pass
        # The fairest no-skill reference depends on the metric: majority class wins on accuracy, random guessing in
        # proportion to the classes wins on macro-F1 with many balanced classes. The best of them is the bar to clear.
        baselines = [
            ("Baseline (always predicts the majority class)", DummyClassifier(strategy="most_frequent")),
            ("Baseline (guesses classes at their observed rates)", DummyClassifier(strategy="stratified", random_state=42)),
            ("Baseline (guesses classes uniformly at random)", DummyClassifier(strategy="uniform", random_state=42)),
        ]
    else:
        n_splits = min(5, max(2, len(X) // 10))
        cv = TimeSeriesSplit(n_splits=n_splits) if cv_kind == "time" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        candidates = [
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
             "Non-linear ensemble capturing complex continuous interactions without overfitting.", "High"),
            ("Ridge Regression", Ridge(alpha=1.0, random_state=42),
             "L2 regularized linear regression. Robust against collinearity and small variance fluctuations.", "Moderate"),
            ("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
             "Iterative loss minimization for continuous targets. Top-tier tabular accuracy.", "High"),
            ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5, random_state=42),
             "Piecewise constant regressor with simple explainable decision thresholds.", "Moderate"),
            ("Support Vector Regressor (SVR)", SVR(kernel="rbf"),
             "Epsilon-insensitive loss regressor for non-linear continuous mappings.", "Moderate"),
        ]
        try:
            from xgboost import XGBRegressor
            candidates.insert(1, ("XGBoost Regressor", XGBRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
                                  "Extreme gradient boosting regressor optimized for fast execution and high R².", "High"))
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor
            candidates.insert(2, ("LightGBM Regressor", LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1, random_state=42, n_jobs=-1),
                                  "High-speed tree boosting with gradient-based one-side sampling.", "High"))
        except Exception:
            pass
        baselines = [
            ("Baseline (always predicts the mean)", DummyRegressor(strategy="mean")),
            ("Baseline (always predicts the median)", DummyRegressor(strategy="median")),
        ]

    skipped: List[str] = []
    if len(X) > SLOW_MODEL_ROW_LIMIT:
        skipped = [c[0] for c in candidates if c[0].startswith("Support Vector")]
        candidates = [c for c in candidates if not c[0].startswith("Support Vector")]

    def evaluate(name, model, desc, suit, is_baseline=False) -> Optional[Dict[str, Any]]:
        try:
            pipeline = _make_pipeline(model, use_smote and not is_baseline, smote_k)
            start_t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, error_score="raise")
            elapsed = round(time.time() - start_t, 3)
            mean_score = float(np.mean(scores))
            if np.isnan(mean_score):
                raise ValueError("score is NaN")
        except Exception as e:
            logger.warning(f"Benchmark model '{name}' failed: {e}")
            errors.append(f"{name}: {e}")
            return None
        shown = abs(mean_score) if metric_kind == "error" else mean_score
        return {
            "model_name": name,
            "problem_type": problem_type,
            "metric_name": primary_metric,
            "metric_value": round(shown, 4),
            "metric_std": round(float(np.std(scores)), 4),
            "metric_display": _display(shown, metric_kind),
            "training_time_sec": elapsed,
            "description": desc,
            "suitability": suit,
            "is_recommended": False,
            "is_baseline": is_baseline,
            "_sort": mean_score,  # sklearn convention: higher is better (errors are negated)
        }

    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for name, model, desc, suit in candidates:
        row = evaluate(name, model, desc, suit)
        if row:
            results.append(row)

    if not results:
        raise BenchmarkError(f"Every candidate model failed to train. First error: {errors[0] if errors else 'unknown'}")

    results.sort(key=lambda x: x["_sort"], reverse=True)
    for idx, item in enumerate(results):
        item["rank"] = idx + 1
        item["is_recommended"] = idx == 0
    base_rows = [evaluate(name, model, "Reference point: a model that ignores every feature. Real models should beat it clearly.",
                          "Baseline", is_baseline=True) for name, model in baselines]
    base_rows = [r for r in base_rows if r]
    base_row = max(base_rows, key=lambda r: r["_sort"]) if base_rows else None
    if base_row:
        base_row["rank"] = len(results) + 1
        results.append(base_row)
    for item in results:
        item.pop("_sort", None)

    balance_note = ""
    if use_smote:
        balance_note = " SMOTE oversampling and balanced class weights were applied inside the training folds."
    elif use_weights:
        balance_note = " Balanced class weights were applied."
    best = results[0]
    split_label = {"time": "time-ordered", "kfold": "K-fold", "stratified": "stratified"}[cv_kind]
    summary_text = (
        f"Top Recommended Model: **{best['model_name']}** achieved the best {n_splits}-fold {split_label} cross-validated "
        f"{primary_metric} of **{best['metric_display']}** (± {best['metric_std']}) on {len(X)} rows and {X.shape[1]} features.{balance_note}"
    )
    verdict, lift = None, None
    if base_row:
        summary_text += f" The no-skill baseline scores {base_row['metric_display']}."
        verdict, lift = _verdict(best, base_row, metric_kind, n_splits)
        if verdict == "no_signal":
            for item in results:
                item["is_recommended"] = False
            summary_text = (
                f"No model beats the no-skill baseline by more than fold-to-fold noise: the best, {best['model_name']}, "
                f"scores {best['metric_display']} (± {best['metric_std']}) vs {base_row['metric_display']} for always guessing. "
                f"The features carry no usable signal for '{target_col}', so these scores are chance level. Do not deploy a model; "
                f"add informative features or reconsider the target."
            )
        elif verdict == "weak":
            summary_text += (" The improvement over the baseline is small relative to the variation between folds, so treat "
                             "this model as weak and validate it on fresh data before relying on it.")
    if errors:
        summary_text += f" {len(errors)} candidate model(s) failed and were excluded."
    evaluation = {
        "metric": metric_key,
        "metric_label": primary_metric,
        "higher_is_better": metric_kind != "error",
        "cv": cv_kind,
        "n_splits": n_splits,
        "time_column": plan.get("time_column") if cv_kind == "time" else None,
        "rows_used": int(len(X)),
        "rows_available": int(rows_available),
        "skipped_models": skipped,
        "source": plan.get("source", "default") if plan else "default",
        "reason": plan.get("reason"),
        "verdict": verdict,
        "lift_over_baseline": lift,
    }
    return results, primary_metric, summary_text, n_splits, evaluation


def _verdict(best: Dict[str, Any], base: Dict[str, Any], kind: str, n_splits: int) -> Tuple[str, float]:
    """
    'no_signal' / 'weak' / 'useful': is the best model's gain over no-skill guessing real?
    The gain is measured in standard errors of the k-fold mean, combining the spread of both scores (a random-guess
    baseline wobbles a lot). The cut-offs are conservative because the best of several models is picked after the fact.
    """
    b, s = float(best["metric_value"]), float(base["metric_value"])
    std = float(np.hypot(float(best.get("metric_std") or 0.0), float(base.get("metric_std") or 0.0)))
    scale = (abs(s) or 1.0) if kind == "error" else 1.0
    lift = ((s - b) if kind == "error" else (b - s)) / scale
    se = max(std / scale / np.sqrt(max(n_splits, 2)), 0.005)
    z = lift / se
    if lift < MIN_LIFT or z < NO_SIGNAL_Z:
        return "no_signal", round(lift, 4)
    if lift < USEFUL_LIFT or z < USEFUL_Z:
        return "weak", round(lift, 4)
    return "useful", round(lift, 4)
