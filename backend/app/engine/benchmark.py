import logging
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
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


def benchmark_models(
    df_cleaned: pd.DataFrame,
    target_col: str,
    problem_type: str = "classification",
    imbalance_strategy: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """
    Cross-validates candidate models. Imputation, scaling and SMOTE are fitted inside each training
    fold so validation folds never leak into preprocessing.
    imbalance_strategy: None | "class_weights" | "smote_class_weights" (from approved recommendations).
    Returns (leaderboard, primary_metric, summary_text, cv_folds).
    """
    if target_col not in df_cleaned.columns:
        raise BenchmarkError(f"Target column '{target_col}' is not present in the cleaned dataset.")

    df = df_cleaned[df_cleaned[target_col].notna()]
    if len(df) < MIN_BENCHMARK_ROWS:
        raise BenchmarkError(f"Only {len(df)} labelled rows are available; at least {MIN_BENCHMARK_ROWS} are needed to benchmark models.")

    y = prepare_target(df[target_col], problem_type)
    X = _prepare_features(df.drop(columns=[target_col]))

    if len(X) > MAX_BENCHMARK_ROWS:
        stratify = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        X, _, y, _ = train_test_split(X, y, train_size=MAX_BENCHMARK_ROWS, stratify=stratify, random_state=42)

    use_weights = problem_type == "classification" and imbalance_strategy in ("class_weights", "smote_class_weights")
    use_smote = False
    smote_k = 5

    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        if min_class < 2:
            raise BenchmarkError("At least one target class has fewer than 2 rows, so cross-validation is impossible.")
        n_splits = min(5, min_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        primary_metric = "F1-Score (Macro)"
        scorer = make_scorer(f1_score, average="macro", zero_division=0)
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
    else:
        n_splits = min(5, max(2, len(X) // 10))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        primary_metric = "R² Score"
        scorer = make_scorer(r2_score)
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

    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for name, model, desc, suit in candidates:
        try:
            pipeline = _make_pipeline(model, use_smote, smote_k)
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
            continue
        display = f"{round(mean_score * 100, 2)}%" if problem_type == "classification" else f"{round(mean_score, 4)}"
        results.append({
            "model_name": name,
            "problem_type": problem_type,
            "metric_name": primary_metric,
            "metric_value": round(mean_score, 4),
            "metric_display": display,
            "training_time_sec": elapsed,
            "description": desc,
            "suitability": suit,
            "is_recommended": False,
        })

    if not results:
        raise BenchmarkError(f"Every candidate model failed to train. First error: {errors[0] if errors else 'unknown'}")

    results.sort(key=lambda x: x["metric_value"], reverse=True)
    for idx, item in enumerate(results):
        item["rank"] = idx + 1
        item["is_recommended"] = idx == 0

    balance_note = ""
    if use_smote:
        balance_note = " SMOTE oversampling and balanced class weights were applied inside the training folds."
    elif use_weights:
        balance_note = " Balanced class weights were applied."
    best = results[0]
    summary_text = (
        f"Top Recommended Model: **{best['model_name']}** achieved the highest {n_splits}-fold cross-validated "
        f"{primary_metric} of **{best['metric_display']}** on {len(X)} rows and {X.shape[1]} features.{balance_note}"
    )
    if len(errors):
        summary_text += f" {len(errors)} candidate model(s) failed and were excluded."
    return results, primary_metric, summary_text, n_splits
