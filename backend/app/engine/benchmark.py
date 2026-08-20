import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR


def benchmark_models(
    df_cleaned: pd.DataFrame,
    target_col: str,
    problem_type: str = "classification",
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Fits and cross-validates a pool of candidate ML algorithms on the cleaned dataset.
    Returns (leaderboard_list, primary_metric_name, best_model_summary_text).
    """
    if target_col not in df_cleaned.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Sample dataset if too large for instant viva/demo responsiveness (<1000 rows sample)
    if len(df_cleaned) > 1000:
        df_sample = df_cleaned.sample(n=1000, random_state=42)
    else:
        df_sample = df_cleaned.copy()

    X = df_sample.drop(columns=[target_col], errors="ignore")
    y = df_sample[target_col]

    # Convert y to appropriate type
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name=target_col)
        else:
            y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0.0)

    # Drop high cardinality string/ID columns from X if any
    for col in X.columns:
        if X[col].dtype == object and X[col].nunique() > 20:
            X = X.drop(columns=[col])

    # Fill any remaining NaNs in X
    X = X.fillna(0.0)
    # Ensure numeric columns only
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    results = []

    if problem_type == "classification":
        primary_metric = "F1-Score (Macro)"
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average="macro", zero_division=0)

        # Model Pool
        candidates = [
            ("Random Forest", RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42, n_jobs=-1), "Ensemble of decision trees. Excellent balance of non-linear accuracy, robustness, and feature interpretability.", "High"),
            ("Logistic Regression", LogisticRegression(max_iter=200, random_state=42), "Linear decision boundary baseline. Highly explainable, fast inference, optimal for linearly separable data.", "Moderate"),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=25, max_depth=3, random_state=42), "Sequential boosting algorithm. High predictive power for complex tabular interactions.", "High"),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42), "Single rule tree. Fully white-box transparent, instantaneous training, prone to slight variance.", "Moderate"),
            ("Support Vector Machine (SVC)", SVC(kernel="rbf", max_iter=500, random_state=42), "Kernel-based max-margin classifier. Effective in high-dimensional transformed spaces.", "Moderate"),
        ]

        # Try XGBoost if installed
        try:
            from xgboost import XGBClassifier
            candidates.insert(1, ("XGBoost", XGBClassifier(n_estimators=25, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1), "State-of-the-art gradient boosted trees. Regularized learning objective with superior benchmark accuracy.", "High"))
        except Exception:
            pass

        # Try LightGBM if installed
        try:
            from lightgbm import LGBMClassifier
            candidates.insert(2, ("LightGBM", LGBMClassifier(n_estimators=25, max_depth=3, verbose=-1, random_state=42, n_jobs=-1), "Leaf-wise gradient boosting. Extremely fast training on large tabular datasets.", "High"))
        except Exception:
            pass

        for name, model, desc, suit in candidates:
            try:
                start_t = time.time()
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                elapsed = round(time.time() - start_t, 3)
                mean_score = float(np.mean(scores))
                if np.isnan(mean_score):
                    mean_score = 0.0

                results.append({
                    "model_name": name,
                    "problem_type": "classification",
                    "metric_name": primary_metric,
                    "metric_value": round(mean_score, 4),
                    "metric_display": f"{round(mean_score * 100, 2)}%",
                    "training_time_sec": elapsed,
                    "description": desc,
                    "suitability": suit,
                    "is_recommended": False,
                })
            except Exception:
                continue

        # Sort descending by F1-Score
        results.sort(key=lambda x: x["metric_value"], reverse=True)

    else:
        # Regression Problem
        primary_metric = "R² Score"
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(r2_score)

        candidates = [
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=25, max_depth=5, random_state=42, n_jobs=-1), "Non-linear ensemble capturing complex continuous interactions without overfitting.", "High"),
            ("Ridge Regression", Ridge(alpha=1.0, random_state=42), "L2 regularized linear regression. Robust against collinearity and small variance fluctuations.", "Moderate"),
            ("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=25, max_depth=3, random_state=42), "Iterative loss minimization for continuous targets. Top-tier tabular accuracy.", "High"),
            ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5, random_state=42), "Piecewise constant regressor with simple explainable decision thresholds.", "Moderate"),
            ("Support Vector Regressor (SVR)", SVR(kernel="rbf", max_iter=500), "Epsilon-insensitive loss regressor for non-linear continuous mappings.", "Moderate"),
        ]

        try:
            from xgboost import XGBRegressor
            candidates.insert(1, ("XGBoost Regressor", XGBRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=-1), "Extreme gradient boosting regressor optimized for fast execution and high R².", "High"))
        except Exception:
            pass

        try:
            from lightgbm import LGBMRegressor
            candidates.insert(2, ("LightGBM Regressor", LGBMRegressor(n_estimators=25, max_depth=3, verbose=-1, random_state=42, n_jobs=-1), "High-speed tree boosting with gradient-based one-side sampling.", "High"))
        except Exception:
            pass

        for name, model, desc, suit in candidates:
            try:
                start_t = time.time()
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                elapsed = round(time.time() - start_t, 3)
                mean_score = float(np.mean(scores))
                if np.isnan(mean_score):
                    mean_score = 0.0

                results.append({
                    "model_name": name,
                    "problem_type": "regression",
                    "metric_name": primary_metric,
                    "metric_value": round(mean_score, 4),
                    "metric_display": f"{round(mean_score, 4)}",
                    "training_time_sec": elapsed,
                    "description": desc,
                    "suitability": suit,
                    "is_recommended": False,
                })
            except Exception:
                continue

        # Sort descending by R² Score
        results.sort(key=lambda x: x["metric_value"], reverse=True)

    # Assign ranks & recommendation flag
    for idx, item in enumerate(results):
        item["rank"] = idx + 1
        item["id"] = f"bench_{idx + 1}"
        if idx == 0:
            item["is_recommended"] = True

    best_model = results[0]["model_name"] if results else "Standard Baseline"
    best_score = results[0]["metric_display"] if results else "N/A"
    summary_text = (
        f"Top Recommended Model: **{best_model}** achieved the highest cross-validated {primary_metric} of **{best_score}**. "
        f"It is ideally suited for this dataset's feature dimensionality and distribution characteristics."
    )

    return results, primary_metric, summary_text
