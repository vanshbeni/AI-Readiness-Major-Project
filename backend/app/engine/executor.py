import re
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from app.core.config import settings
from app.engine.profiler import infer_column_type


def execute_pipeline(
    df_raw: pd.DataFrame,
    recommendations: List[Dict[str, Any]],
    target_col: str = None,
    problem_type: str = "classification",
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Executes domain-aware remediation transformations in a safe mathematical sequence:
    1. Deduplication
    2. Datetime feature extraction (year, month, day)
    3. Mixed unit parsing (e.g. duration_value + duration_unit)
    4. Multi-label delimited token multi-hot encoding
    5. Invalid domain value clipping (negative to 0.0)
    6. Drop unviable / collinear features
    7. Semantic Missing Value Imputation (Unknown for entities, Median/Mean for numeric, Mode for low-card categories)
    8. High-cardinality entity binning & ID filtering
    9. Outlier Winsorization (IQR bounds - excluding year/age features)
    10. Categorical One-Hot Encoding
    11. StandardScaler numeric normalization
    12. SMOTE target resampling (if applicable)

    Returns (df_cleaned, applied_transformations_list, python_script_code).
    """
    df = df_raw.copy()
    applied_steps = []
    script_lines = [
        "# ==========================================================",
        "# Auto-Generated Domain-Aware Pre-ML Preprocessing Pipeline",
        "# AI Data Readiness Platform",
        "# ==========================================================",
        "import re",
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:",
        "    df = df_raw.copy()",
    ]

    approved_recs = [r for r in recommendations if r.get("is_approved", True)]
    rec_by_type = {}
    for r in approved_recs:
        rec_by_type.setdefault(r["issue_type"], []).append(r)

    # -------------------------------------------------------------
    # Step 1: Exact Deduplication
    # -------------------------------------------------------------
    if "duplicate_rows" in rec_by_type:
        before_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_count = before_len - len(df)
        applied_steps.append(f"Deduplication: Removed {dropped_count} exact duplicate records.")
        script_lines.append("    # Step 1: Remove exact duplicates")
        script_lines.append("    df = df.drop_duplicates().reset_index(drop=True)")

    # -------------------------------------------------------------
    # Step 2: Datetime Feature Engineering
    # -------------------------------------------------------------
    datetime_recs = rec_by_type.get("unparsed_datetime_feature", [])
    if datetime_recs:
        script_lines.append("    # Step 2: Extract chronological signals from date columns")
        for r in datetime_recs:
            col = r.get("column")
            if col and col in df.columns and col != target_col:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dt_parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
                    year_col = f"{col}_year"
                    month_col = f"{col}_month"
                    df[year_col] = dt_parsed.dt.year.fillna(dt_parsed.dt.year.median() if dt_parsed.dt.year.notna().any() else 2020)
                    df[month_col] = dt_parsed.dt.month.fillna(6)
                    df = df.drop(columns=[col])
                    applied_steps.append(f"Datetime Engineering: Parsed '{col}' into '{year_col}' and '{month_col}'.")
                    script_lines.append(f"    if '{col}' in df.columns:")
                    script_lines.append(f"        dt_p = pd.to_datetime(df['{col}'], format='mixed', errors='coerce')")
                    script_lines.append(f"        df['{year_col}'] = dt_p.dt.year.fillna(dt_p.dt.year.median())")
                    script_lines.append(f"        df['{month_col}'] = dt_p.dt.month.fillna(6)")
                    script_lines.append(f"        df = df.drop(columns=['{col}'])")
                except Exception:
                    pass

    # -------------------------------------------------------------
    # Step 3: Mixed Unit Disentanglement (e.g. Duration: '90 min', '2 Seasons')
    # -------------------------------------------------------------
    unit_recs = rec_by_type.get("unit_mixed_feature", [])
    if unit_recs:
        script_lines.append("    # Step 3: Parse mixed magnitude and unit strings")
        unit_regex = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\s*$")
        for r in unit_recs:
            col = r.get("column")
            if col and col in df.columns and col != target_col:
                try:
                    vals = []
                    units = []
                    for item in df[col].astype(str):
                        m = unit_regex.match(str(item).strip())
                        if m:
                            vals.append(float(m.group(1)))
                            units.append(m.group(2).lower() if m.group(2) else "unit")
                        else:
                            vals.append(np.nan)
                            units.append("unknown")

                    val_col = f"{col}_value"
                    unit_col = f"{col}_unit"
                    df[val_col] = pd.Series(vals, index=df.index).fillna(pd.Series(vals).median())
                    df[unit_col] = pd.Series(units, index=df.index)
                    df = df.drop(columns=[col])
                    applied_steps.append(f"Unit Disentanglement: Parsed '{col}' into numeric '{val_col}' and categorical '{unit_col}'.")
                    script_lines.append(f"    if '{col}' in df.columns:")
                    script_lines.append(f"        # Disentangle {col} numeric value and unit")
                    script_lines.append(f"        extracted = df['{col}'].astype(str).str.extract(r'(\\d+(?:\\.\\d+)?)\\s*([a-zA-Z%]+)?')")
                    script_lines.append(f"        df['{val_col}'] = pd.to_numeric(extracted[0], errors='coerce').fillna(0.0)")
                    script_lines.append(f"        df['{unit_col}'] = extracted[1].fillna('unit').str.lower()")
                    script_lines.append(f"        df = df.drop(columns=['{col}'])")
                except Exception:
                    pass

    # -------------------------------------------------------------
    # Step 4: Multi-Label Delimited Token Encoding
    # -------------------------------------------------------------
    multilabel_recs = rec_by_type.get("multilabel_delimited_text", [])
    if multilabel_recs:
        script_lines.append("    # Step 4: Multi-Hot encode delimited token lists")
        for r in multilabel_recs:
            col = r.get("column")
            if col and col in df.columns and col != target_col:
                try:
                    # Collect all individual tokens
                    all_tokens = []
                    for cell in df[col].dropna().astype(str):
                        for token in re.split(r"[,;|]", cell):
                            t_clean = token.strip()
                            if t_clean:
                                all_tokens.append(t_clean)

                    top_tokens = pd.Series(all_tokens).value_counts().head(12).index.tolist()
                    for token in top_tokens:
                        safe_token_name = f"{col}_{re.sub(r'[^a-zA-Z0-9]', '_', token).lower()}"
                        df[safe_token_name] = df[col].astype(str).apply(lambda x: 1.0 if token in str(x) else 0.0)

                    df = df.drop(columns=[col])
                    applied_steps.append(f"Multi-Label Encoding: Tokenized '{col}' into {len(top_tokens)} binary tag indicators.")
                    script_lines.append(f"    if '{col}' in df.columns:")
                    script_lines.append(f"        top_tokens = {top_tokens}")
                    script_lines.append(f"        for token in top_tokens:")
                    script_lines.append(f"            df[f'{col}_{{re.sub(r\"[^a-zA-Z0-9]\", \"_\", token).lower()}}'] = df['{col}'].astype(str).apply(lambda x: 1.0 if token in str(x) else 0.0)")
                    script_lines.append(f"        df = df.drop(columns=['{col}'])")
                except Exception:
                    pass

    # -------------------------------------------------------------
    # Step 5: Invalid Domain Values Clipping
    # -------------------------------------------------------------
    if "invalid_domain_values" in rec_by_type:
        script_lines.append("    # Step 5: Fix domain invalid negative values")
        for r in rec_by_type["invalid_domain_values"]:
            col = r.get("column")
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].clip(lower=0.0)
                applied_steps.append(f"Domain Validity: Clipped negative values in '{col}' to 0.0.")
                script_lines.append(f"    if '{col}' in df.columns: df['{col}'] = df['{col}'].clip(lower=0.0)")

    # -------------------------------------------------------------
    # Step 6: Drop Redundant / Collinear Columns
    # -------------------------------------------------------------
    cols_to_drop = []
    if "constant_feature" in rec_by_type:
        for r in rec_by_type["constant_feature"]:
            col = r.get("column")
            if col and col in df.columns and col != target_col:
                cols_to_drop.append(col)

    if "multicollinear_features" in rec_by_type:
        for r in rec_by_type["multicollinear_features"]:
            col = r.get("column")
            if col and col in df.columns and col != target_col:
                cols_to_drop.append(col)

    if "missing_values" in rec_by_type:
        for r in rec_by_type["missing_values"]:
            if "Drop" in r.get("method", ""):
                col = r.get("column")
                if col and col in df.columns and col != target_col:
                    cols_to_drop.append(col)

    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
        applied_steps.append(f"Dimensionality: Dropped redundant/collinear features: {', '.join(cols_to_drop)}.")
        script_lines.append(f"    # Step 6: Drop redundant features")
        script_lines.append(f"    df = df.drop(columns={cols_to_drop}, errors='ignore')")

    # -------------------------------------------------------------
    # Step 7: Semantic Missing Value Imputation
    # -------------------------------------------------------------
    if "missing_values" in rec_by_type:
        script_lines.append("    # Step 7: Semantic Missing value imputation")
        for r in rec_by_type["missing_values"]:
            col = r.get("column")
            method = r.get("method", "")
            if not col or col not in df.columns or "Drop" in method:
                continue

            # Entity / High-Cardinality text (e.g. director, actor, notes)
            if "Unknown" in method or df[col].dtype == object and df[col].nunique() > 20:
                has_col = f"has_{col}"
                df[has_col] = df[col].notna().astype(float)
                df[col] = df[col].fillna("Unknown")
                applied_steps.append(f"Imputation: Filled missing in entity '{col}' with 'Unknown' & added indicator '{has_col}'.")
                script_lines.append(f"    if '{col}' in df.columns:")
                script_lines.append(f"        df['has_{col}'] = df['{col}'].notna().astype(float)")
                script_lines.append(f"        df['{col}'] = df['{col}'].fillna('Unknown')")

            elif pd.api.types.is_numeric_dtype(df[col]):
                if "Median" in method:
                    fill_val = float(df[col].median())
                    df[col] = df[col].fillna(fill_val)
                    applied_steps.append(f"Imputation: Replaced missing in '{col}' with Median ({round(fill_val, 2)}).")
                    script_lines.append(f"    if '{col}' in df.columns: df['{col}'] = df['{col}'].fillna({round(fill_val, 4)})")
                else:
                    fill_val = float(df[col].mean())
                    df[col] = df[col].fillna(fill_val)
                    applied_steps.append(f"Imputation: Replaced missing in '{col}' with Mean ({round(fill_val, 2)}).")
                    script_lines.append(f"    if '{col}' in df.columns: df['{col}'] = df['{col}'].fillna({round(fill_val, 4)})")
            else:
                # Mode for low-cardinality nominal category
                mode_series = df[col].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
                df[col] = df[col].fillna(fill_val)
                applied_steps.append(f"Imputation: Replaced missing in '{col}' with Mode ('{fill_val}').")
                script_lines.append(f"    if '{col}' in df.columns: df['{col}'] = df['{col}'].fillna('{fill_val}')")

    # Safety pass for any residual missing cells
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median() if not np.isnan(df[col].median()) else 0.0)
            else:
                df[col] = df[col].fillna("Unknown")

    # -------------------------------------------------------------
    # Step 8: High Cardinality Binning & Free-Text Cleanup
    # -------------------------------------------------------------
    script_lines.append("    # Step 8: Categorical cardinality binning & ID filtering")
    for col in list(df.select_dtypes(include=['object', 'category']).columns):
        if col == target_col:
            continue
        nunique = df[col].nunique()
        # Drop unique ID or free-text prose columns (>50 categories or >40% uniqueness ratio)
        if nunique > 50 or (len(df) > 50 and (nunique / len(df)) > 0.40):
            df = df.drop(columns=[col])
            applied_steps.append(f"Dimensionality: Dropped non-predictive high-cardinality column '{col}' ({nunique} unique).")
            script_lines.append(f"    if '{col}' in df.columns: df = df.drop(columns=['{col}'])")
        elif nunique > 10:
            top_10 = df[col].value_counts().head(10).index.tolist()
            df[col] = df[col].apply(lambda x: str(x) if x in top_10 else "Other")
            applied_steps.append(f"Encoding: Grouped categories in '{col}' into Top 10 + 'Other'.")
            script_lines.append(f"    if '{col}' in df.columns:")
            script_lines.append(f"        top_10 = {top_10}")
            script_lines.append(f"        df['{col}'] = df['{col}'].apply(lambda x: str(x) if x in top_10 else 'Other')")

    # -------------------------------------------------------------
    # Step 9: Outlier Winsorization / Capping (Excluding Year/Age ranges)
    # -------------------------------------------------------------
    if "statistical_outliers" in rec_by_type:
        script_lines.append("    # Step 9: Outlier Winsorization / Capping")
        for r in rec_by_type["statistical_outliers"]:
            col = r.get("column")
            stats = r.get("stats_context", {})
            if col and col in df.columns and col != target_col and pd.api.types.is_numeric_dtype(df[col]):
                # Do NOT clip year columns (e.g. release_year)
                if "year" in str(col).lower():
                    continue
                lb = float(stats.get("lower_bound", df[col].quantile(0.01)))
                ub = float(stats.get("upper_bound", df[col].quantile(0.99)))
                df[col] = df[col].clip(lower=lb, upper=ub)
                applied_steps.append(f"Outlier Capping: Winsorized '{col}' bounds to [{lb}, {ub}].")
                script_lines.append(f"    if '{col}' in df.columns: df['{col}'] = df['{col}'].clip(lower={lb}, upper={ub})")

    # -------------------------------------------------------------
    # Step 10: Categorical One-Hot Encoding
    # -------------------------------------------------------------
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != target_col]
    if cat_cols:
        script_lines.append("    # Step 10: One-Hot Encoding")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        applied_steps.append(f"Encoding: Applied One-Hot Encoding to categorical features: {', '.join(cat_cols)}.")
        script_lines.append(f"    df = pd.get_dummies(df, columns={cat_cols}, drop_first=True, dtype=float)")

    # -------------------------------------------------------------
    # Step 11: Standard Feature Scaling (Numeric Features)
    # -------------------------------------------------------------
    num_feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if num_feature_cols:
        script_lines.append("    # Step 11: Standard Feature Scaling")
        scaler = StandardScaler()
        df[num_feature_cols] = scaler.fit_transform(df[num_feature_cols])
        applied_steps.append(f"Scaling: Standardized {len(num_feature_cols)} numeric features using StandardScaler.")
        script_lines.append("    scaler = StandardScaler()")
        script_lines.append(f"    df[{num_feature_cols}] = scaler.fit_transform(df[{num_feature_cols}])")

    # -------------------------------------------------------------
    # Step 12: Target Resampling (SMOTE if Classification)
    # -------------------------------------------------------------
    if "class_imbalance" in rec_by_type and problem_type == "classification" and target_col and target_col in df.columns:
        for r in rec_by_type["class_imbalance"]:
            if "SMOTE" in r.get("method", ""):
                try:
                    from imblearn.over_sampling import SMOTE
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    min_class_count = int(y.value_counts().min())
                    if min_class_count > 2 and len(X.columns) <= 100:
                        k_neigh = min(5, min_class_count - 1)
                        smote = SMOTE(k_neighbors=k_neigh, random_state=42)
                        X_res, y_res = smote.fit_resample(X, y)
                        df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
                        applied_steps.append(f"Resampling: Applied SMOTE oversampling on target '{target_col}'.")
                        script_lines.append("    # Step 12: SMOTE target resampling")
                        script_lines.append("    from imblearn.over_sampling import SMOTE")
                        script_lines.append(f"    X, y = df.drop(columns=['{target_col}']), df['{target_col}']")
                        script_lines.append(f"    df_res, y_res = SMOTE(k_neighbors={k_neigh}, random_state=42).fit_resample(X, y)")
                        script_lines.append(f"    df = pd.concat([pd.DataFrame(df_res, columns=X.columns), pd.Series(y_res, name='{target_col}')], axis=1)")
                except Exception:
                    pass

    script_lines.append("    return df")
    script_lines.append("")
    script_lines.append("# Run clean pipeline on raw dataset")
    script_lines.append("# cleaned_df = clean_data(pd.read_csv('your_raw_dataset.csv'))")

    script_code = "\n".join(script_lines)
    return df, applied_steps, script_code
