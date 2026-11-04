import re

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from app.engine.leakage import leakage_candidates
from app.engine.profiler import infer_column_type, name_has_token, name_tokens, count_duplicate_records

NON_NEGATIVE_KEYWORDS = {
    "age", "salary", "income", "price", "cost", "revenue", "count", "quantity", "qty",
    "amount", "hour", "fare", "distance", "tenure", "charge", "fee", "weight", "height",
    "duration", "population", "area", "sqft", "room", "bedroom", "bathroom",
}

TRANSFORMED_TYPES = {"datetime", "unit_mixed_numeric", "multilabel_text"}
TEXT_TYPES = {"categorical", "text", "person_name_or_text"}
HIGH_CARDINALITY_LIMIT = 50
RARE_GROUPING_LIMIT = 10
DISCRETE_LEVELS = 15
MAX_INVALID_NEGATIVE_SHARE = 0.05


STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "has", "have", "had", "not", "but",
    "you", "your", "our", "its", "his", "her", "their", "they", "them", "she", "him", "who", "what", "which", "will",
    "can", "all", "any", "one", "two", "of", "to", "in", "on", "at", "by", "is", "it", "as", "an", "or", "be", "a",
}


PERSON_HEADER_TOKENS = {
    "name", "names", "first", "last", "full", "middle", "given", "family", "surname", "sur", "firstname", "lastname",
    "fullname", "employee", "emp", "customer", "client", "person", "user", "passenger", "patient", "student", "member",
    "contact", "owner", "applicant", "candidate", "of",
}
PERSON_NAME_CORE = {"name", "names", "surname", "firstname", "lastname", "fullname"}
PERSON_VALUE_RE = re.compile(r"^[^\W\d_][^\W\d_'.\-]*(?:[ ,.'\-]+[^\W\d_][^\W\d_'.\-]*){0,4}\.?$")
WEAK_SIGNAL = 0.10
NO_SIGNAL = 0.05
SIGNAL_ALPHA = 0.01
MAX_SIGNAL_COLUMNS = 150


def pii_kind(col: str, clean: pd.Series) -> Optional[str]:
    """'email' / 'phone' for contact data, 'person_name' for a column that holds people's names."""
    from app.engine.normalize import is_email, is_phone
    strs = clean.astype(str).str.strip()
    if len(strs) < 5:
        return None
    sample = strs.head(2000)
    if sample.map(is_email).mean() >= 0.8:
        return "email"
    if sample.map(is_phone).mean() >= 0.8:
        return "phone"
    tokens = set(name_tokens(col))
    if tokens and tokens <= PERSON_HEADER_TOKENS and tokens & PERSON_NAME_CORE:
        if sample.map(lambda v: bool(PERSON_VALUE_RE.match(v))).mean() >= 0.9:
            return "person_name"
    return None


def date_format_signatures(clean: pd.Series) -> Dict[str, int]:
    """'2020/02/20' -> '9999/99/99', 'April 5, 2018' -> 'A 9, 9999'; counts per written format."""
    sig = clean.astype(str).str.strip().map(lambda v: re.sub(r"[A-Za-z]+", "A", re.sub(r"\d", "9", v)))
    return {str(k): int(c) for k, c in sig.value_counts().items()}


def min_support(n_rows: int) -> int:
    """Smallest number of rows a category / token needs before it is trusted as a feature value."""
    return max(5, int(np.ceil(0.01 * n_rows)))


def is_high_cardinality(nunique: int, total_rows: int) -> bool:
    return nunique > HIGH_CARDINALITY_LIMIT or (total_rows > 50 and nunique / total_rows > 0.40)


def code_prefix_signal(clean: pd.Series, total_rows: int) -> Optional[Dict[str, Any]]:
    """Alphanumeric codes ('C85', 'B96', 'SKU-123') whose letter prefix groups rows (deck, product family, ...)."""
    from app.engine.transforms import code_prefix
    strs = clean.astype(str)
    prefixes = strs.map(code_prefix)
    matched = prefixes.dropna()
    if len(strs) < 10 or len(matched) < 0.8 * len(strs):
        return None
    counts = matched.value_counts()
    keep = [str(p) for p, c in counts.items() if c >= min_support(total_rows)][:15]
    if len(keep) < 2 or counts.size >= 0.5 * clean.nunique():
        return None
    return {"keep": keep, "prefix_count": int(counts.size), "coverage": round(float(counts[keep].sum() / len(strs)), 3)}


MIN_WORD_ASSOCIATION = 0.1


def flag_association(flag: pd.Series, y: pd.Series, problem_type: Optional[str]) -> float:
    """Bias-corrected Cramér's V (classification) or |r| (regression) between a 0/1 flag and the target."""
    data = pd.DataFrame({"f": flag.astype(int), "y": y}).dropna()
    if len(data) < 10 or data["f"].nunique() < 2:
        return 0.0
    if problem_type == "regression":
        yn = pd.to_numeric(data["y"], errors="coerce")
        ok = yn.notna()
        r = np.corrcoef(data["f"][ok], yn[ok])[0, 1] if ok.sum() > 2 and yn[ok].nunique() > 1 else 0.0
        return float(abs(r)) if np.isfinite(r) else 0.0
    table = pd.crosstab(data["f"], data["y"].astype(str)).to_numpy(dtype=float)
    n = table.sum()
    r_, k_ = table.shape
    if min(r_, k_) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    phi2 = max(0.0, chi2 / n - (k_ - 1) * (r_ - 1) / (n - 1))
    r_c, k_c = r_ - (r_ - 1) ** 2 / (n - 1), k_ - (k_ - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def word_token_signal(clean: pd.Series, total_rows: int, target: Optional[pd.Series] = None,
                      problem_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Frequent words inside short free text (titles in names, model words in product names). With a target,
    only words whose presence is related to it are kept, so incidental words (common first names) are not."""
    from app.engine.transforms import word_set
    strs = clean.astype(str)
    if len(strs) < 20 or strs.str.len().median() > 80:
        return None
    sets = strs.map(word_set)
    doc_freq: Dict[str, int] = {}
    for s in sets:
        for w in s:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    support = min_support(total_rows)
    words = sorted(
        (w for w, c in doc_freq.items()
         if c >= support and c <= 0.6 * len(strs) and w not in STOP_WORDS and len(w) >= 2),
        key=lambda w: -doc_freq[w],
    )[:40]
    if target is not None and problem_type in ("classification", "regression"):
        y = target.loc[sets.index]
        # Strength floor plus a 1% significance floor, Bonferroni-corrected for the number of words tested,
        # so words that match the target by chance (common first names) drop out.
        from scipy.stats import chi2
        critical = float(chi2.ppf(1 - 0.01 / max(len(words), 1), df=1))
        floor = max(MIN_WORD_ASSOCIATION, float(np.sqrt(critical / len(sets))))
        scored = {w: flag_association(sets.map(lambda s, w=w: w in s), y, problem_type) for w in words}
        words = sorted((w for w in words if scored[w] >= floor), key=lambda w: -scored[w])
    words = words[:15]
    if not words:
        return None
    covered = float(sets.map(lambda s: bool(s & set(words))).mean())
    if covered < 0.2:
        return None
    return {"words": words, "coverage": round(covered, 3)}


def detect_issues(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scans the DataFrame for data quality issues and semantic feature engineering opportunities.
    Returns a list of structured issue dictionaries.
    """
    issues: List[Dict[str, Any]] = []
    total_rows, _ = df.shape
    if total_rows == 0:
        return issues

    col_types = {str(c): infer_column_type(df[c], str(c)) for c in df.columns}

    # 1. Duplicate records (ignoring unique identifier columns)
    dup_count = count_duplicate_records(df)
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows * 100), 2)
        severity = "high" if dup_pct > 5.0 else ("medium" if dup_pct > 1.0 else "low")
        issues.append({
            "issue_type": "duplicate_rows",
            "column": None,
            "severity": severity,
            "title": f"{dup_count} Duplicate Rows ({dup_pct}%)",
            "details": {"duplicate_count": dup_count, "duplicate_pct": dup_pct, "total_rows": total_rows},
        })

    # 2. Missing target labels
    if target_col and target_col in df.columns:
        missing_target = int(df[target_col].isna().sum())
        if missing_target > 0:
            pct = round(missing_target / total_rows * 100, 2)
            issues.append({
                "issue_type": "missing_target",
                "column": target_col,
                "severity": "critical" if pct > 20 else "high",
                "title": f"{missing_target} Rows Missing the Target '{target_col}' ({pct}%)",
                "details": {"column": target_col, "missing_count": missing_target, "missing_pct": pct},
            })

    categorical_to_encode: List[str] = []

    for col in df.columns:
        col = str(col)
        series = df[col]
        col_type = col_types[col]
        is_target = col == target_col
        if pd.api.types.is_float_dtype(series):
            clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            clean = series.dropna()
        nunique = int(clean.nunique())

        # 3. Identifier columns carry no predictive signal
        if col_type == "id" and not is_target:
            issues.append({
                "issue_type": "identifier_column",
                "column": col,
                "severity": "low",
                "title": f"Unique Identifier Column '{col}'",
                "details": {"column": col, "unique_count": nunique},
            })
            continue

        # 3b. Personal data (emails, phone numbers, people's names): no generalisable signal, privacy risk
        if not is_target and not pd.api.types.is_numeric_dtype(series):
            kind = pii_kind(col, clean)
            if kind:
                # Words inside people's names (titles) only earn a place as features when they relate to a target.
                if kind == "person_name" and target_col:
                    words = word_token_signal(clean, total_rows,
                                              df[target_col] if target_col and target_col in df.columns else None, problem_type)
                    if words and words["words"]:
                        issues.append({
                            "issue_type": "text_token_features", "column": col, "severity": "info",
                            "title": f"Frequent words in '{col}' ({', '.join(words['words'][:4])}) can become features",
                            "details": {"column": col, **words, "sample_values": []},
                        })
                label = {"email": "email addresses", "phone": "phone numbers", "person_name": "people's names"}[kind]
                # Contact data where a handful of values cover hundreds of rows is test / dummy data, not real contacts.
                dummy = kind in ("email", "phone") and len(clean) >= 20 and nunique <= max(5, 0.01 * len(clean))
                issues.append({
                    "issue_type": "personal_data",
                    "column": col,
                    "severity": "high" if dummy else "medium",
                    "title": f"Personal Data in '{col}' ({label}, {nunique} distinct"
                             + (f" shared by {len(clean)} rows: looks like dummy/test values)" if dummy else ")"),
                    "details": {"column": col, "kind": kind, "unique_count": nunique, "dummy_values": dummy,
                                "missing_pct": round(float(series.isna().mean() * 100), 2)},
                })
                continue

        # 4. Missing values (features only; target handled above)
        missing_cnt = int(series.isna().sum())
        if missing_cnt > 0 and not is_target:
            missing_pct = round((missing_cnt / total_rows * 100), 2)
            if missing_pct > 50.0:
                severity = "critical"
            elif missing_pct > 20.0:
                severity = "high"
            elif missing_pct > 5.0:
                severity = "medium"
            else:
                severity = "low"
            issues.append({
                "issue_type": "missing_values",
                "column": col,
                "severity": severity,
                "title": f"Missing Values in '{col}' ({missing_pct}%)",
                "details": {
                    "column": col,
                    "missing_count": missing_cnt,
                    "missing_pct": missing_pct,
                    "inferred_type": col_type,
                    "cardinality": nunique,
                },
            })

        if is_target:
            continue

        # 5. Constant / zero-variance features (any dtype)
        if nunique == 1:
            issues.append({
                "issue_type": "constant_feature",
                "column": col,
                "severity": "medium",
                "title": f"Constant / Zero Variance Feature '{col}'",
                "details": {"column": col, "unique_values": nunique},
            })
            continue

        # 6. Unparsed datetime strings
        if (
            col_type == "datetime"
            and not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_datetime64_any_dtype(series)
        ):
            issues.append({
                "issue_type": "unparsed_datetime_feature",
                "column": col,
                "severity": "medium",
                "title": f"Raw Date/Timestamp String in '{col}'",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(3).tolist()]},
            })
            formats = {k: c for k, c in date_format_signatures(clean).items() if c >= max(2, 0.02 * len(clean))}
            if len(formats) >= 2:
                examples = [str(clean[clean.astype(str).str.strip().map(
                    lambda v, k=k: re.sub(r"[A-Za-z]+", "A", re.sub(r"\d", "9", v)) == k)].iloc[0]) for k in list(formats)[:4]]
                issues.append({
                    "issue_type": "mixed_date_formats",
                    "column": col,
                    "severity": "high" if len(formats) >= 3 else "medium",
                    "title": f"{len(formats)} Different Date Formats in '{col}' (e.g. {', '.join(repr(e) for e in examples[:3])})",
                    "details": {"column": col, "format_count": len(formats), "examples": examples,
                                "count": int(len(clean) - max(formats.values())),
                                "pct": round((len(clean) - max(formats.values())) / total_rows * 100, 2)},
                })

        # 7. Mixed unit strings ('90 min', '2 Seasons')
        if col_type == "unit_mixed_numeric":
            issues.append({
                "issue_type": "unit_mixed_feature",
                "column": col,
                "severity": "high",
                "title": f"Mixed Value & Units in '{col}' (e.g. '90 min', '1 Season')",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(4).tolist()]},
            })

        # 8. Delimited multi-label lists
        if col_type == "multilabel_text":
            issues.append({
                "issue_type": "multilabel_delimited_text",
                "column": col,
                "severity": "medium",
                "title": f"Delimited Multi-Label List in '{col}' (e.g. genres, tags)",
                "details": {"column": col, "sample_values": [str(v) for v in clean.head(3).tolist()]},
            })

        # 8b. Spelling / case variants of the same category ('Delhi', 'DELHI', 'delhi')
        if col_type in TEXT_TYPES and not pd.api.types.is_numeric_dtype(series) and nunique <= 500:
            from app.engine.normalize import category_mapping
            mapping = category_mapping(clean)
            if mapping:
                variant_cells = int(clean.astype(str).isin(set(mapping)).sum())
                groups = len(set(mapping.values()))
                issues.append({
                    "issue_type": "inconsistent_categories",
                    "column": col,
                    "severity": "medium" if variant_cells / total_rows > 0.05 else "low",
                    "title": f"Inconsistent Spelling in '{col}': {groups} value(s) written {len(mapping) + groups} ways "
                             f"({variant_cells} cells, e.g. {', '.join(repr(k) for k in list(mapping)[:2])})",
                    "details": {"column": col, "mapping": dict(list(mapping.items())[:50]), "count": variant_cells,
                                "pct": round(variant_cells / total_rows * 100, 2)},
                })

        # 9. Text / categorical cardinality handling
        if col_type in TEXT_TYPES and not pd.api.types.is_numeric_dtype(series):
            supported = int((clean.astype(str).value_counts() >= min_support(total_rows)).sum())
            if is_high_cardinality(nunique, total_rows) or (nunique > 2 and supported < 2):
                prefix = code_prefix_signal(clean, total_rows)
                if prefix:
                    issues.append({
                        "issue_type": "code_prefix_feature",
                        "column": col,
                        "severity": "info",
                        "title": f"Codes in '{col}' share {len(prefix['keep'])} letter prefixes worth keeping",
                        "details": {"column": col, **prefix, "sample_values": [str(v) for v in clean.head(3).tolist()]},
                    })
                words = word_token_signal(clean, total_rows,
                                          df[target_col] if target_col and target_col in df.columns else None, problem_type)
                if words and prefix:
                    # Words that are just the code prefix would duplicate the prefix category.
                    prefixes = {p.lower() for p in prefix["keep"]}
                    words["words"] = [w for w in words["words"] if w not in prefixes]
                if words and words["words"]:
                    issues.append({
                        "issue_type": "text_token_features",
                        "column": col,
                        "severity": "info",
                        "title": f"Frequent words in '{col}' ({', '.join(words['words'][:4])}) can become features",
                        "details": {"column": col, **words, "sample_values": [str(v) for v in clean.head(3).tolist()]},
                    })
                issues.append({
                    "issue_type": "high_cardinality_text",
                    "column": col,
                    "severity": "low",
                    "title": f"High-Cardinality Free Text in '{col}' ({nunique} unique values)",
                    "details": {"column": col, "unique_count": nunique, "unique_pct": round(nunique / total_rows * 100, 2),
                                "supported_categories": supported, "min_support": min_support(total_rows)},
                })
                continue
            if nunique > RARE_GROUPING_LIMIT:
                issues.append({
                    "issue_type": "rare_categories",
                    "column": col,
                    "severity": "low",
                    "title": f"Many Rare Categories in '{col}' ({nunique} unique values)",
                    "details": {"column": col, "unique_count": nunique, "keep_top": RARE_GROUPING_LIMIT},
                })
            categorical_to_encode.append(col)
        elif col_type == "boolean" and not pd.api.types.is_numeric_dtype(series):
            categorical_to_encode.append(col)

        if col_type in TRANSFORMED_TYPES or not pd.api.types.is_numeric_dtype(series) or col_type == "boolean":
            continue

        # 10. Domain validity (negative values where non-negative expected)
        if name_has_token(col, NON_NEGATIVE_KEYWORDS):
            neg_count = int((clean < 0).sum())
            # Entry errors are rare; a column where negatives are common (net income, price change) means them.
            if 0 < neg_count <= MAX_INVALID_NEGATIVE_SHARE * len(clean):
                issues.append({
                    "issue_type": "invalid_domain_values",
                    "column": col,
                    "severity": "high",
                    "title": f"Invalid Negative Values in '{col}' ({neg_count} rows)",
                    "details": {
                        "column": col,
                        "negative_count": neg_count,
                        "negative_pct": round(neg_count / len(clean) * 100, 2),
                        "min_value": float(clean.min()),
                    },
                })

        # 11. Statistical outliers (not for year columns, low-cardinality codes, or small whole-number counts
        #     like siblings / rooms, whose rare high levels are real values rather than errors)
        discrete = bool((clean % 1 == 0).all()) and nunique <= DISCRETE_LEVELS
        if col_type == "numeric" and len(clean) > 10 and not discrete:
            q25, q75 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                # Extreme fence (3x IQR), never beyond the observed range: only these are capped by default.
                extreme_lower = max(float(q25 - 3 * iqr), float(clean.min()))
                extreme_upper = min(float(q75 + 3 * iqr), float(clean.max()))
                outlier_cnt = int(((clean < lower_bound) | (clean > upper_bound)).sum())
                extreme_cnt = int(((clean < extreme_lower) | (clean > extreme_upper)).sum())
                outlier_pct = round(outlier_cnt / len(clean) * 100, 2)
                if outlier_cnt > 0 and outlier_pct > 1.0:
                    severity = "high" if outlier_pct > 10.0 else ("medium" if outlier_pct > 3.0 else "low")
                    if extreme_cnt == 0:
                        severity = "low"
                    issues.append({
                        "issue_type": "statistical_outliers",
                        "column": col,
                        "severity": severity,
                        "title": f"Outliers in '{col}' ({outlier_cnt} rows, {outlier_pct}%)",
                        "details": {
                            "column": col,
                            "outlier_count": outlier_cnt,
                            "outlier_pct": outlier_pct,
                            "lower_bound": round(float(lower_bound), 3),
                            "upper_bound": round(float(upper_bound), 3),
                            "extreme_count": extreme_cnt,
                            "extreme_lower": round(extreme_lower, 3),
                            "extreme_upper": round(extreme_upper, 3),
                            "skewness": round(float(clean.skew()), 3) if len(clean) > 2 else 0.0,
                        },
                    })

    # 11b. Target leakage: a single feature that predicts the target (almost) perfectly
    if target_col and target_col in df.columns and problem_type in ("classification", "regression"):
        skip = {c for c, t in col_types.items() if t == "id"}
        for leak in leakage_candidates(df, target_col, problem_type, skip=skip):
            issues.append({
                "issue_type": "target_leakage",
                "column": leak["column"],
                "severity": "critical",
                "title": f"Possible Target Leakage: '{leak['column']}' predicts '{target_col}' almost perfectly ({leak['method']} {leak['score']})",
                "details": {"column": leak["column"], "score": leak["score"], "method": leak["method"], "target": target_col},
            })

    # 11c. Predictive signal: does ANY usable feature relate to the target? Clean-looking but random data
    #      (every column independent of the target) must not be called ML-ready.
    if target_col and target_col in df.columns and problem_type in ("classification", "regression") and total_rows >= 30:
        unusable = {i["column"] for i in issues if i["issue_type"] in (
            "identifier_column", "personal_data", "constant_feature", "target_leakage")}
        from app.engine.stats import association_pvalue, target_association
        scores, pvalues = {}, {}
        for c in [str(c) for c in df.columns if str(c) != target_col and str(c) not in unusable][:MAX_SIGNAL_COLUMNS]:
            a = target_association(df, c, target_col, problem_type)
            if a is not None:
                scores[c] = a
                pvalues[c] = association_pvalue(df, c, target_col, problem_type)
        if scores:
            # A relationship counts only if it is both statistically real (1% level, Bonferroni-corrected for the number
            # of features tested, so chance matches on random data do not count) and not negligible in strength.
            alpha = SIGNAL_ALPHA / len(scores)
            real = {c: a for c, a in scores.items() if pvalues.get(c) is not None and pvalues[c] < alpha and a >= NO_SIGNAL}
            best = max(real or scores, key=(real or scores).get)
            strongest = scores[best]
            if not real or real[best] < WEAK_SIGNAL:
                none = not real
                issues.append({
                    "issue_type": "weak_target_signal",
                    "column": target_col,
                    "severity": "high" if none else "medium",
                    "title": (f"No Feature Is Related to '{target_col}'" if none else f"Very Weak Signal for '{target_col}'")
                             + f" (strongest: '{best}', association {round(strongest, 3)}"
                             + (", not statistically significant)" if none else ")"),
                    "details": {"column": target_col, "max_association": round(min(strongest, NO_SIGNAL) if none else strongest, 4),
                                "observed_max_association": round(strongest, 4), "best_feature": best,
                                "features_checked": len(scores), "significant_features": len(real), "no_signal": none,
                                "top": dict(sorted(scores.items(), key=lambda kv: -kv[1])[:5])},
                })

    # 11d. Clustering: are there groups at all? (real data vs the same data with columns shuffled independently)
    if problem_type == "clustering" and total_rows >= 30:
        from app.engine.unsupervised import WEAK_GAP, USEFUL_GAP, cluster_structure
        unusable = {i["column"] for i in issues if i["issue_type"] in (
            "identifier_column", "personal_data", "constant_feature", "high_cardinality_text")}
        structure = cluster_structure(df.drop(columns=[c for c in unusable if c in df.columns]))
        if structure and structure["gap"] < USEFUL_GAP:
            none = structure["gap"] < WEAK_GAP
            issues.append({
                "issue_type": "weak_cluster_structure",
                "column": None,
                "severity": "high" if none else "medium",
                "title": ("No Natural Groups in the Data" if none else "Only Weak Group Structure")
                         + f" (silhouette {structure['silhouette']} vs {structure['null_silhouette']} with columns shuffled)",
                "details": {**structure, "no_signal": none},
            })

    # 12. Categorical encoding requirement
    if categorical_to_encode:
        issues.append({
            "issue_type": "categorical_encoding",
            "column": None,
            "severity": "info",
            "title": f"{len(categorical_to_encode)} Categorical Feature(s) Need Numeric Encoding",
            "details": {"columns": categorical_to_encode},
        })

    # 13. Multicollinearity among numeric features
    numeric_cols = [
        str(c) for c in df.select_dtypes(include=[np.number]).columns
        if col_types[str(c)] in ("numeric", "categorical") and str(c) != target_col and df[c].nunique() > 1
    ]
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                r_val = float(corr_matrix.loc[col1, col2])
                if not np.isnan(r_val) and r_val > 0.85:
                    high_corr_pairs.append({"col1": col1, "col2": col2, "correlation": round(r_val, 3)})
        high_corr_pairs.sort(key=lambda p: p["correlation"], reverse=True)
        already_dropped = set()
        for pair in high_corr_pairs[:5]:
            if pair["col1"] in already_dropped or pair["col2"] in already_dropped:
                continue
            already_dropped.add(pair["col2"])
            issues.append({
                "issue_type": "multicollinear_features",
                "column": f"{pair['col1']} & {pair['col2']}",
                "severity": "medium",
                "title": f"High Correlation ({pair['correlation']}) between '{pair['col1']}' and '{pair['col2']}'",
                "details": pair,
            })

    # 14. Target class imbalance (binary and multi-class)
    if target_col and target_col in df.columns and problem_type == "classification":
        counts = df[target_col].dropna().value_counts()
        if len(counts) >= 2:
            majority_ratio = float(counts.iloc[0] / counts.sum())
            minority_ratio = float(counts.iloc[-1] / counts.sum())
            imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])
            ratio_str = f"{round(majority_ratio * 100, 1)}% vs {round(minority_ratio * 100, 1)}%"
            if imbalance_ratio >= 3.0:
                if majority_ratio > 0.90 or imbalance_ratio >= 10:
                    severity = "critical"
                elif imbalance_ratio >= 5:
                    severity = "high"
                else:
                    severity = "medium"
                issues.append({
                    "issue_type": "class_imbalance",
                    "column": target_col,
                    "severity": severity,
                    "title": f"Target Class Imbalance on '{target_col}' ({ratio_str})",
                    "details": {
                        "column": target_col,
                        "majority_class": str(counts.index[0]),
                        "majority_ratio": round(majority_ratio, 3),
                        "minority_class": str(counts.index[-1]),
                        "minority_ratio": round(minority_ratio, 3),
                        "imbalance_ratio": round(imbalance_ratio, 2),
                        "n_classes": int(len(counts)),
                        "ratio_display": ratio_str,
                    },
                })

    return issues
