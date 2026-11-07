import json
import logging
import time
from typing import Any, Dict, List

from app.core.config import settings

logger = logging.getLogger(__name__)


def _samples(stats: Dict[str, Any]) -> str:
    samples = stats.get("sample_values", [])
    return f" (e.g. {', '.join(str(s) for s in samples[:2])})" if samples else ""


def generate_fallback_explanation(rec: Dict[str, Any]) -> str:
    """Deterministic explanation grounded in the computed statistics."""
    stats = rec.get("stats_context", {})
    key = stats.get("method_key", "")
    col = rec.get("column") or ""

    if key == "drop_duplicates":
        return (
            f"The dataset contains {stats.get('duplicate_count', 0)} duplicate records ({stats.get('duplicate_pct', 0.0)}% of rows), "
            f"ignoring unique ID columns. Repeated observations bias the loss toward those rows and can leak between "
            f"train and validation splits; removing them keeps evaluation honest."
        )
    if key == "drop_missing_target":
        return (
            f"{stats.get('missing_count', 0)} rows ({stats.get('missing_pct', 0.0)}%) have no value for the target '{col}'. "
            f"Imputing a label would invent ground truth, so these rows cannot be used for supervised training and are removed."
        )
    if key == "drop_identifier":
        return (
            f"Column '{col}' is unique for every row ({stats.get('unique_count', 0)} distinct values). Identifiers carry no "
            f"generalizable signal and let models memorize rows, so the column is removed before modelling."
        )
    if key == "drop_column_missing":
        return (
            f"Column '{col}' is {stats.get('missing_pct', 0.0)}% missing, above the 70% viability threshold. Imputing most of a "
            f"column creates synthetic values, so dropping it is safer. It is off by default because it is destructive."
        )
    if key == "auto_prep":
        if stats.get("kind") == "padded_text":
            return (
                f"{stats.get('count', 0)} value(s) in '{col}' carry leading or trailing spaces (e.g. "
                f"{', '.join(repr(e) for e in stats.get('examples', [])[:2])}). ' grace ' and 'grace' would otherwise count as "
                f"different values, so the spaces are trimmed before profiling."
            )
        if stats.get("kind") == "disguised_missing":
            return (
                f"{stats.get('count', 0)} cell(s) in '{col}' ({stats.get('pct', 0.0)}%) hold placeholder text such as "
                f"{', '.join(repr(e) for e in stats.get('examples', [])[:3])} instead of a value. They are counted as missing "
                f"before profiling, so the health score and the fixes see the real gaps."
            )
        return (
            f"'{col}' holds numbers, but {stats.get('count', 0)} value(s) are written as text "
            f"({', '.join(repr(e) for e in stats.get('examples', [])[:3])}). They are converted to numbers before profiling, "
            f"so the column is treated as numeric instead of as a set of fake categories."
        )
    if key == "merge_variants":
        return (
            f"{stats.get('count', 0)} cell(s) in '{col}' write the same value differently (case or spacing, e.g. "
            f"{', '.join(repr(k) for k in list((stats.get('mapping') or {}))[:2])}). Without merging, each spelling becomes "
            f"its own category and splits the signal, so every variant is mapped to its most common clean form."
        )
    if key == "drop_personal_data":
        kind = {"email": "email addresses", "phone": "phone numbers", "person_name": "people's names"}.get(stats.get("kind"), "personal data")
        return (
            f"'{col}' contains {kind}. Identifying details do not generalise to new people (a model can only memorise them) "
            f"and keeping them is a privacy risk, so the column is removed. Any recurring words that do carry signal are "
            f"extracted first."
        )
    if key == "missing_indicator":
        return (
            f"'{col}' is {stats.get('missing_pct', 0.0)}% blank. Filling that many cells would invent most of the column, but "
            f"whether a value was recorded is often informative on its own, so the column is replaced by a 0/1 'has_{col}' flag."
        )
    if key in ("impute_median", "impute_mean"):
        skew = stats.get("skewness", 0.0)
        flag = (f" A '{col}_was_missing' flag records exactly which rows were filled, so the filled value never "
                f"masquerades as a real measurement.") if stats.get("add_flag") else ""
        if key == "impute_median":
            grouped = (" When another column (e.g. a class or category) clearly explains it, the median of the row's group is "
                       "used instead of one global value, which keeps natural variation.") if stats.get("group_by") == "auto" else ""
            return (
                f"Numeric column '{col}' has {stats.get('missing_pct', 0.0)}% missing values (skewness {skew}). The median is "
                f"robust to extremes.{grouped}{flag}"
            )
        return (
            f"Numeric column '{col}' has {stats.get('missing_pct', 0.0)}% missing values and an approximately symmetric "
            f"distribution (skewness {skew}), so the mean preserves the column's central tendency.{flag}"
        )
    if key == "extract_prefix":
        return (
            f"'{col}' holds codes{_samples(stats)} that are too varied to use directly, but their letter prefix groups rows "
            f"({', '.join(str(k) for k in stats.get('keep', [])[:5])}). That prefix is kept as a category before the raw code is dropped."
        )
    if key == "word_flags":
        return (
            f"'{col}' is free text{_samples(stats)}, but some words recur across many rows "
            f"({', '.join(str(w) for w in stats.get('words', [])[:5])}). Each becomes a 0/1 feature before the raw text is dropped."
        )
    if key == "impute_mode":
        return (
            f"'{col}' is a low-cardinality categorical feature ({stats.get('cardinality', 0)} categories) with "
            f"{stats.get('missing_pct', 0.0)}% missing entries. Filling with the most frequent category keeps the column's "
            f"existing category set intact."
        )
    if key == "impute_unknown":
        flag = f" and a has_{col} flag records whether a value was present" if stats.get("add_flag", True) else ""
        return (
            f"'{col}' ({stats.get('cardinality', 0)} distinct values) has {stats.get('missing_pct', 0.0)}% missing. Assigning the "
            f"most common value would inject false signal, so missing entries become an explicit 'Unknown' category{flag}."
        )
    if key == "parse_datetime":
        return (
            f"Column '{col}' contains date strings{_samples(stats)}. Treating dates as categories destroys their ordering; "
            f"extracting year, month and day gives models numeric temporal signals."
        )
    if key == "split_units":
        return (
            f"Column '{col}' mixes a number with a unit label{_samples(stats)}. Splitting it into a numeric value and a unit "
            f"category keeps magnitudes comparable while preserving what the unit means."
        )
    if key == "multi_hot":
        return (
            f"Column '{col}' holds delimited lists{_samples(stats)}. Treating each combination as one category fragments the "
            f"data; one binary indicator per frequent token captures each tag independently."
        )
    if key == "replace_negatives_median":
        return (
            f"'{col}' has {stats.get('negative_count', 0)} negative values (minimum {stats.get('min_value', 0.0)}) in a "
            f"quantity that cannot be negative. These are data-entry errors, so they are treated as blanks and filled (and "
            f"flagged) together with the column's other gaps, rather than clipped to an artificial 0."
        )
    if key == "cap_outliers":
        if stats.get("extreme_count"):
            return (
                f"{stats.get('extreme_count', 0)} value(s) in '{col}' lie beyond the extreme 3x IQR fence "
                f"[{stats.get('lower_bound', 0.0)}, {stats.get('upper_bound', 0.0)}]. Only those are capped (rows are kept), and a "
                f"'{col}_was_capped' flag marks them so the repeated boundary value stays explainable."
            )
        return (
            f"{stats.get('outlier_count', 0)} values ({stats.get('outlier_pct', 0.0)}%) in '{col}' are beyond 1.5x IQR but none are "
            f"extreme. Long tails like these are usually real (prices, incomes, counts), so they are kept; capping is optional."
        )
    if key == "drop_collinear":
        return (
            f"'{stats.get('col1')}' and '{stats.get('col2')}' are strongly correlated (Pearson r = {stats.get('correlation', 0.0)}). "
            f"Keeping both inflates variance in linear models; dropping '{stats.get('col2')}' removes the redundancy. "
            f"Off by default because tree models are usually unaffected."
        )
    if key == "drop_constant":
        return f"Column '{col}' has the same value in every row, so it carries no predictive signal and is removed."
    if key == "drop_leakage":
        return (
            f"On its own, '{col}' predicts '{stats.get('target_column')}' almost perfectly ({stats.get('leak_method')} "
            f"{stats.get('score')}). Real predictors are rarely that clean: this usually means the column is recorded after "
            f"the outcome or encodes it. Keeping it would make every model look excellent and fail in production."
        )
    if key == "drop_high_cardinality":
        return (
            f"'{col}' has {stats.get('unique_count', 0)} distinct values ({stats.get('unique_pct', 0.0)}% of rows), so it behaves "
            f"like free text. One-hot encoding it would create hundreds of sparse columns, so it is removed."
        )
    if key == "group_rare":
        return (
            f"'{col}' has {stats.get('unique_count', 0)} categories. Up to {stats.get('keep_top', 10)} frequent values that each "
            f"cover enough rows are kept and the rest become 'Other', so no indicator column describes just a handful of rows."
        )
    if key == "one_hot_encode":
        cols = stats.get("columns", [])
        return (
            f"{len(cols)} text column(s) ({', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}) must be numeric for most models. "
            f"One-hot encoding creates one indicator per category using a fixed category list, so new data is encoded the same way."
        )
    if key in ("smote_class_weights", "class_weights"):
        ratio = stats.get("ratio_display", "")
        if key == "smote_class_weights":
            return (
                f"The target '{col}' is severely imbalanced ({ratio}). SMOTE synthesizes minority examples and balanced class "
                f"weights raise the cost of minority errors. Both are applied only inside training folds, so the exported data "
                f"and validation scores stay free of synthetic rows."
            )
        return (
            f"The target '{col}' is moderately imbalanced ({ratio}). Balanced class weights penalize minority-class errors more "
            f"heavily during training without altering the dataset."
        )
    return f"Recommended remediation '{rec.get('method')}' addresses data quality defects in '{col}'."


def _apply_fallback(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in recommendations:
        rec["explanation_text"] = generate_fallback_explanation(rec)
        rec["explanation_source"] = "statistical"
    return recommendations


EXPLAIN_SYSTEM = (
    "You are a Senior Data Scientist explaining pre-ML data cleaning decisions.\n"
    "For each item, write a 2-3 sentence technical justification of WHY the chosen method fits the provided statistics.\n"
    "RULES:\n"
    "1. Cite the exact numbers in stats_context.\n"
    "2. Never invent numbers that are not in the input.\n"
    "3. Explain the method that was chosen; do not recommend a different one.\n"
    "4. Return ONLY JSON: {\"items\": [{\"index\": int, \"explanation\": string}]}."
)


def _explanation_payload(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "index": idx,
            "issue_type": r["issue_type"],
            "column": r.get("column"),
            "method": r["method"],
            "stats_context": {k: v for k, v in r.get("stats_context", {}).items() if k not in ("method_key", "sample_values", "agent_note")},
        }
        for idx, r in enumerate(recommendations)
    ]


def _merge_explanations(recommendations: List[Dict[str, Any]], explanation_map: Dict[int, str]) -> List[Dict[str, Any]]:
    for idx, rec in enumerate(recommendations):
        if explanation_map.get(idx):
            rec["explanation_text"] = str(explanation_map[idx])
            rec["explanation_source"] = "ai"
        else:
            rec["explanation_text"] = generate_fallback_explanation(rec)
            rec["explanation_source"] = "statistical"
    return recommendations


def _explain_with_openrouter(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from app.agents import llm
    try:
        parsed, reply = llm.chat_json(
            EXPLAIN_SYSTEM,
            "Data Payload:\n" + json.dumps(_explanation_payload(recommendations), default=str),
            deadline=time.monotonic() + min(settings.AGENT_RUN_BUDGET_SEC, 120),
            max_tokens=3000,
        )
        items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
        explanation_map = {
            int(item["index"]): item["explanation"]
            for item in items or []
            if isinstance(item, dict) and "index" in item and item.get("explanation")
        }
        logger.info(f"Generated explanations using {reply.model}.")
        return _merge_explanations(recommendations, explanation_map)
    except Exception as e:
        logger.warning(f"OpenRouter explanation generation failed ({type(e).__name__}). Using statistical explanations.")
        return _apply_fallback(recommendations)

def explain_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage 3: grounded LLM explanations via OpenRouter, falling back to deterministic statistical explanations when no
    key is configured or the model fails. Each recommendation gets explanation_source = "ai" | "statistical".
    """
    if not recommendations:
        return []
    from app.agents import llm
    if llm.is_configured():
        return _explain_with_openrouter(recommendations)
    return _apply_fallback(recommendations)
