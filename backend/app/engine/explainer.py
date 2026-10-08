import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

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
    if key in ("impute_median", "impute_mean"):
        skew = stats.get("skewness", 0.0)
        if key == "impute_median":
            return (
                f"Numeric column '{col}' has {stats.get('missing_pct', 0.0)}% missing values and a skewed distribution "
                f"(skewness {skew}). The median is robust to the long tail, so it fills gaps without being pulled by extremes."
            )
        return (
            f"Numeric column '{col}' has {stats.get('missing_pct', 0.0)}% missing values and an approximately symmetric "
            f"distribution (skewness {skew}), so the mean preserves the column's central tendency."
        )
    if key == "impute_mode":
        return (
            f"'{col}' is a low-cardinality categorical feature ({stats.get('cardinality', 0)} categories) with "
            f"{stats.get('missing_pct', 0.0)}% missing entries. Filling with the most frequent category keeps the column's "
            f"existing category set intact."
        )
    if key == "impute_unknown":
        return (
            f"'{col}' is a free-text / entity feature ({stats.get('cardinality', 0)} distinct values) with "
            f"{stats.get('missing_pct', 0.0)}% missing. Assigning the most common entity would inject false signal, so missing "
            f"entries become 'Unknown' and a has_{col} flag records whether a value was present."
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
            f"quantity that cannot be negative. These are treated as data-entry errors and replaced with the median of the "
            f"valid values, rather than clipped to an artificial 0."
        )
    if key == "cap_outliers":
        return (
            f"{stats.get('outlier_count', 0)} values ({stats.get('outlier_pct', 0.0)}%) in '{col}' fall outside the 1.5x IQR "
            f"range [{stats.get('lower_bound', 0.0)}, {stats.get('upper_bound', 0.0)}]. Capping them at these bounds limits "
            f"their leverage without discarding rows."
        )
    if key == "drop_collinear":
        return (
            f"'{stats.get('col1')}' and '{stats.get('col2')}' are strongly correlated (Pearson r = {stats.get('correlation', 0.0)}). "
            f"Keeping both inflates variance in linear models; dropping '{stats.get('col2')}' removes the redundancy. "
            f"Off by default because tree models are usually unaffected."
        )
    if key == "drop_constant":
        return f"Column '{col}' has the same value in every row, so it carries no predictive signal and is removed."
    if key == "drop_high_cardinality":
        return (
            f"'{col}' has {stats.get('unique_count', 0)} distinct values ({stats.get('unique_pct', 0.0)}% of rows), so it behaves "
            f"like free text. One-hot encoding it would create hundreds of sparse columns, so it is removed."
        )
    if key == "group_rare":
        return (
            f"'{col}' has {stats.get('unique_count', 0)} categories. Keeping the {stats.get('keep_top', 10)} most frequent and "
            f"grouping the rest as 'Other' prevents sparse, unreliable indicator columns."
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


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _error_reason(resp) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", ""))[:160]
    except ValueError:
        return ""


def _call_gemini(req_body: Dict[str, Any], api_key: str) -> Optional[Tuple[str, str]]:
    """
    Tries GEMINI_MODEL then GEMINI_FALLBACK_MODELS, retrying transient failures (overload, rate limits, timeouts)
    with exponential backoff, all within GEMINI_TOTAL_BUDGET_SEC. Returns (model_name, response_text) or None.
    """
    import requests

    models: List[str] = []
    for m in [settings.GEMINI_MODEL, *settings.GEMINI_FALLBACK_MODELS]:
        if m and m not in models:
            models.append(m)

    deadline = time.monotonic() + settings.GEMINI_TOTAL_BUDGET_SEC
    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        for attempt in range(settings.GEMINI_MAX_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                logger.warning("Gemini time budget exhausted. Using statistical explanations.")
                return None
            try:
                resp = requests.post(
                    url,
                    json=req_body,
                    headers={"x-goog-api-key": api_key},
                    timeout=min(settings.GEMINI_TIMEOUT_SEC, remaining),
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"Gemini {model_name} attempt {attempt + 1} failed ({type(e).__name__}).")
                status, reason, retry_after = None, "", None
            else:
                if resp.status_code == 200:
                    try:
                        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except (ValueError, KeyError, IndexError, TypeError):
                        logger.warning(f"Gemini {model_name} returned an unexpected response shape (possibly blocked).")
                        break
                    return model_name, text
                status, reason = resp.status_code, _error_reason(resp)
                retry_after = resp.headers.get("Retry-After")
                logger.warning(f"Gemini {model_name} attempt {attempt + 1} returned {status}: {reason}")
                if status in (400, 401, 403):
                    # Invalid key or request: other models won't fare better.
                    return None
                if status not in RETRYABLE_STATUS:
                    break  # e.g. 404 model not found -> try next model

            if attempt < settings.GEMINI_MAX_RETRIES:
                delay = 2.0 * (2 ** attempt)
                if retry_after and retry_after.isdigit():
                    delay = max(delay, min(float(retry_after), 15.0))
                if deadline - time.monotonic() - delay < 3:
                    break
                time.sleep(delay)
    logger.warning("All Gemini models failed or are unavailable. Using statistical explanations.")
    return None


def explain_recommendations_with_gemini(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage 3: Grounded LLM explanation layer. Uses Gemini when configured and falls back to deterministic
    statistical explanations otherwise. Each recommendation gets `explanation_source` = "ai" | "statistical".
    """
    if not recommendations:
        return []

    gemini_key = settings.GEMINI_API_KEY.strip()
    if not gemini_key:
        return _apply_fallback(recommendations)

    try:
        items_payload = [
            {
                "index": idx,
                "issue_type": r["issue_type"],
                "column": r.get("column"),
                "method": r["method"],
                "stats_context": {k: v for k, v in r.get("stats_context", {}).items() if k != "method_key"},
            }
            for idx, r in enumerate(recommendations)
        ]
        system_instruction = (
            "You are a Senior Data Scientist explaining pre-ML data cleaning decisions.\n"
            "For each item, write a 2-3 sentence technical justification of WHY the chosen method fits the provided statistics.\n"
            "RULES:\n"
            "1. Cite the exact numbers in stats_context.\n"
            "2. Never invent numbers that are not in the input.\n"
            "3. Explain the method that was chosen; do not recommend a different one.\n"
            "4. Return ONLY a JSON array of objects with keys 'index' (int) and 'explanation' (string)."
        )
        req_body = {
            "contents": [{"parts": [{"text": f"{system_instruction}\n\nData Payload:\n{json.dumps(items_payload, indent=2, default=str)}"}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        result = _call_gemini(req_body, gemini_key)
        if result is None:
            return _apply_fallback(recommendations)
        model_name, raw_text = result
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        parsed = json.loads(raw_text.strip())
        explanation_map = {
            item["index"]: item["explanation"]
            for item in parsed
            if isinstance(item, dict) and "index" in item and item.get("explanation")
        }
        for idx, rec in enumerate(recommendations):
            if idx in explanation_map:
                rec["explanation_text"] = str(explanation_map[idx])
                rec["explanation_source"] = "ai"
            else:
                rec["explanation_text"] = generate_fallback_explanation(rec)
                rec["explanation_source"] = "statistical"
        logger.info(f"Generated explanations using {model_name}.")
    except Exception as e:
        # Only the exception type is logged: request exceptions can embed request details.
        logger.warning(f"Gemini explanation generation failed ({type(e).__name__}). Using statistical explanations.")
        return _apply_fallback(recommendations)

    return recommendations
