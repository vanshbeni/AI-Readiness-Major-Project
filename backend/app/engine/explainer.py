import json
import logging
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


def generate_fallback_explanation(rec: Dict[str, Any]) -> str:
    """Deterministic fallback explanation strictly grounded in computed statistics and Data Science best practices."""
    issue_type = rec["issue_type"]
    method = rec["method"]
    stats = rec.get("stats_context", {})
    col = rec.get("column", "")

    if issue_type == "duplicate_rows":
        cnt = stats.get("duplicate_count", 0)
        pct = stats.get("duplicate_pct", 0.0)
        return (
            f"The dataset contains {cnt} exact duplicate records ({pct}% of total rows). "
            f"Retaining identical rows biases model loss functions toward repeated observations. "
            f"Deduplication ensures clean, unbiased generalization across train and validation splits."
        )

    if issue_type == "missing_values":
        pct = stats.get("missing_pct", 0.0)
        dtype = stats.get("inferred_type", "text")
        skew = stats.get("skewness", 0.0)
        card = stats.get("cardinality", 0)

        if "Unknown" in method or dtype in ["person_name_or_text", "text"] or card > 20:
            return (
                f"Column '{col}' is a high-cardinality entity feature ({card} unique values) with {pct}% missing entries. "
                f"Applying mode imputation would falsely assign the top single entity to thousands of unrelated records. "
                f"Imputing with 'Unknown' preserves semantic truth without injecting deceptive category bias."
            )
        if pct > 70.0:
            return (
                f"Column '{col}' exhibits {pct}% missing data, exceeding the viable imputation threshold of 70%. "
                f"Imputing over two-thirds of a column introduces synthetic artifacts; dropping it prevents dimensionality noise."
            )
        if "Median" in method:
            return (
                f"Column '{col}' has {pct}% missing values and a non-normal skewed distribution (skewness: {skew}). "
                f"Median imputation is mathematically robust against extreme distribution tails and preserves sample medians."
            )
        if "Mean" in method:
            return (
                f"Column '{col}' has {pct}% missing values with a symmetric distribution (skewness: {skew}). "
                f"Mean imputation preserves the first central statistical moment without shifting feature scale."
            )
        return (
            f"Column '{col}' is a low-cardinality nominal category with {pct}% missing entries. "
            f"Mode imputation accurately fills gaps with the dominant class without fragmenting category distribution."
        )

    if issue_type == "unparsed_datetime_feature":
        samples = stats.get("sample_values", [])
        sample_str = f" (e.g. {', '.join([str(s) for s in samples[:2]])})" if samples else ""
        return (
            f"Column '{col}' contains chronological date/timestamp strings{sample_str}. "
            f"Treating dates as nominal categories destroys temporal trends. "
            f"Parsing with datetime and engineering 'year_added', 'month_added', and 'day_added' extracts structured numeric signals for models."
        )

    if issue_type == "unit_mixed_feature":
        samples = stats.get("sample_values", [])
        sample_str = f" (e.g. {', '.join([str(s) for s in samples[:2]])})" if samples else ""
        return (
            f"Column '{col}' conflates numeric magnitude with unit labels{sample_str} (e.g. runtime minutes vs episodic seasons). "
            f"Disentangling into a numeric 'duration_value' and categorical 'duration_unit' isolates continuous duration while preserving entity type distinctions."
        )

    if issue_type == "multilabel_delimited_text":
        samples = stats.get("sample_values", [])
        sample_str = f" (e.g. {', '.join([str(s) for s in samples[:2]])})" if samples else ""
        return (
            f"Column '{col}' contains comma-separated multi-label tokens{sample_str}. "
            f"Categorical bucketing treats combinations as isolated labels; Multi-Hot Binarization tokenizes individual tags into independent binary features."
        )

    if issue_type == "invalid_domain_values":
        cnt = stats.get("negative_count", 0)
        min_v = stats.get("min_value", 0.0)
        return (
            f"Feature '{col}' contains {cnt} invalid negative values (minimum: {min_v}) in a non-negative domain. "
            f"Clipping negative values to zero (0.0) enforces physical domain bounds without discarding valid sample rows."
        )

    if issue_type == "statistical_outliers":
        cnt = stats.get("outlier_count", 0)
        pct = stats.get("outlier_pct", 0.0)
        lb = stats.get("lower_bound", 0.0)
        ub = stats.get("upper_bound", 0.0)
        return (
            f"Identified {cnt} outliers ({pct}% of rows) in '{col}' falling outside the 1.5x IQR interval [{lb}, {ub}]. "
            f"Winsorizing (capping) extreme values to these boundary limits stabilizes gradient updates and prevents high variance."
        )

    if issue_type == "multicollinear_features":
        col1 = stats.get("col1")
        col2 = stats.get("col2")
        r = stats.get("correlation", 0.0)
        return (
            f"Strong multicollinearity detected between '{col1}' and '{col2}' (Pearson r = {r}). "
            f"Redundant features inflate variance and distort coefficient weights. Dropping '{col2}' preserves predictive variance while simplifying dimensionality."
        )

    if issue_type == "class_imbalance":
        ratio = stats.get("ratio_display", "")
        if "SMOTE" in method:
            return (
                f"Severe target imbalance detected on '{col}' ({ratio}). "
                f"Synthetic Minority Over-sampling Technique (SMOTE) generates realistic minority vectors along decision boundaries, preventing majority-class collapse."
            )
        return (
            f"Moderate target imbalance detected on '{col}' ({ratio}). "
            f"Cost-sensitive balanced class weighting penalizes minority misclassifications without artificially inflating dataset sample size."
        )

    if issue_type == "constant_feature":
        return (
            f"Column '{col}' exhibits zero variance across all records. A feature with identical values carries zero predictive signal and should be dropped."
        )

    return f"Recommended remediation '{method}' addresses data quality defects in '{col}' based on observed statistical distribution."


def explain_recommendations_with_gemini(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage 3: Grounded LLM Explanation Layer
    Augments recommendations with natural-language reasoning. Uses Gemini API (gemini-3-flash-preview);
    falls back cleanly to deterministic statistical explanations otherwise.
    """
    if not recommendations:
        return []

    gemini_key = settings.GEMINI_API_KEY.strip()
    if not gemini_key:
        logger.info("GEMINI_API_KEY not set. Using deterministic grounded statistical explanations.")
        for rec in recommendations:
            rec["explanation_text"] = generate_fallback_explanation(rec)
        return recommendations

    model_name = getattr(settings, "GEMINI_MODEL", "gemini-3-flash-preview") or "gemini-3-flash-preview"

    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"

        items_payload = [
            {
                "index": idx,
                "issue_type": r["issue_type"],
                "column": r.get("column"),
                "method": r["method"],
                "stats_context": r.get("stats_context", {}),
            }
            for idx, r in enumerate(recommendations)
        ]

        system_instruction = (
            "You are a Senior Principal Data Scientist and Pre-ML Data Quality Diagnostic Engine.\n"
            "Below is a list of detected tabular data quality issues, computed statistics, and chosen remediation methods.\n"
            "Generate an insightful, mathematically sound 2-3 sentence technical justification for each item explaining WHY this specific method was chosen based on the provided numbers and data semantics.\n\n"
            "CRITICAL DATA SCIENCE RULES:\n"
            "1. CITE NUMBERS: Cite the exact numbers provided in stats_context (percentages, skewness, row counts, bounds).\n"
            "2. NO HALLUCINATIONS: Do not invent numbers or statistics not present in the input.\n"
            "3. HIGH CARDINALITY & ENTITIES: Never recommend mode imputation for person names (like directors/actors) or free text; explain why 'Unknown' imputation prevents false signal injection.\n"
            "4. TIMESTAMPS: Explain why datetime parsing (extracting year/month/day) is superior to categorical bucketing.\n"
            "5. MULTI-LABEL LISTS: Explain why multi-hot token encoding captures combinations of tags/genres without combinatorial explosion.\n"
            "6. UNITS: Explain why separating numerical value and unit prevents conflating different metrics.\n"
            "7. Return ONLY a valid JSON array of objects with keys 'index' (int) and 'explanation' (string)."
        )

        req_body = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_instruction}\n\nData Payload:\n{json.dumps(items_payload, indent=2)}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        resp = requests.post(url, json=req_body, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]

            parsed = json.loads(raw_text.strip())
            explanation_map = {item["index"]: item["explanation"] for item in parsed if "index" in item and "explanation" in item}

            for idx, rec in enumerate(recommendations):
                if idx in explanation_map and explanation_map[idx]:
                    rec["explanation_text"] = explanation_map[idx]
                else:
                    rec["explanation_text"] = generate_fallback_explanation(rec)
            logger.info(f"Successfully generated explanations using {model_name}.")
        else:
            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}. Using fallback.")
            for rec in recommendations:
                rec["explanation_text"] = generate_fallback_explanation(rec)

    except Exception as e:
        logger.warning(f"Gemini API explanation generation failed ({e}). Falling back to deterministic explanations.")
        for rec in recommendations:
            rec["explanation_text"] = generate_fallback_explanation(rec)

    return recommendations
