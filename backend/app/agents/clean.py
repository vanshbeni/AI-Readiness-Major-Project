"""
Clean agent: turns detected issues into a fix plan by choosing, per recommendation, a method from a fixed
catalogue of executable alternatives (the executor already implements every one of them).

Rules propose changes from computed facts (e.g. keep a predictive high-cardinality column by grouping rare
values instead of dropping it; drop the weaker of two collinear features). The LLM reviews and may choose other
catalogue options. Nothing is applied until the user approves at the gate.
"""
from typing import Any, Dict, List, Optional

from app.agents import policy as P
from app.agents.base import as_list, ask_llm
from app.agents.context import min_support, target_association
from app.agents.runtime import RunContext, register
from app.api.deps import invalidate_execution
from app.engine.explainer import generate_fallback_explanation
from app.engine.rules import INDICATOR_ONLY_PCT
from app.models.dataset import Objective, Recommendation

ACTOR = "Clean agent"
NUMERIC_TYPES = {"numeric", "year_range"}
CATEGORY_TYPES = {"categorical", "boolean"}

METHOD_LABELS = {
    "impute_median": "Median Imputation",
    "impute_mean": "Mean Imputation",
    "impute_mode": "Mode (Most Frequent) Imputation",
    "impute_unknown": "Impute with 'Unknown' & Binary Presence Flag",
    "drop_column_missing": "Drop High-Missingness Column",
    "missing_indicator": "Replace with Presence Indicator (has_<column>)",
    "cap_outliers": "Cap Extreme Values + Capped Flag",
    "replace_negatives_median": "Replace Invalid Negatives with Median",
    "drop_high_cardinality": "Drop High-Cardinality Free-Text Column",
    "group_rare": "Group Rare Categories (Top N + 'Other')",
    "drop_collinear": "Drop Redundant Collinear Feature",
    "class_weights": "Balanced Class Weighting (Cost-Sensitive)",
    "smote_class_weights": "SMOTE Oversampling + Balanced Class Weights (in training folds)",
}
DESTRUCTIVE_KEYS = {"drop_column_missing", "drop_high_cardinality", "drop_collinear", "drop_identifier",
                    "drop_constant", "drop_leakage", "drop_missing_target"}

SYSTEM = """You are the Clean agent in a data-readiness platform. You review a cleaning plan before it runs.
For each recommendation you get: the issue, the column, computed statistics, how strongly the column is associated
with the target (0..1), the current method, and the ONLY alternative methods the executor supports.

Decide per recommendation: approve or skip, and which method_key (from its alternatives) with params.
Principles:
- Do not drop a column that carries signal for the target if a non-destructive alternative exists.
- target_association is computed over categories with at least min_support rows, so near-unique columns
  (names, codes, free text) score ~0. Never keep such a column with group_rare: supported_categories tells you how
  many values have enough rows; fewer than 2 means grouping would only memorise individual rows.
- Prefer median over mean; mean imputation flattens variation. Keep the missing flag for sizeable gaps.
- Columns that are mostly blank are best kept as a presence indicator (missing_indicator), not filled.
- Only extreme values should be capped; tail values in skewed money/count columns are usually real.
- For collinear pairs keep the column more associated with the target.
- Skip a fix if it would destroy legitimate information; say why.
- Only include recommendations you want to CHANGE (method, params or approval).
- One short reason per decision citing the numbers you were given.

Return ONLY JSON:
{"decisions":[{"rec_id":str,"approve":bool,"method_key":str,"params":{},"reason":str}]}"""


def _alternatives(rec: Recommendation, col_type: str) -> List[Dict[str, Any]]:
    p = rec.params or {}
    key = p.get("method_key")
    it = rec.issue_type
    if it == "missing_values":
        if col_type in NUMERIC_TYPES:
            keys = ["impute_median", "impute_mean", "missing_indicator", "drop_column_missing"]
        elif col_type in CATEGORY_TYPES:
            keys = ["impute_mode", "impute_unknown", "missing_indicator", "drop_column_missing"]
        else:
            keys = ["impute_unknown", "impute_mode", "missing_indicator", "drop_column_missing"]
        return [{"method_key": k} for k in keys]
    if it in ("high_cardinality_text", "rare_categories"):
        alts = [{"method_key": "group_rare", "params": {"keep_top": "integer 5-50"}}]
        if it == "high_cardinality_text":
            alts.insert(0, {"method_key": "drop_high_cardinality"})
        return alts
    if it == "multicollinear_features":
        return [{"method_key": "drop_collinear", "params": {"drop": f"'{p.get('col1')}' or '{p.get('col2')}'"}}]
    if it == "class_imbalance":
        return [{"method_key": "class_weights"}, {"method_key": "smote_class_weights"}]
    return [{"method_key": key}]


def _validate(rec: Recommendation, col_type: str, method_key: Any, params: Any,
              fact: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    allowed = {a["method_key"] for a in _alternatives(rec, col_type)}
    if method_key not in allowed:
        return None
    params = dict(params or {}) if isinstance(params, dict) else {}
    out: Dict[str, Any] = {}
    if (method_key in ("impute_median", "impute_mean", "impute_mode", "impute_unknown")
            and float((rec.params or {}).get("missing_pct") or 0) > INDICATOR_ONLY_PCT):
        return None
    if method_key == "group_rare":
        supported = (fact or {}).get("supported_categories")
        if supported is not None and supported < 2:
            return None
        try:
            out["keep_top"] = max(2, min(100, int(params.get("keep_top", 20))))
        except (TypeError, ValueError):
            out["keep_top"] = 20
        if supported is not None:
            out["keep_top"] = min(out["keep_top"], int(supported))
    if method_key == "drop_collinear":
        p = rec.params or {}
        drop = str(params.get("drop") or rec.column or "").strip("'\" ")
        if drop not in (p.get("col1"), p.get("col2")):
            return None
        out["drop"] = drop
    return out


def _new_title(rec: Recommendation, method_key: str, params: Dict[str, Any]) -> str:
    col = rec.column or ""
    if method_key == "group_rare":
        return f"Keep the {params['keep_top']} most frequent values of '{col}' and group the rest as 'Other'"
    if method_key == "drop_collinear":
        p = rec.params or {}
        other = p.get("col1") if params["drop"] == p.get("col2") else p.get("col2")
        return f"Drop '{params['drop']}' (correlation {p.get('correlation')} with '{other}')"
    if method_key == "missing_indicator":
        return f"Keep only whether '{col}' was recorded (has_{col})"
    if method_key == "drop_column_missing":
        return f"Drop '{col}' ({(rec.params or {}).get('missing_pct', '?')}% missing)"
    if method_key in ("impute_median", "impute_mean", "impute_mode", "impute_unknown"):
        return f"Impute '{col}' using {METHOD_LABELS[method_key].split(' (')[0].replace(' Imputation', '')}"
    return rec.reason_title


def _facts_for(rec: Recommendation, col_type: str, association: Optional[float]) -> Dict[str, Any]:
    stats = {k: v for k, v in (rec.params or {}).items() if k not in ("sample_values", "agent_note", "method_key")}
    return {
        "rec_id": rec.id,
        "issue_type": rec.issue_type,
        "column": rec.column,
        "column_type": col_type,
        "severity": rec.severity,
        "current_method": (rec.params or {}).get("method_key"),
        "currently_approved": bool(rec.is_approved),
        "target_association": association,
        "stats": stats,
        "alternatives": _alternatives(rec, col_type),
    }


def propose(ctx: RunContext) -> List[Dict[str, Any]]:
    db = ctx.db
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == ctx.dataset.id).all()
    if not obj or not recs:
        ctx.step(ACTOR, "warn", "No recommendations to review. Run diagnostics first.")
        return []
    df, _, types = P.load_prepared(db, ctx.dataset)
    from app.engine.profiler import infer_column_type, type_overrides
    with type_overrides(types):
        col_types = {c: infer_column_type(df[c], c) for c in df.columns}

    def assoc(col):
        return target_association(df, col, obj.target_column, obj.problem_type) if col else None

    facts = []
    for r in recs:
        if r.issue_type == "multicollinear_features":
            p = r.params or {}
            f = _facts_for(r, col_types.get(p.get("col2"), "numeric"), assoc(p.get("col2")))
            f["association_col1"], f["association_col2"] = assoc(p.get("col1")), assoc(p.get("col2"))
        else:
            f = _facts_for(r, col_types.get(r.column or "", "text"), assoc(r.column))
            if r.issue_type in ("high_cardinality_text", "rare_categories") and r.column in df.columns:
                counts = df[r.column].dropna().astype(str).value_counts()
                f["min_support"] = min_support(len(df))
                f["supported_categories"] = int((counts >= f["min_support"]).sum())
                f["distinct_values"] = int(counts.size)
        facts.append(f)
        if f.get("target_association") is not None:
            ctx.facts.add(f"association of '{r.column}' with '{obj.target_column}'", f["target_association"])
    ctx.step(ACTOR, "tool", f"Computed target association and alternatives for {len(recs)} recommendation(s).",
             data=[{"column": f["column"], "issue": f["issue_type"], "association": f.get("target_association")} for f in facts])

    by_id = {r.id: r for r in recs}
    fact_by_id = {f["rec_id"]: f for f in facts}
    decisions: Dict[str, Dict[str, Any]] = {}

    # Rule-based proposals
    for f in facts:
        r = by_id[f["rec_id"]]
        a = f.get("target_association")
        if (r.issue_type == "high_cardinality_text" and a is not None and a >= 0.25
                and f.get("supported_categories", 0) >= 2):
            keep = min(20, f["supported_categories"])
            decisions[r.id] = {"approve": True, "method_key": "group_rare", "params": {"keep_top": keep}, "source": "rules",
                               "reason": f"'{r.column}' is associated with the target ({a}) and {f['supported_categories']} values have "
                                         f"{f['min_support']}+ rows; grouping the rest keeps that signal instead of dropping it."}
        if r.issue_type == "multicollinear_features":
            p = r.params or {}
            a1, a2 = f.get("association_col1"), f.get("association_col2")
            if a1 is not None and a2 is not None and a2 > a1 + 0.02:
                decisions[r.id] = {"approve": bool(r.is_approved), "method_key": "drop_collinear", "params": {"drop": p.get("col1")},
                                   "source": "rules",
                                   "reason": f"'{p.get('col2')}' is more associated with the target ({a2} vs {a1}), so drop '{p.get('col1')}' instead."}

    answer = ask_llm(
        ctx, ACTOR, SYSTEM,
        {"target": obj.target_column, "problem_type": obj.problem_type, "rows": int(len(df)),
         "recommendations": facts, "rule_suggestions": decisions},
        purpose="review the cleaning plan and pick the best executable method per issue",
    )
    if isinstance(answer, dict):
        for d in as_list(answer.get("decisions")):
            if not isinstance(d, dict) or d.get("rec_id") not in by_id:
                continue
            r = by_id[d["rec_id"]]
            mk = d.get("method_key") or (r.params or {}).get("method_key")
            valid = _validate(r, fact_by_id[r.id]["column_type"], mk, d.get("params"), fact_by_id[r.id])
            if valid is None:
                why = ("fewer than 2 values have enough rows, so grouping would only memorise rows" if mk == "group_rare" else
                       "the column is mostly blank, so filling it would invent most of its values" if str(mk).startswith("impute_") else
                       "not an executable alternative")
                ctx.step("Critic", "critic", f"Rejected agent choice '{mk}' for '{r.column}': {why}.")
                continue
            decisions[r.id] = {"approve": bool(d.get("approve", True)), "method_key": mk, "params": valid,
                               "source": "agent", "reason": str(d.get("reason") or "")[:300]}

    proposals = []
    for rid, d in decisions.items():
        r = by_id[rid]
        current_key = (r.params or {}).get("method_key")
        valid = _validate(r, fact_by_id[rid]["column_type"], d["method_key"], d["params"], fact_by_id[rid]) or {}
        same_method = d["method_key"] == current_key and (
            d["method_key"] not in ("group_rare", "drop_collinear")
            or (d["method_key"] == "group_rare" and valid.get("keep_top") == (r.params or {}).get("keep_top"))
            or (d["method_key"] == "drop_collinear" and valid.get("drop") == r.column)
        )
        if same_method and d["approve"] == bool(r.is_approved):
            continue
        to_label = METHOD_LABELS.get(d["method_key"], r.method)
        proposals.append({
            "id": rid,
            "column": r.column,
            "issue_type": r.issue_type,
            "label": (f"{'Skip' if not d['approve'] else 'Apply'}: {_new_title(r, d['method_key'], valid) if not same_method else r.reason_title}"),
            "from": f"{r.method}{'' if r.is_approved else ' (skipped)'}",
            "to": f"{to_label}{'' if d['approve'] else ' (skipped)'}",
            "approve": d["approve"],
            "method_key": d["method_key"],
            "params": valid,
            "reason": d["reason"],
            "source": d["source"],
            "destructive": d["method_key"] in DESTRUCTIVE_KEYS,
            "default": True,
        })
    return proposals


def apply_proposals(ctx: RunContext, selected: List[Dict[str, Any]]) -> List[str]:
    db = ctx.db
    recs = {r.id: r for r in db.query(Recommendation).filter(Recommendation.dataset_id == ctx.dataset.id).all()}
    changed = []
    for prop in selected:
        r = recs.get(prop["id"])
        if not r:
            continue
        params = dict(r.params or {})
        mk = prop["method_key"]
        if mk == "drop_collinear":
            drop = prop["params"]["drop"]
            if drop != params.get("col2"):
                params["col1"], params["col2"] = params.get("col2"), params.get("col1")
            params["column_name"] = drop
            r.column = drop
        if mk == "group_rare":
            params["keep_top"] = prop["params"]["keep_top"]
        r.reason_title = _new_title(r, mk, prop["params"]) if mk != params.get("method_key") or mk in ("group_rare", "drop_collinear") else r.reason_title
        params["method_key"] = mk
        params["agent_note"] = prop.get("reason") or "Changed by the Clean agent."
        r.params = params
        r.method = METHOD_LABELS.get(mk, r.method)
        r.is_destructive = mk in DESTRUCTIVE_KEYS
        r.is_approved = bool(prop["approve"])
        r.explanation_text = generate_fallback_explanation({"stats_context": params, "column": r.column, "method": r.method})
        r.explanation_source = "statistical"
        changed.append(prop["label"])
    if changed:
        invalidate_execution(db, ctx.dataset)
    db.commit()
    return changed


@register("clean")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    if response is None:
        proposals = propose(ctx)
        if not proposals:
            ctx.step(ACTOR, "result", "The current plan already looks right. No changes proposed.")
            ctx.finish({"changed": []})
            return
        ctx.save_state(proposals=proposals)
        ctx.gate({"type": "clean", "title": f"Review {len(proposals)} change(s) to the cleaning plan", "items": proposals})
        return
    approvals = response.get("approvals") or {}
    selected = [p for p in ctx.state.get("proposals", []) if approvals.get(p["id"], p.get("default", True))]
    changed = apply_proposals(ctx, selected)
    ctx.step(ACTOR, "result", f"Updated {len(changed)} recommendation(s)." if changed else "No changes applied.", data=changed)
    ctx.finish({"changed": changed})
