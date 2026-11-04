"""
Understand agent: resolves what each column *means* before any cleaning.

Code computes evidence (placeholder tokens, sentinel values, numeric/currency parse rates, day-first evidence,
spelling variants, PII patterns) and proposes rule-based decisions. The LLM reviews them, adjusts confidence,
adds decisions from the same fixed action set, and asks the user about ambiguous cases. Every decision is
validated against the data; approved ones are stored as column overrides in the dataset policy.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.agents import policy as P
from app.agents.base import as_list, ask_llm, clamp01
from app.agents.context import PLACEHOLDER_TOKENS, dataset_brief, looks_numeric_text
from app.agents.runtime import RunContext, register
from app.core.storage import read_dataset
from app.engine.profiler import ALLOWED_TYPE_OVERRIDES, infer_column_type, type_overrides
from app.engine.transforms import _token, to_number
from app.models.dataset import Objective

ACTOR = "Understand agent"
ACTIONS = {"treat_as_missing", "parse_numeric", "parse_dayfirst", "normalize_categories", "set_type", "ignore_column"}
DESTRUCTIVE = {"ignore_column"}
AUTO_APPLY_CONFIDENCE = 0.8

SYSTEM = """You are the Understand agent in a data-readiness platform. You decide what columns MEAN before cleaning.
You receive computed column signals and rule-based candidate decisions. Review them like a senior data scientist.

Allowed actions (use ONLY these):
- treat_as_missing {"values": [tokens or numbers]}   e.g. "?", "unknown", -999 used as "no value"
- parse_numeric   {"decimal": "." or ","}            text that is really numbers ("$1,234", "12,50", "₹900")
- parse_dayfirst  {}                                  dates written dd/mm/yyyy
- normalize_categories {"mapping": {"variant": "canonical"}}  spelling/case variants of the same category
- set_type        {"type": one of numeric|categorical|boolean|datetime|id|text|year_range}
- ignore_column   {}                                  exclude (personal data, pure noise). Never the target.

Rules:
- Base every decision on the signals. Never invent values that are not in the signals.
- Confidence 0..1. Use < 0.8 when a human should confirm, and add a question for genuinely ambiguous cases.
- You may reject a candidate (by id) with a reason.
- Keep reasons to one short sentence that cites the evidence.

Return ONLY JSON:
{"decisions":[{"column":str,"action":str,"params":{},"confidence":number,"reason":str}],
 "rejected":[{"id":str,"reason":str}],
 "questions":[{"column":str,"question":str,"options":[{"label":str,"action":str|null,"params":{}}]}],
 "column_notes":{"<column>":"one-line meaning of the column"}}"""


def _label(action: str, col: str, params: Dict[str, Any]) -> str:
    if action == "treat_as_missing":
        vals = ", ".join(repr(v) for v in params.get("values", [])[:4])
        return f"Treat {vals} in '{col}' as missing"
    if action == "parse_numeric":
        return f"Read '{col}' as numbers (decimal '{params.get('decimal', '.')}')"
    if action == "parse_dayfirst":
        return f"Read '{col}' as day-first dates (dd/mm/yyyy)"
    if action == "normalize_categories":
        return f"Merge {len(params.get('mapping', {}))} spelling variant(s) in '{col}'"
    if action == "set_type":
        return f"Treat '{col}' as {params.get('type')}"
    return f"Exclude '{col}' from the analysis"


def _validate(df: pd.DataFrame, target: Optional[str], col: Any, action: Any, params: Any) -> Optional[Dict[str, Any]]:
    """Normalised (column, action, params) if the decision is valid for this data, else None."""
    if not isinstance(col, str) or col not in df.columns or action not in ACTIONS:
        return None
    params = dict(params or {}) if isinstance(params, dict) else {}
    s = df[col]
    if col == target and action in ("ignore_column", "set_type"):
        return None
    if action == "treat_as_missing":
        present = {_token(v) for v in s.dropna().unique()[:5000]}
        values = [v for v in as_list(params.get("values")) if _token(v) in present]
        return {"values": values[:20]} if values else None
    if action == "parse_numeric":
        if pd.api.types.is_numeric_dtype(s):
            return None
        decimal = "," if params.get("decimal") == "," else "."
        sample = s.dropna().astype(str).str.strip()
        sample = sample[~sample.str.lower().isin(PLACEHOLDER_TOKENS) & (sample != "")].head(500)
        if not len(sample) or sample.map(lambda v: to_number(v, decimal)).notna().mean() < 0.8:
            return None
        # Alphanumeric codes ('C85', 'SKU-12') would lose their letters and become fake measurements.
        if sample.map(looks_numeric_text).mean() < 0.8:
            return None
        return {"decimal": decimal}
    if action == "parse_dayfirst":
        return {} if not pd.api.types.is_numeric_dtype(s) else None
    if action == "normalize_categories":
        mapping = P.category_mapping(s)
        values = set(s.dropna().astype(str).unique()[:5000])
        for k, v in (params.get("mapping") or {}).items():
            if str(k) in values and str(v) in values and str(k) != str(v):
                mapping[str(k)] = str(v)
        return {"mapping": mapping} if mapping else None
    if action == "set_type":
        return {"type": params.get("type")} if params.get("type") in ALLOWED_TYPE_OVERRIDES else None
    return {}


def _rule_candidates(df: pd.DataFrame, signals: List[Dict[str, Any]], target: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(col, action, params, confidence, reason):
        valid = _validate(df, target, col, action, params)
        if valid is not None:
            out.append({"column": col, "action": action, "params": valid, "confidence": confidence, "reason": reason, "source": "rules"})

    for sig in signals:
        col, typ = sig["column"], sig["type"]
        tokens = sig.get("placeholder_tokens") or {}
        if tokens:
            total = sum(tokens.values())
            conf = 0.92 if sig.get("numeric_parse_ratio", 0) >= 0.9 else 0.8
            add(col, "treat_as_missing", {"values": list(tokens)}, conf,
                f"{total} cell(s) hold placeholder text ({', '.join(list(tokens)[:3])}) instead of a value.")
        for cand in sig.get("sentinel_candidates") or []:
            add(col, "treat_as_missing", {"values": [cand["value"]]}, 0.65,
                f"{cand['count']} rows hold {cand['value']:g}, a typical 'unknown' code far outside the normal range.")
        # Plain numbers held back only by placeholders become numeric once those are treated as missing.
        if (sig.get("numeric_parse_ratio", 0) >= 0.9 and sig.get("parsed_numeric_ratio", 0) >= 0.9
                and sig.get("plain_number_ratio", 0) < 0.9 and typ not in ("unit_mixed_numeric", "datetime")):
            decimal = "," if sig.get("decimal_comma_ratio", 0) > 0.5 else "."
            add(col, "parse_numeric", {"decimal": decimal}, 0.9,
                f"{round(sig['parsed_numeric_ratio'] * 100)}% of values are numbers written with symbols or separators.")
        triples = sig.get("date_triples")
        if triples and triples["share"] >= 0.8 and triples["year_first"] == 0:
            if triples["first_over_12"] > 0 and triples["second_over_12"] == 0:
                add(col, "parse_dayfirst", {}, 0.95,
                    f"{triples['first_over_12']} dates have a first number above 12, so the format is day-first.")
        if sig.get("parses_as_date_ratio", 0) >= 0.9 and typ in ("text", "categorical", "person_name_or_text"):
            add(col, "set_type", {"type": "datetime"}, 0.85,
                f"{round(sig['parses_as_date_ratio'] * 100)}% of a random sample parses as dates.")
        if typ == "id" and sig.get("consecutive_unique_integers") and sig.get("year_like_range"):
            add(col, "set_type", {"type": "year_range"}, 0.9,
                "Consecutive unique integers between 1800 and 2100: a year per row, not an identifier.")
        if typ == "id" and sig.get("all_unique_floats"):
            add(col, "set_type", {"type": "numeric"}, 0.9, "Unique decimal values are measurements, not identifiers.")
        if sig.get("spelling_variants"):
            add(col, "normalize_categories", {}, 0.85,
                f"{sig['spelling_variants']['groups']} categor(ies) appear with different case or spacing.")
        if col != target and (sig.get("email_ratio", 0) >= 0.8 or sig.get("phone_ratio", 0) >= 0.8):
            add(col, "ignore_column", {}, 0.8, "Looks like personal contact data (PII); it has no predictive value.")
    return out


def _ambiguity_questions(signals: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    questions = []
    for sig in signals:
        triples = sig.get("date_triples")
        if (triples and triples["share"] >= 0.8 and triples["year_first"] == 0
                and triples["first_over_12"] == 0 and triples["second_over_12"] == 0
                and not pd.api.types.is_numeric_dtype(df[sig["column"]])):
            questions.append({
                "column": sig["column"],
                "question": f"Dates in '{sig['column']}' like 03/04 are ambiguous. Which comes first?",
                "options": [
                    {"label": "Day first (03/04 = 3 April)", "action": "parse_dayfirst", "params": {}},
                    {"label": "Month first (03/04 = 4 March)", "action": None, "params": {}},
                ],
            })
    return questions


def propose(ctx: RunContext) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    db = ctx.db
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    target = obj.target_column if obj else None
    overrides = P.column_overrides(db, ctx.dataset.id)
    df = P.apply_prep(read_dataset(ctx.dataset.raw_file_path), overrides)

    ctx.step(ACTOR, "tool", f"Profiling {df.shape[1]} columns for meaning: placeholders, sentinels, number formats, dates, variants, PII.")
    with type_overrides(P.type_map(overrides)):
        types = {c: infer_column_type(df[c], c) for c in df.columns}
    brief = dataset_brief(df, types, target, obj.problem_type if obj else None)
    candidates = _rule_candidates(df, brief["column_signals"], target)
    questions = _ambiguity_questions(brief["column_signals"], df)
    ctx.step(ACTOR, "tool", f"Rules found {len(candidates)} candidate decision(s) and {len(questions)} ambiguity question(s).",
             data=[{"column": c["column"], "action": c["action"], "confidence": c["confidence"]} for c in candidates])
    for i, c in enumerate(candidates):
        c["id"] = f"r{i + 1}"

    notes: Dict[str, str] = {}
    answer = ask_llm(
        ctx, ACTOR, SYSTEM,
        {"dataset": brief, "already_applied": overrides, "candidates": candidates, "rule_questions": questions},
        purpose="review the column candidates and look for anything the rules missed",
    )
    if isinstance(answer, dict):
        rejected = {str(r.get("id")): str(r.get("reason", ""))[:200] for r in as_list(answer.get("rejected")) if isinstance(r, dict)}
        for c in candidates:
            if c["id"] in rejected:
                c["confidence"] = min(c["confidence"], 0.4)
                c["reason"] = f"{c['reason']} Agent doubts it: {rejected[c['id']]}"
                c["source"] = "rules (agent disagrees)"
        for d in as_list(answer.get("decisions")):
            if not isinstance(d, dict):
                continue
            valid = _validate(df, target, d.get("column"), d.get("action"), d.get("params"))
            if valid is None:
                ctx.step("Critic", "critic", f"Rejected an invalid agent decision: {d.get('action')} on {d.get('column')!r}.")
                continue
            same = next((c for c in candidates if c["column"] == d["column"] and c["action"] == d["action"]), None)
            if same:
                same.update(params=valid, confidence=clamp01(d.get("confidence"), same["confidence"]),
                            reason=str(d.get("reason") or same["reason"])[:300], source="rules + agent")
            else:
                candidates.append({"id": f"a{len(candidates) + 1}", "column": d["column"], "action": d["action"], "params": valid,
                                   "confidence": clamp01(d.get("confidence")), "reason": str(d.get("reason") or "")[:300], "source": "agent"})
        for q in as_list(answer.get("questions"))[:5]:
            if isinstance(q, dict) and q.get("column") in df.columns and q.get("question") and isinstance(q.get("options"), list):
                opts = [o for o in q["options"] if isinstance(o, dict) and o.get("label")][:4]
                if len(opts) >= 2 and not any(x["column"] == q["column"] for x in questions):
                    questions.append({"column": q["column"], "question": str(q["question"])[:240], "options": opts})
        if isinstance(answer.get("column_notes"), dict):
            notes = {str(k): str(v)[:160] for k, v in answer["column_notes"].items() if k in df.columns}

    # Skip decisions that are already in force
    def already(c):
        o = overrides.get(c["column"], {})
        return ((c["action"] == "parse_dayfirst" and o.get("dayfirst")) or (c["action"] == "ignore_column" and o.get("ignore"))
                or (c["action"] == "set_type" and o.get("type") == c["params"].get("type"))
                or (c["action"] == "parse_numeric" and o.get("parse_numeric")))

    proposals = []
    for c in candidates:
        if already(c):
            continue
        destructive = c["action"] in DESTRUCTIVE
        proposals.append({
            **c,
            "label": _label(c["action"], c["column"], c["params"]),
            "destructive": destructive,
            "default": c["confidence"] >= AUTO_APPLY_CONFIDENCE and not destructive,
        })
    for i, q in enumerate(questions):
        q["id"] = f"q{i + 1}"
    return proposals, questions, notes


def apply_decisions(ctx: RunContext, selected: List[Dict[str, Any]]) -> List[str]:
    db = ctx.db
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    target = obj.target_column if obj else None
    pol = P.get_policy(db, ctx.dataset.id, create=True)
    overrides: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in (pol.column_overrides or {}).items()}
    df = P.apply_prep(read_dataset(ctx.dataset.raw_file_path), overrides)
    applied = []
    for d in selected:
        valid = _validate(df, target, d.get("column"), d.get("action"), d.get("params"))
        if valid is None:
            continue
        col, action = d["column"], d["action"]
        o = overrides.setdefault(col, {})
        if action == "treat_as_missing":
            o["na_tokens"] = list(dict.fromkeys([*o.get("na_tokens", []), *valid["values"]]))
        elif action == "parse_numeric":
            o["parse_numeric"] = valid
        elif action == "parse_dayfirst":
            o["dayfirst"] = True
        elif action == "normalize_categories":
            o["normalize"] = valid
        elif action == "set_type":
            o["type"] = valid["type"]
        elif action == "ignore_column":
            o["ignore"] = True
        o.setdefault("notes", []).append({"action": action, "reason": d.get("reason", ""), "source": d.get("source", "user")})
        applied.append(_label(action, col, valid))
    pol.column_overrides = overrides
    db.commit()
    return applied


def selected_from_response(proposals: List[Dict[str, Any]], questions: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    approvals = response.get("approvals") or {}
    answers = response.get("answers") or {}
    chosen = [p for p in proposals if approvals.get(p["id"], p.get("default", False))]
    for q in questions:
        idx = answers.get(q["id"])
        if idx is None:
            continue
        try:
            opt = q["options"][int(idx)]
        except (ValueError, IndexError, TypeError):
            continue
        if opt.get("action"):
            chosen.append({"column": q["column"], "action": opt["action"], "params": opt.get("params") or {},
                           "reason": f"User answer: {opt['label']}", "source": "user"})
    return chosen


@register("understand")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    if response is None:
        proposals, questions, notes = propose(ctx)
        if not proposals and not questions:
            ctx.step(ACTOR, "result", "Every column already reads correctly. No decisions needed.")
            ctx.finish({"applied": [], "notes": notes, "needs_rediagnose": False})
            return
        ctx.save_state(proposals=proposals, questions=questions, notes=notes)
        ctx.gate({
            "type": "understand",
            "title": f"Review {len(proposals)} column decision(s)" + (f" and {len(questions)} question(s)" if questions else ""),
            "items": proposals,
            "questions": questions,
            "notes": notes,
        })
        return

    selected = selected_from_response(ctx.state.get("proposals", []), ctx.state.get("questions", []), response)
    applied = apply_decisions(ctx, selected)
    ctx.step(ACTOR, "result", f"Saved {len(applied)} column decision(s) to the dataset policy." if applied else "No decisions applied.",
             data=applied)
    ctx.finish({"applied": applied, "notes": ctx.state.get("notes", {}), "needs_rediagnose": bool(applied)})
