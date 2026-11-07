"""
Evaluate (model) agent: designs the benchmark from data facts: which metric, which split strategy,
how many rows. Rules give a sound default; the LLM may choose differently from the same registry with a reason.
"""
import os
import warnings
from typing import Any, Dict, Optional

import pandas as pd

from app.agents import policy as P
from app.agents.base import ask_llm
from app.agents.runtime import RunContext, register
from app.core.storage import read_dataset
from app.engine.benchmark import CV_STRATEGIES, MAX_BENCHMARK_ROWS, MAX_PLAN_ROWS, METRICS
from app.engine.profiler import infer_column_type, type_overrides
from app.engine.tasks import is_unsupervised
from app.models.dataset import Objective

ACTOR = "Evaluate agent"

SYSTEM = """You are the Evaluate agent. You design how candidate models are compared, so the leaderboard reflects
real-world performance. You receive facts about the dataset and a rule-based plan.

Choose:
- metric: one of the allowed metric keys for the problem type
- cv: "stratified" | "kfold" | "time"  (use "time" only when rows are events ordered in time and a time column is given)
- time_column: required when cv = "time", must be one of the candidate time columns
- max_rows: integer rows to use (more = slower but more reliable; at most the rows available)
Explain the choice in 1-2 plain sentences a non-expert understands.

Return ONLY JSON: {"metric":str,"cv":str,"time_column":str|null,"max_rows":int,"reason":str}"""


def _facts(ctx: RunContext) -> Dict[str, Any]:
    db = ctx.db
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    df, _, types = P.load_prepared(db, ctx.dataset)
    target = obj.target_column
    y = df[target].dropna()
    facts: Dict[str, Any] = {
        "problem_type": obj.problem_type,
        "target": target,
        "rows_with_target": int(len(y)),
        "features": int(df.shape[1] - 1),
        "has_cleaned_data": bool(ctx.dataset.cleaned_file_path and os.path.exists(ctx.dataset.cleaned_file_path)),
        "allowed_metrics": list(METRICS[obj.problem_type]),
    }
    if obj.problem_type == "classification":
        shares = y.astype(str).value_counts(normalize=True)
        facts["classes"] = int(len(shares))
        facts["minority_share"] = round(float(shares.iloc[-1]), 4)
        facts["majority_share"] = round(float(shares.iloc[0]), 4)
    else:
        yn = pd.to_numeric(y, errors="coerce").dropna()
        facts["target_skew"] = round(float(yn.skew()), 3) if len(yn) > 2 else 0.0
        facts["target_has_outliers"] = bool(len(yn) > 10 and ((yn - yn.median()).abs() > 5 * yn.std()).any())

    with type_overrides(types):
        dated = [c for c in df.columns if c != target and infer_column_type(df[c], c) in ("datetime",)]
    time_cols = []
    for c in dated[:5]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed", utc=True)
        ok = parsed.dropna()
        if len(ok) < 10:
            continue
        diffs = ok.diff().dropna()
        time_cols.append({
            "column": c,
            "span_days": round((ok.max() - ok.min()).total_seconds() / 86400, 1),
            "rows_in_time_order_share": round(float((diffs >= pd.Timedelta(0)).mean()), 3),
        })
    facts["time_columns"] = time_cols
    for k in ("rows_with_target", "minority_share", "target_skew"):
        if k in facts:
            ctx.facts.add(k.replace("_", " "), facts[k])
    return facts


def rule_plan(f: Dict[str, Any]) -> Dict[str, Any]:
    if f["problem_type"] == "classification":
        if f["classes"] == 2 and f["minority_share"] < 0.1:
            metric, why = "average_precision", f"positives are rare ({round(f['minority_share'] * 100, 1)}%), so PR AUC measures how well models find them"
        elif f["minority_share"] < 0.25:
            metric, why = "f1_macro", "classes are imbalanced, so macro F1 weighs every class equally"
        else:
            metric, why = "f1_macro", "macro F1 balances precision and recall across classes"
        cv = "stratified"
    else:
        if abs(f.get("target_skew", 0)) > 2 or f.get("target_has_outliers"):
            metric, why = "mae", "the target is skewed / has extremes, so MAE is more robust and easier to interpret than R²"
        else:
            metric, why = "r2", "R² shows the share of target variation the model explains"
        cv = "kfold"
    plan: Dict[str, Any] = {"metric": metric, "cv": cv, "time_column": None}
    ordered = [t for t in f.get("time_columns", []) if t["rows_in_time_order_share"] >= 0.95 and t["span_days"] >= 7]
    if ordered:
        plan["cv"], plan["time_column"] = "time", ordered[0]["column"]
        why += f"; rows are recorded in time order by '{ordered[0]['column']}', so models are tested on later rows only (no peeking into the future)"
    rows = f["rows_with_target"]
    plan["max_rows"] = int(min(rows, 20000 if f["features"] <= 200 else 5000, MAX_PLAN_ROWS))
    if plan["max_rows"] < MAX_BENCHMARK_ROWS:
        plan["max_rows"] = rows
    plan["reason"] = why[0].upper() + why[1:] + "."
    return plan


def _validate(plan: Dict[str, Any], f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if plan.get("metric") not in METRICS[f["problem_type"]] or plan.get("cv") not in CV_STRATEGIES:
        return None
    if plan["metric"] in ("roc_auc", "average_precision") and f.get("classes", 2) != 2:
        return None
    if f["problem_type"] == "regression" and plan["cv"] == "stratified":
        plan["cv"] = "kfold"
    tc = plan.get("time_column")
    if plan["cv"] == "time" and tc not in {t["column"] for t in f.get("time_columns", [])}:
        return None
    try:
        rows = int(plan.get("max_rows") or f["rows_with_target"])
    except (TypeError, ValueError):
        rows = f["rows_with_target"]
    return {"metric": plan["metric"], "cv": plan["cv"], "time_column": tc if plan["cv"] == "time" else None,
            "max_rows": max(200, min(rows, f["rows_with_target"], MAX_PLAN_ROWS)), "reason": str(plan.get("reason") or "")[:400]}


def propose(ctx: RunContext) -> Dict[str, Any]:
    f = _facts(ctx)
    ctx.step(ACTOR, "tool", "Collected evaluation facts: target distribution, time columns, size.", data=f)
    plan = {**rule_plan(f), "source": "rules"}
    answer = ask_llm(ctx, ACTOR, SYSTEM, {"facts": f, "rule_plan": plan}, purpose="choose the evaluation metric and split")
    if isinstance(answer, dict):
        valid = _validate(dict(answer), f)
        if valid:
            plan = {**valid, "source": "agent"}
        else:
            ctx.step("Critic", "critic", "The agent's plan used a metric/split that doesn't fit this data; kept the rule-based plan.")
    label = METRICS[f["problem_type"]][plan["metric"]][0]
    plan["metric_label"] = label
    return plan


def save_plan(ctx: RunContext, plan: Dict[str, Any]) -> None:
    pol = P.get_policy(ctx.db, ctx.dataset.id, create=True)
    pol.eval_plan = plan
    ctx.db.commit()


UNSUPERVISED_NOTE = {
    "clustering": "Clustering has no target to split on: every method is scored by silhouette against the same data with "
                  "its columns shuffled (a reference with no real groups), so there is no plan to choose.",
    "anomaly_detection": "Anomaly detection has no labels: detectors are scored by how well they agree with each other, "
                         "against flagging the same number of rows at random, so there is no plan to choose.",
}


@register("evaluate")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    obj = ctx.db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    if obj and is_unsupervised(obj.problem_type):
        ctx.step(ACTOR, "result", UNSUPERVISED_NOTE[obj.problem_type])
        ctx.finish({"plan": None, "saved": False})
        return
    if response is None:
        plan = propose(ctx)
        ctx.save_state(plan=plan)
        split = {"time": f"time-ordered by '{plan['time_column']}'", "kfold": "shuffled K-fold", "stratified": "stratified K-fold"}[plan["cv"]]
        ctx.gate({
            "type": "evaluate",
            "title": "Approve the evaluation plan",
            "items": [{
                "id": "plan",
                "label": f"Score models by {plan['metric_label']}, {split}, up to {plan['max_rows']:,} rows",
                "reason": plan["reason"],
                "source": plan["source"],
                "destructive": False,
                "default": True,
            }],
            "plan": plan,
        })
        return
    plan = ctx.state.get("plan", {})
    if (response.get("approvals") or {}).get("plan", True):
        save_plan(ctx, plan)
        ctx.step(ACTOR, "result", f"Evaluation plan saved: {plan['metric_label']} with {plan['cv']} split. Re-run the benchmark to use it.")
        ctx.finish({"plan": plan, "saved": True})
    else:
        ctx.step(ACTOR, "result", "Plan rejected; the default evaluation stays in place.")
        ctx.finish({"plan": plan, "saved": False})
