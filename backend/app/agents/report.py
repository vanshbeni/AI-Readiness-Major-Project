"""
Report agent: writes the executive summary. Every number comes from the facts store through {{F#}}
placeholders that code fills in, so the text cannot contain invented figures; the critic rejects stray numbers.
"""
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents import policy as P
from app.agents.base import ask_llm
from app.agents.critic import stray_numbers
from app.agents.facts import fmt
from app.agents.runtime import RunContext, register
from app.models.dataset import DatasetSnapshot, Objective

ACTOR = "Report agent"

SYSTEM = """You are the Report agent. Write a concise executive report (markdown) about a dataset's ML readiness.
You get a list of FACTS (id, label, value) and short context. Rules:
- NEVER type a number yourself. Every number must be a placeholder like {{F3}} referring to a fact id.
- Use only facts that exist. Do not speculate beyond the facts.
- Sections (### headings): Summary, Data quality, What was fixed, Model results, Caveats, Next steps.
- Plain language for a non-expert, 150-280 words total, bullet points where helpful.
Return ONLY JSON: {"markdown": "..."}"""


def gather(ctx: RunContext) -> Dict[str, Any]:
    db = ctx.db
    f = ctx.facts
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    snap = db.query(DatasetSnapshot).filter(DatasetSnapshot.dataset_id == ctx.dataset.id).first()
    pol = P.get_policy(db, ctx.dataset.id)
    context: Dict[str, Any] = {"dataset": ctx.dataset.filename}
    if obj:
        context.update(target=obj.target_column, problem_type=obj.problem_type)
    f.add("rows in the uploaded file", ctx.dataset.row_count)
    f.add("columns in the uploaded file", ctx.dataset.col_count)

    diag = snap.diagnostics if snap else None
    if diag:
        hs = diag["health_score"]
        f.add("health score before cleaning (out of 100)", hs["composite_score"])
        context["grade_before"] = hs["grade"]
        sev = Counter(i["severity"] for i in diag.get("issues", []))
        f.add("issues detected", len(diag.get("issues", [])))
        for s in ("critical", "high", "medium", "low"):
            if sev.get(s):
                f.add(f"{s} severity issues", sev[s])
        context["top_issues"] = [i["title"] for i in sorted(
            diag.get("issues", []), key=lambda i: ["critical", "high", "medium", "low", "info"].index(i["severity"]) if i["severity"] in ["critical", "high", "medium", "low", "info"] else 9
        )[:5]]
        recs = diag.get("recommendations", [])
        f.add("fixes recommended", len(recs))
    overrides = (pol.column_overrides if pol else None) or {}
    if overrides:
        f.add("columns with approved meaning decisions (Understand agent)", len(overrides))
    ex = snap.execution if snap else None
    if ex:
        f.add("health score after cleaning (out of 100)", ex["after_health_score"])
        f.add("health score change (points)", ex["health_score_delta"])
        f.add("cleaning steps applied", len(ex.get("transformations_applied", [])))
        f.add("rows after cleaning", ex["cleaned_rows"])
        f.add("columns after cleaning (model-ready)", ex["cleaned_cols"])
        context["steps"] = ex.get("transformations_applied", [])[:8]
        audit = ex.get("quality_audit") or []
        fixed = [a for a in audit if a.get("status") == "fixed"]
        still_open = [a for a in audit if a.get("status") == "open"]
        if audit:
            f.add("output defects found and fixed by the post-cleaning audit", len(fixed))
            f.add("output defects still open after the audit", len(still_open))
            context["audit_fixed"] = [a["detail"] for a in fixed][:6]
            context["audit_open"] = [a["detail"] for a in still_open][:6]
            context["caveats"] = [a["detail"] for a in audit if a.get("status") == "advice"
                                  and a.get("severity") in ("medium", "low", "high")][:5]
    lb = snap.leaderboard if snap else None
    if lb and lb.get("models"):
        models = [m for m in lb["models"] if not m.get("is_baseline")]
        base = next((m for m in lb["models"] if m.get("is_baseline")), None)
        best = models[0]
        context.update(best_model=best["model_name"], metric=lb["primary_metric"])
        f.add(f"best model {lb['primary_metric']}", best["metric_value"])
        if best.get("metric_std") is not None:
            f.add("best model score variation across folds (std)", best["metric_std"])
        if base:
            f.add(f"no-skill baseline {lb['primary_metric']}", base["metric_value"])
        f.add("models compared", len(models))
        ev = lb.get("evaluation") or {}
        if ev:
            f.add("rows used for benchmarking", ev.get("rows_used"))
            f.add("labelled rows available", ev.get("rows_available"))
            context["evaluation"] = {"metric": ev.get("metric_label"), "split": ev.get("cv"), "chosen_by": ev.get("source"),
                                     "reason": ev.get("reason")}
            if ev.get("verdict"):
                context["model_verdict"] = {
                    "no_signal": "NO model beats the no-skill baseline beyond noise: the features carry no usable signal; "
                                 "say clearly that the data is not ready for modelling.",
                    "weak": "the best model beats the baseline only slightly relative to fold noise: call it weak.",
                    "useful": "the best model beats the baseline clearly.",
                }[ev["verdict"]]
    return context


def fallback_markdown(ctx: RunContext, context: Dict[str, Any]) -> str:
    def val(label_start: str) -> Optional[str]:
        fact = next((x for x in ctx.facts.facts if x["label"].startswith(label_start)), None)
        return fmt(fact["value"]) if fact else None

    lines = ["### Summary"]
    target = context.get("target")
    lines.append(f"- `{context['dataset']}` was prepared for predicting **{target}** ({context.get('problem_type')})." if target else
                 f"- `{context['dataset']}` was profiled.")
    if val("health score before"):
        after = val("health score after")
        lines.append(f"- Health score {val('health score before')}/100" + (f" → {after}/100 after cleaning." if after else "."))
    lines.append("### Data quality")
    if val("issues detected"):
        lines.append(f"- {val('issues detected')} issues detected; top: " + "; ".join(context.get("top_issues", [])[:3]) + ".")
    if context.get("steps"):
        lines.append("### What was fixed")
        lines += [f"- {s}" for s in context["steps"][:6]]
    if context.get("best_model"):
        lines.append("### Model results")
        base = val("no-skill baseline")
        lines.append(f"- Best model: **{context['best_model']}** with {context['metric']} {val('best model')}"
                     + (f" (baseline {base})." if base else "."))
        if context.get("evaluation", {}).get("reason"):
            lines.append(f"- Evaluation: {context['evaluation']['reason']}")
        if context.get("model_verdict"):
            lines.append(f"- Verdict: {context['model_verdict'].split(':')[0].capitalize()}.")
    caveats = context.get("audit_open", []) + context.get("caveats", [])
    if caveats:
        lines.append("### Caveats")
        lines += [f"- {c}" for c in caveats[:5]]
    lines.append("### Next steps")
    lines.append("- Review any skipped fixes, then export the cleaned data and pipeline script.")
    return "\n".join(lines)


def write(ctx: RunContext) -> Dict[str, Any]:
    context = gather(ctx)
    ctx.step(ACTOR, "tool", f"Collected {len(ctx.facts.facts)} verified facts for the report.", data=ctx.facts.facts)
    payload = {"facts": ctx.facts.facts, "context": context}
    source, markdown = "rules", None
    for attempt in range(2):
        answer = ask_llm(ctx, ACTOR, SYSTEM, payload, purpose="write the executive report" if attempt == 0 else "revise the report without typed numbers",
                         max_tokens=1600)
        if not isinstance(answer, dict) or not isinstance(answer.get("markdown"), str):
            break
        template = answer["markdown"]
        rendered, unknown = ctx.facts.render(template)
        stray = stray_numbers(template)
        if not unknown and not stray:
            source, markdown = "agent", rendered
            ctx.step("Critic", "critic", "Report verified: every number comes from a fact placeholder.")
            break
        ctx.step("Critic", "critic", f"Report rejected: unknown facts {unknown[:5]} / typed numbers {stray[:5]}.")
        payload = {**payload, "feedback": f"Your draft typed numbers {stray[:8]} or used unknown fact ids {unknown[:5]}. Use only {{{{F#}}}} placeholders."}
    if markdown is None:
        markdown = fallback_markdown(ctx, context)
    report = {"markdown": markdown, "source": source, "facts": ctx.facts.facts, "generated_at": datetime.utcnow().isoformat() + "Z"}
    pol = P.get_policy(ctx.db, ctx.dataset.id, create=True)
    pol.report = report
    ctx.db.commit()
    return report


@register("report")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    report = write(ctx)
    ctx.step(ACTOR, "result", f"Report ready ({'written by the agent, verified' if report['source'] == 'agent' else 'deterministic template'}).")
    ctx.finish({"report": report})
