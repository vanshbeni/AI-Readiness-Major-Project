"""
Orchestrator: plans and drives a full pipeline run across the specialist agents, pausing at human gates.

Autopilot plan:
  1. Understand agent proposes column decisions          -> gate (changes how data is read)
  2. Diagnose with the approved decisions (engine)
  3. Clean agent reviews the fix plan                     -> gate (changes what is removed/imputed)
  4. Execute the approved fixes (engine)
  5. Evaluate agent picks metric / split (non-destructive, applied and shown)
  6. Benchmark (engine)
  7. Report agent writes the verified summary
"""
from typing import Any, Dict, Optional

from app.agents import clean, evaluate, llm, report, understand
from app.agents.runtime import RunContext, register
from app.engine.tasks import is_unsupervised
from app.models.dataset import Objective
from app.services.pipeline import ServiceError, benchmark, execute, run_diagnostics

ACTOR = "Orchestrator"


def _service(ctx: RunContext, label: str, fn, *args, **kwargs):
    ctx.step(ACTOR, "tool", f"{label}…")
    try:
        return fn(ctx.db, ctx.dataset, *args, **kwargs)
    except ServiceError as e:
        raise RuntimeError(f"{label} failed: {e.detail}")


@register("autopilot")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    phase = ctx.state.get("phase", "start")

    if phase == "start":
        mode = f"model {llm.status()['model']}" if llm.is_configured() else "deterministic rules (no LLM key)"
        ctx.step(ACTOR, "plan", "Plan: understand columns → diagnose → plan fixes → clean → choose evaluation → "
                                f"benchmark → report. I'll stop for your approval before anything changes the data. Using {mode}.")
        proposals, questions, notes = understand.propose(ctx)
        if proposals or questions:
            ctx.save_state(phase="understand_gate", u_proposals=proposals, u_questions=questions, notes=notes)
            ctx.gate({"type": "understand", "stage": "Step 1 of 2",
                      "title": f"Approve how {len(proposals)} column decision(s) should be read",
                      "items": proposals, "questions": questions, "notes": notes})
        ctx.step(understand.ACTOR, "result", "Every column already reads correctly.")
        phase = "diagnose"

    if phase == "understand_gate":
        selected = understand.selected_from_response(ctx.state.get("u_proposals", []), ctx.state.get("u_questions", []), response or {})
        applied = understand.apply_decisions(ctx, selected)
        ctx.step(understand.ACTOR, "result", f"Applied {len(applied)} column decision(s).", data=applied)
        phase = "diagnose"

    if phase == "diagnose":
        diag = _service(ctx, "Running diagnostics with the approved column decisions", run_diagnostics, explain=llm.is_configured())
        hs = diag.health_score
        ctx.step(ACTOR, "result", f"Health score {hs.composite_score}/100 ({hs.grade}); {len(diag.issues)} issue(s), "
                                  f"{len(diag.recommendations)} recommended fix(es).")
        proposals = clean.propose(ctx)
        if proposals:
            ctx.save_state(phase="clean_gate", c_proposals=proposals)
            ctx.gate({"type": "clean", "stage": "Step 2 of 2",
                      "title": f"Approve {len(proposals)} change(s) to the cleaning plan", "items": proposals})
        ctx.step(clean.ACTOR, "result", "The rule-based cleaning plan already fits; no changes proposed.")
        phase = "execute"

    if phase == "clean_gate":
        approvals = (response or {}).get("approvals") or {}
        selected = [p for p in ctx.state.get("c_proposals", []) if approvals.get(p["id"], p.get("default", True))]
        changed = clean.apply_proposals(ctx, selected)
        ctx.step(clean.ACTOR, "result", f"Updated {len(changed)} recommendation(s).", data=changed)
        phase = "execute"

    if phase == "execute":
        ctx.save_state(phase="execute")
        ex = _service(ctx, "Executing the approved cleaning plan", execute)
        ctx.step(ACTOR, "result", ex.message, data={"before": ex.before_health_score, "after": ex.after_health_score,
                                                    "rows": ex.cleaned_rows, "cols": ex.cleaned_cols})
        fixed = [a for a in ex.quality_audit if a.get("status") == "fixed"]
        still_open = [a for a in ex.quality_audit if a.get("status") == "open"]
        ctx.step("Quality auditor", "critic",
                 f"Audited the exported data: {len(fixed)} defect(s) repaired and re-checked, {len(still_open)} still open.",
                 data=[f"[{a['status']}] {a['detail']}" for a in fixed + still_open][:12])
        obj = ctx.db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
        if obj and is_unsupervised(obj.problem_type):
            ctx.step(evaluate.ACTOR, "result", evaluate.UNSUPERVISED_NOTE[obj.problem_type])
        else:
            plan = evaluate.propose(ctx)
            evaluate.save_plan(ctx, plan)
            ctx.step(evaluate.ACTOR, "result", f"Evaluation plan applied: {plan['metric_label']}, {plan['cv']} split, "
                                               f"up to {plan['max_rows']:,} rows. {plan['reason']} (You can change it on the results page.)")
        lb = _service(ctx, "Benchmarking candidate models", benchmark)
        best = lb.models[0]
        ctx.step(ACTOR, "result", f"Best model: {best.model_name} with {lb.primary_metric} {best.metric_display}.")
        rep = report.write(ctx)
        ctx.step(report.ACTOR, "result", "Report written" + (" and verified." if rep["source"] == "agent" else " from the template."))
        ctx.finish({
            "health_before": ex.before_health_score,
            "health_after": ex.after_health_score,
            "best_model": best.model_name,
            "metric": lb.primary_metric,
            "score": best.metric_display,
            "report": rep,
        })
