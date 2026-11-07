"""
Analyst agent: answers questions about the data with a plan -> tool -> observe loop (read-only tools),
then the critic verifies every number in the answer against tool results before it is shown.
"""
import json
from typing import Any, Dict, List, Optional

from app.agents import llm
from app.agents import policy as P
from app.agents.critic import verify_numbers
from app.agents.runtime import RunContext, register
from app.agents.tools import DataTools, ToolError
from app.core.config import settings
from app.core.storage import read_dataset
from app.models.dataset import Objective

ACTOR = "Analyst agent"

SYSTEM_TEMPLATE = """You are the Analyst agent of a data analysis platform. Answer the user's question about a dataset
by calling tools step by step. The dataset is the DuckDB table `data` ({rows} rows). {target_line}

Tools:
{tools}

Protocol: reply with ONE JSON object per turn, no prose:
  {{"thought": "short plan", "action": "<tool name>", "args": {{...}}}}
or, when you have enough evidence:
  {{"thought": "...", "action": "final", "args": {{"answer": "markdown answer", "charts": ["C1"]}}}}

Rules:
- Every number in the final answer must come from a tool result; put the source in brackets right after it, e.g. "42% [T1]".
- Prefer aggregates (GROUP BY, AVG, COUNT) over listing rows. Quote column names with double quotes.
- Create a chart (from a sql table) when it helps; reference chart ids in "charts".
- Keep the answer short: 2-6 sentences or bullets, plain language, mention caveats (missing data, small groups).
- You have at most {max_steps} tool calls."""


def _observation(result: Dict[str, Any]) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    return text if len(text) < 6000 else text[:6000] + "…(truncated)"


@register("ask")
def run(ctx: RunContext, response: Optional[Dict[str, Any]]) -> None:
    question = str((ctx.run.input or {}).get("question", "")).strip()
    history = (ctx.run.input or {}).get("history") or []
    source = (ctx.run.input or {}).get("source", "raw")
    if not question:
        raise ValueError("No question was asked.")
    if not llm.is_configured():
        ctx.step(ACTOR, "warn", "The Analyst agent needs an LLM. Set OPENROUTER_API_KEY on the backend.")
        ctx.finish({"answer": "The Analyst agent needs an LLM to plan queries. Add `OPENROUTER_API_KEY` to `backend/.env` and restart the backend.",
                    "charts": [], "tables": [], "verification": None})
        return

    db = ctx.db
    obj = db.query(Objective).filter(Objective.dataset_id == ctx.dataset.id).first()
    if source == "cleaned" and ctx.dataset.cleaned_file_path:
        df = read_dataset(ctx.dataset.cleaned_file_path)
    else:
        df, _, _ = P.load_prepared(db, ctx.dataset)
    tools = DataTools(df, ctx.facts)
    system = SYSTEM_TEMPLATE.format(
        rows=len(df),
        target_line=(f"The prediction target is \"{obj.target_column}\" ({obj.problem_type})." if obj and obj.target_column
                     else f"The task is {obj.problem_type.replace('_', ' ')} (no target column)." if obj else ""),
        tools=json.dumps(DataTools.SPEC, indent=1),
        max_steps=settings.AGENT_MAX_STEPS,
    )
    ctx.step("Orchestrator", "plan", f"Routing the question to the Analyst agent with read-only tools (max {settings.AGENT_MAX_STEPS} calls).")

    messages: List[Dict[str, str]] = []
    for h in history[-3:]:
        messages += [{"role": "user", "content": f"Earlier question: {h.get('q', '')}"},
                     {"role": "assistant", "content": json.dumps({"action": "final", "args": {"answer": str(h.get('a', ''))[:800]}})}]
    first = f"Question: {question}\nStart by calling schema unless you already know the columns."
    final: Optional[Dict[str, Any]] = None
    calls = 0
    revised = False
    forced = False
    user_msg = first

    while True:
        try:
            parsed, reply = llm.chat_json(system, user_msg, usage=ctx.usage, deadline=ctx.deadline, history=messages, max_tokens=1400)
        except llm.LLMError as e:
            message = llm.friendly_error(e)
            ctx.step(ACTOR, "error", message)
            ctx.finish({"question": question, "answer": message, "charts": [], "tables": [], "facts": [], "verification": None,
                        "error": True})
            return
        messages += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": json.dumps(parsed, default=str)[:4000]}]
        action = str(parsed.get("action", "")) if isinstance(parsed, dict) else ""
        args = parsed.get("args") if isinstance(parsed, dict) and isinstance(parsed.get("args"), dict) else {}
        thought = str(parsed.get("thought", ""))[:300] if isinstance(parsed, dict) else ""

        if action == "final":
            answer = str(args.get("answer", "")).strip()
            verified, unverified = verify_numbers(answer, ctx.facts)
            if unverified and not revised and calls < settings.AGENT_MAX_STEPS:
                revised = True
                ctx.step("Critic", "critic", f"Numbers {unverified[:6]} are not backed by any tool result. Sending the answer back for revision.")
                user_msg = (f"Critic: these numbers are not in any tool result: {unverified[:10]}. "
                            "Run the queries needed to support them, or remove them, then give the final answer again.")
                continue
            final = {"answer": answer, "chart_ids": [str(c) for c in (args.get("charts") or []) if isinstance(c, str)],
                     "verification": {"verified_numbers": verified, "unverified_numbers": unverified}}
            ctx.step("Critic", "critic", f"Verified {verified} number(s) against tool results" +
                     (f"; {len(unverified)} unverified ({', '.join(unverified[:5])})." if unverified else "; all numbers are backed by data."))
            break

        if calls >= settings.AGENT_MAX_STEPS:
            if forced:
                final = {"answer": "I ran out of steps before reaching a reliable answer. Try a narrower question.",
                         "chart_ids": [], "verification": None}
                break
            forced = True
            user_msg = "Tool budget exhausted. Give the final answer now using only the results you have."
            continue

        calls += 1
        if thought:
            ctx.step(ACTOR, "plan", thought)
        try:
            result = tools.run(action, args)
            summary = {"schema": "Read the schema.", "sql": f"SQL → {result.get('table_id')} ({result.get('row_count')} rows)",
                       "stat_test": "Statistical test done.", "chart": f"Chart {result.get('chart_id')} created."}.get(action, action)
            ctx.step("Tool", "tool", f"{action}: {summary}", data={"args": args, "result": result if action != "schema" else None})
            user_msg = f"Observation: {_observation(result)}"
        except ToolError as e:
            ctx.step("Tool", "warn", f"{action} failed: {e}", data={"args": args})
            user_msg = f"Observation: ERROR {e}"

    charts = [c for c in tools.charts if c["id"] in final.get("chart_ids", [])] or tools.charts[:2]
    used_tables = [t for t in ctx.facts.tables if f"[{t['id']}" in final["answer"] or any(c["table"] == t["id"] for c in charts)]
    ctx.step(ACTOR, "result", "Answer ready.")
    ctx.finish({
        "question": question,
        "answer": final["answer"],
        "charts": charts,
        "tables": [{**t, "rows": t["rows"][:50]} for t in used_tables[:4]],
        "facts": [f for f in ctx.facts.facts if f"[{f['id']}" in final["answer"]],
        "verification": final.get("verification"),
    })
