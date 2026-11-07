"""Helpers shared by the specialist agents."""
import json
from typing import Any, Dict, List, Optional

from app.agents import llm
from app.agents.runtime import RunContext


def ask_llm(
    ctx: RunContext,
    actor: str,
    system: str,
    payload: Any,
    *,
    purpose: str,
    max_tokens: int = 2200,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Any]:
    """
    One JSON-returning LLM call with trace logging. Returns None (after logging why) when no model is
    configured or every model failed, so callers can continue with their deterministic proposal.
    """
    if not llm.is_configured():
        ctx.step(actor, "warn", "No LLM configured (OPENROUTER_API_KEY). Using the deterministic rules only.")
        return None
    user = payload if isinstance(payload, str) else json.dumps(payload, default=str, ensure_ascii=False)
    ctx.step(actor, "llm", f"Asking the model to {purpose}…")
    try:
        parsed, reply = llm.chat_json(system, user, usage=ctx.usage, deadline=ctx.deadline, max_tokens=max_tokens, history=history)
    except llm.LLMUnavailable as e:
        ctx.step(actor, "warn", f"LLM unavailable: {e}. Continuing with deterministic rules.")
        return None
    except llm.LLMError as e:
        ctx.step(actor, "warn", f"The model did not give a usable answer ({e}). Continuing with deterministic rules.")
        return None
    ctx.step(actor, "llm", f"Model answered ({reply.model}).")
    return parsed


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
