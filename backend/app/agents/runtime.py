"""
Orchestrator runtime: background execution of agent runs, live traces, resumable human gates, budgets.

A run is a small state machine. Handlers record steps through `RunContext.step`, may pause with
`RunContext.gate(...)` (status -> "waiting"), and are re-invoked with the user's response on resume.
Swap the in-process executor for a Redis-backed queue (RQ/Arq) without touching the agents.
"""
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.agents.facts import FactStore
from app.agents.llm import Usage
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.storage import json_safe
from app.models.dataset import AgentRun, Dataset

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")
HANDLERS: Dict[str, Callable[["RunContext", Optional[Dict[str, Any]]], None]] = {}
ACTIVE = ("queued", "running", "waiting")


class GatePause(Exception):
    """Raised by a handler to pause the run until the user responds to a gate."""


class RunCancelled(Exception):
    pass


def register(kind: str):
    def deco(fn):
        HANDLERS[kind] = fn
        return fn
    return deco


class RunContext:
    def __init__(self, db: Session, run: AgentRun):
        self.db = db
        self.run = run
        self.dataset: Dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        self.state: Dict[str, Any] = dict(run.state or {})
        self.facts = FactStore(self.state.get("facts"))
        self.usage = Usage(**{k: v for k, v in (run.usage or {}).items() if k in ("calls", "prompt_tokens", "completion_tokens", "models")})
        started = self.state.get("started_monotonic_offset", 0.0)
        self.deadline = time.monotonic() + max(30.0, settings.AGENT_RUN_BUDGET_SEC - started)
        self._t0 = time.monotonic()

    # ── trace ──
    def step(self, actor: str, kind: str, text: str, data: Any = None) -> None:
        self.check_cancelled()
        steps = list(self.run.steps or [])
        steps.append({
            "i": len(steps),
            "t": datetime.utcnow().isoformat() + "Z",
            "actor": actor,
            "kind": kind,      # plan | tool | llm | critic | gate | result | warn | error
            "text": text,
            "data": json_safe(data) if data is not None else None,
        })
        self.run.steps = steps
        self._persist()

    def save_state(self, **updates) -> None:
        self.state.update(updates)
        self._persist()

    def gate(self, gate: Dict[str, Any]) -> None:
        """Pauses the run for user approval. The handler is called again with the response."""
        self.step("Human gate", "gate", gate.get("title", "Waiting for your approval"))
        self.run.gate = json_safe(gate)
        self.run.status = "waiting"
        self._persist()
        raise GatePause()

    def finish(self, result: Any) -> None:
        self.run.result = json_safe(result)
        self.run.status = "completed"
        self.run.gate = None
        self._persist()

    def check_cancelled(self) -> None:
        self.db.refresh(self.run, attribute_names=["status"])
        if self.run.status == "cancelled":
            raise RunCancelled()

    def _persist(self) -> None:
        self.state["facts"] = self.facts.to_dict()
        self.run.state = json_safe(self.state)
        self.run.usage = self.usage.as_dict()
        self.run.updated_at = datetime.utcnow()
        self.db.commit()


def _execute(run_id: str, response: Optional[Dict[str, Any]]) -> None:
    db = SessionLocal()
    try:
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        if not run or run.status == "cancelled":
            return
        handler = HANDLERS.get(run.kind)
        run.status = "running"
        db.commit()
        ctx = RunContext(db, run)
        if not ctx.dataset:
            raise RuntimeError("Dataset no longer exists.")
        try:
            handler(ctx, response)
            if run.status == "running":
                ctx.finish(run.result or {})
        except GatePause:
            ctx.state["started_monotonic_offset"] = ctx.state.get("started_monotonic_offset", 0.0) + (time.monotonic() - ctx._t0)
            ctx._persist()
        except RunCancelled:
            pass
    except Exception as e:
        logger.error(f"Agent run {run_id} failed: {e}\n{traceback.format_exc()}")
        db.rollback()
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        if run and run.status != "cancelled":
            steps = list(run.steps or [])
            steps.append({"i": len(steps), "t": datetime.utcnow().isoformat() + "Z", "actor": "Orchestrator",
                          "kind": "error", "text": f"Run failed: {str(e)[:400]}", "data": None})
            run.steps = steps
            run.status = "failed"
            run.error = str(e)[:2000]
            db.commit()
    finally:
        db.close()


def start_run(db: Session, dataset: Dataset, kind: str, payload: Optional[Dict[str, Any]] = None) -> AgentRun:
    if kind not in HANDLERS:
        raise ValueError(f"Unknown agent '{kind}'.")
    run = AgentRun(dataset_id=dataset.id, kind=kind, status="queued", input=json_safe(payload or {}), steps=[], state={})
    db.add(run)
    db.commit()
    db.refresh(run)
    _EXECUTOR.submit(_execute, run.id, None)
    return run


def resume_run(db: Session, run: AgentRun, response: Dict[str, Any]) -> AgentRun:
    if run.status != "waiting":
        raise ValueError("This run is not waiting for input.")
    run.status = "queued"
    db.commit()
    _EXECUTOR.submit(_execute, run.id, response)
    return run


def cancel_run(db: Session, run: AgentRun) -> AgentRun:
    if run.status in ACTIVE:
        run.status = "cancelled"
        run.gate = None
        db.commit()
    return run


def fail_stale_runs() -> int:
    """Runs left queued/running by a previous process can never finish; mark them failed on startup."""
    db = SessionLocal()
    try:
        stale = db.query(AgentRun).filter(AgentRun.status.in_(["queued", "running"])).all()
        for r in stale:
            r.status = "failed"
            r.error = "The server restarted while this run was in progress. Start it again."
        db.commit()
        return len(stale)
    finally:
        db.close()
