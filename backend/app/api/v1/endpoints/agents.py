from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents import llm
from app.agents import policy as P
from app.agents.registry import AGENT_KINDS
from app.agents.runtime import ACTIVE, cancel_run, resume_run, start_run
from app.api.deps import get_client_id, get_owned_dataset
from app.core.database import get_db
from app.models.dataset import AgentRun, Dataset, Objective, Recommendation

router = APIRouter()
HEAVY_KINDS = {"autopilot", "understand", "clean", "evaluate", "report"}


class StartRunRequest(BaseModel):
    question: Optional[str] = Field(None, max_length=1000)
    history: Optional[List[Dict[str, Any]]] = None
    source: Optional[str] = Field("raw", pattern="^(raw|cleaned)$")


class ResumeRunRequest(BaseModel):
    approvals: Dict[str, bool] = {}
    answers: Dict[str, int] = {}


def _run_json(run: AgentRun, since: int = 0) -> Dict[str, Any]:
    steps = run.steps or []
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "kind": run.kind,
        "status": run.status,
        "input": run.input,
        "steps": steps[since:],
        "step_count": len(steps),
        "gate": run.gate if run.status == "waiting" else None,
        "result": run.result,
        "usage": run.usage,
        "error": run.error,
        "created_at": run.created_at.isoformat() + "Z" if run.created_at else None,
        "updated_at": run.updated_at.isoformat() + "Z" if run.updated_at else None,
    }


def _owned_run(run_id: str, client_id: str, db: Session) -> AgentRun:
    run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    ds = db.query(Dataset).filter(Dataset.id == run.dataset_id).first() if run else None
    if not run or not ds or (ds.owner_token and ds.owner_token != client_id):
        raise HTTPException(status_code=404, detail="Agent run not found.")
    return run


@router.get("/agents/status")
def agents_status():
    """Whether an LLM is configured, which model, and the privacy mode agents operate under."""
    return {**llm.status(), "agents": list(AGENT_KINDS)}


@router.post("/datasets/{dataset_id}/agents/{kind}")
def start_agent_run(
    kind: str,
    payload: StartRunRequest = StartRunRequest(),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Starts an agent run in the background. Poll GET /agent-runs/{id} for the live trace."""
    if kind not in AGENT_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown agent '{kind}'. Available: {list(AGENT_KINDS)}")
    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    if kind in ("autopilot", "evaluate", "clean") and not obj:
        raise HTTPException(status_code=400, detail="Set the ML objective (target and problem type) first.")
    if kind == "clean" and not db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).count():
        raise HTTPException(status_code=400, detail="Run diagnostics first so the Clean agent has a plan to review.")
    if kind == "ask" and not (payload.question or "").strip():
        raise HTTPException(status_code=400, detail="Ask a question.")

    if kind in HEAVY_KINDS:
        active = db.query(AgentRun).filter(AgentRun.dataset_id == db_dataset.id, AgentRun.status.in_(ACTIVE),
                                           AgentRun.kind.in_(HEAVY_KINDS)).all()
        for r in active:
            if r.status == "waiting":
                cancel_run(db, r)  # an abandoned approval card is superseded by the new run
            else:
                raise HTTPException(status_code=409, detail=f"Another agent run ({r.kind}) is in progress for this dataset. Wait or cancel it.")
    run = start_run(db, db_dataset, kind, payload.model_dump(exclude_none=True))
    return _run_json(run)


@router.get("/agent-runs/{run_id}")
def get_agent_run(
    run_id: str,
    since: int = Query(0, ge=0),
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
):
    return _run_json(_owned_run(run_id, client_id, db), since)


@router.post("/agent-runs/{run_id}/resume")
def resume_agent_run(
    run_id: str,
    payload: ResumeRunRequest,
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
):
    """Answers the gate the run is waiting on (approvals per item id, answers per question id)."""
    run = _owned_run(run_id, client_id, db)
    try:
        resume_run(db, run, payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _run_json(run)


@router.post("/agent-runs/{run_id}/cancel")
def cancel_agent_run(run_id: str, client_id: str = Depends(get_client_id), db: Session = Depends(get_db)):
    return _run_json(cancel_run(db, _owned_run(run_id, client_id, db)))


@router.get("/datasets/{dataset_id}/agents/runs")
def list_agent_runs(
    kind: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    q = db.query(AgentRun).filter(AgentRun.dataset_id == db_dataset.id)
    if kind:
        q = q.filter(AgentRun.kind == kind)
    return [_run_json(r) for r in q.order_by(AgentRun.created_at.desc()).limit(limit).all()]


@router.get("/datasets/{dataset_id}/agents/policy")
def get_agent_policy(db_dataset: Dataset = Depends(get_owned_dataset), db: Session = Depends(get_db)):
    pol = P.get_policy(db, db_dataset.id)
    return {
        "column_overrides": (pol.column_overrides if pol else None) or {},
        "eval_plan": pol.eval_plan if pol else None,
        "report": pol.report if pol else None,
    }


@router.delete("/datasets/{dataset_id}/agents/policy")
def reset_agent_policy(
    part: str = Query("all", pattern="^(all|columns|eval_plan|report)$"),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Removes approved agent decisions. Column changes take effect at the next diagnostics run."""
    pol = P.get_policy(db, db_dataset.id)
    if pol:
        if part in ("all", "columns"):
            pol.column_overrides = {}
        if part in ("all", "eval_plan"):
            pol.eval_plan = None
        if part in ("all", "report"):
            pol.report = None
        db.commit()
    return get_agent_policy(db_dataset, db)
