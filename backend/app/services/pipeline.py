"""
Pipeline services shared by the HTTP endpoints and the agent orchestrator:
diagnose -> execute -> benchmark. Every service applies the dataset's approved policy (prep steps and
column-type overrides) so agent decisions take effect everywhere.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents import policy as P
from app.api.deps import get_snapshot, invalidate_execution
from app.core.config import settings
from app.core.storage import dataset_lock, json_safe, read_dataset, remove_files
from app.engine.benchmark import BenchmarkError, benchmark_models
from app.engine import normalize as N
from app.engine.detector import detect_issues
from app.engine.tasks import is_unsupervised, target_or_none
from app.engine.unsupervised import run_unsupervised
from app.engine.executor import execute_pipeline
from app.engine.explainer import _apply_fallback, explain_recommendations
from app.engine.profiler import count_duplicate_records, identifier_columns, profile_dataset, type_overrides
from app.engine.rules import select_remediation_methods
from app.engine.scorer import calculate_health_score
from app.models.dataset import (
    Dataset,
    HealthScore,
    IssueDetection,
    ModelBenchmark,
    Objective,
    ProcessingJob,
    ProfileReport,
    Recommendation,
)
from app.schemas.dataset import (
    ExecutionResponse,
    FullDiagnosticResponse,
    HealthScoreResponse,
    IssueItem,
    MetricDelta,
    ModelBenchmarkItem,
    ModelBenchmarkLeaderboardResponse,
    ObjectiveResponse,
    ProfileReportResponse,
    RecommendationItem,
)

logger = logging.getLogger(__name__)


def data_dictionary_path(dataset_id: str) -> str:
    return os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_data_dictionary.json")


def assignments_file(dataset_id: str) -> str:
    """Cleaned rows plus the cluster id / anomaly flag found by the unsupervised benchmark."""
    return os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_assignments.csv")


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _objective(db: Session, dataset_id: str) -> Objective:
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise ServiceError(400, "Set the ML objective (target column and problem type) before running diagnostics.")
    return obj


def _signed(value, suffix: str) -> str:
    return f"+{value}{suffix}" if value > 0 else f"{value}{suffix}"


# ─── Diagnose ────────────────────────────────────────────────────────────────

def run_diagnostics(db: Session, db_dataset: Dataset, explain: bool = True) -> FullDiagnosticResponse:
    """Profiling -> issue detection -> health score -> rule matrix -> explanations. Clears downstream results."""
    from app.api.v1.endpoints.datasets import build_dataset_response

    dataset_id = db_dataset.id
    obj = _objective(db, dataset_id)

    with dataset_lock(dataset_id):
        prob_type = obj.problem_type
        target_col = target_or_none(prob_type, obj.target_column)
        overrides = P.column_overrides(db, dataset_id)
        types = P.type_map(overrides)
        df, _, auto_steps = P.prepare(read_dataset(db_dataset.raw_file_path), overrides)
        if target_col and target_col not in df.columns:
            raise ServiceError(400, f"Target column '{target_col}' is not available after the approved column decisions.")

        with type_overrides(types):
            summary_stats, col_profiles = profile_dataset(df, target_col=target_col)
            detected_issues = N.issues_from(auto_steps, target_col) + detect_issues(df, target_col=target_col, problem_type=prob_type)
        comp_score, sub_scores, grade, summary_text, drivers = calculate_health_score(
            summary_stats, col_profiles, detected_issues, problem_type=prob_type
        )
        raw_recommendations = select_remediation_methods(detected_issues, col_profiles, problem_type=prob_type)
        explained_recs = (
            explain_recommendations(raw_recommendations) if explain else _apply_fallback(raw_recommendations)
        )

        summary_stats = json_safe(summary_stats)
        col_profiles = json_safe(col_profiles)

        try:
            db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).delete()
            db.query(IssueDetection).filter(IssueDetection.dataset_id == dataset_id).delete()
            db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").delete()
            db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).delete()
            invalidate_execution(db, db_dataset)

            db.add(ProfileReport(dataset_id=dataset_id, column_profiles=col_profiles, summary_stats=summary_stats))

            issue_items = []
            for issue in detected_issues:
                issue_id = str(uuid.uuid4())
                details = json_safe(issue["details"])
                db.add(IssueDetection(
                    id=issue_id, dataset_id=dataset_id, issue_type=issue["issue_type"],
                    column=issue.get("column"), severity=issue["severity"], details=details,
                ))
                issue_items.append(IssueItem(
                    id=issue_id, issue_type=issue["issue_type"], column=issue.get("column"),
                    severity=issue["severity"], title=issue["title"], details=details,
                ))

            db.add(HealthScore(
                dataset_id=dataset_id, stage="before", composite_score=comp_score,
                sub_scores=sub_scores, summary_text=summary_text,
            ))

            rec_items = []
            for rec in explained_recs:
                rec_id = str(uuid.uuid4())
                db.add(Recommendation(
                    id=rec_id,
                    dataset_id=dataset_id,
                    issue_type=rec["issue_type"],
                    column=rec.get("column"),
                    method=rec["method"],
                    reason_title=rec["reason_title"],
                    explanation_text=rec["explanation_text"],
                    explanation_source=rec.get("explanation_source", "statistical"),
                    params=json_safe(rec.get("stats_context", {})),
                    severity=rec["severity"],
                    is_destructive=rec["is_destructive"],
                    is_approved=rec["is_approved"],
                ))
                rec_items.append(RecommendationItem(
                    id=rec_id,
                    issue_type=rec["issue_type"],
                    column=rec.get("column"),
                    method=rec["method"],
                    reason_title=rec["reason_title"],
                    explanation_text=rec["explanation_text"],
                    explanation_source=rec.get("explanation_source", "statistical"),
                    severity=rec["severity"],
                    is_destructive=rec["is_destructive"],
                    is_approved=rec["is_approved"],
                ))
            db.flush()

            response = FullDiagnosticResponse(
                dataset=build_dataset_response(db, db_dataset),
                objective=ObjectiveResponse.model_validate(obj),
                profile=ProfileReportResponse(dataset_id=dataset_id, summary=summary_stats, columns=col_profiles),
                health_score=HealthScoreResponse(
                    stage="before",
                    composite_score=comp_score,
                    grade=grade,
                    sub_scores=sub_scores,
                    summary_text=summary_text,
                    top_negative_drivers=drivers,
                ),
                issues=issue_items,
                recommendations=rec_items,
            )
            get_snapshot(db, dataset_id).diagnostics = json_safe(response.model_dump(mode="json"))
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ServiceError(409, "Diagnostics were updated concurrently. Please retry.")

    return response


# ─── Execute ─────────────────────────────────────────────────────────────────

def execute(db: Session, db_dataset: Dataset) -> ExecutionResponse:
    """Executes approved prep + fixes, stores the cleaned dataset and script, re-scores and reports deltas."""
    dataset_id = db_dataset.id
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    before_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    if not obj or not before_health:
        raise ServiceError(400, "Run diagnostics before executing the cleaning pipeline.")

    with dataset_lock(dataset_id):
        prob_type = obj.problem_type
        target_col = target_or_none(prob_type, obj.target_column)
        db_recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
        rec_dicts = [
            {"id": r.id, "issue_type": r.issue_type, "column": r.column, "method": r.method,
             "params": r.params or {}, "is_approved": bool(r.is_approved)}
            for r in db_recs
        ]
        before_score = before_health.composite_score

        overrides = P.column_overrides(db, dataset_id)
        types = P.type_map(overrides)
        df_raw = read_dataset(db_dataset.raw_file_path)
        df_prepared, prep_ops, _ = P.prepare(df_raw, overrides)
        with type_overrides(types):
            raw_missing = int(df_prepared.isna().sum().sum())
            raw_dups = count_duplicate_records(df_prepared)

        invalidate_execution(db, db_dataset)
        job = ProcessingJob(id=str(uuid.uuid4()), dataset_id=dataset_id, status="running")
        db.add(job)
        db.commit()

        cleaned_path = os.path.join(settings.CLEANED_DATA_DIR, f"{dataset_id}_cleaned.csv")
        script_path = os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_pipeline.py")
        dictionary_path = data_dictionary_path(dataset_id)
        try:
            with type_overrides(types):
                result = execute_pipeline(
                    df_raw=df_raw, recommendations=rec_dicts, target_col=target_col, problem_type=prob_type,
                    prep_ops=prep_ops,
                )
                df_cleaned, df_final = result.df_cleaned, result.df_model_ready

                # Re-score the cleaned (pre-encoding) data so the comparison is like-for-like with the raw data.
                summary_after, profiles_after = profile_dataset(df_cleaned, target_col=target_col)
                issues_after = detect_issues(df_cleaned, target_col=target_col, problem_type=prob_type)
                raw_ids = set(identifier_columns(df_prepared))
            informative_features = [c for c in df_prepared.columns if c != target_col and c not in raw_ids]
            dropped_ratio = len(result.dropped_informative_columns) / max(len(informative_features), 1)
            after_score, sub_scores_after, _, summary_text_after, _ = calculate_health_score(
                summary_after, profiles_after, issues_after, problem_type=prob_type,
                dropped_informative_ratio=dropped_ratio,
            )

            df_final.to_csv(cleaned_path, index=False)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(result.script_code)
            with open(dictionary_path, "w", encoding="utf-8") as f:
                json.dump(json_safe({"columns": result.data_dictionary, "quality_audit": result.audit}), f, indent=2)
        except Exception as e:
            logger.exception("Pipeline execution failed")
            remove_files(cleaned_path, script_path, dictionary_path)
            job.status = "failed"
            job.error_message = str(e)[:2000]
            job.completed_at = datetime.utcnow()
            db.commit()
            raise ServiceError(500, f"Pipeline transformation failed: {e}")

        db_dataset.cleaned_file_path = cleaned_path
        db.add(HealthScore(
            dataset_id=dataset_id, stage="after", composite_score=after_score,
            sub_scores=sub_scores_after, summary_text=summary_text_after,
        ))
        job.status = "completed"
        job.pipeline_file_path = cleaned_path
        job.script_file_path = script_path
        job.applied_steps = result.applied_steps
        job.completed_at = datetime.utcnow()

        cleaned_missing = int(df_final.isna().sum().sum())
        with type_overrides(types):
            cleaned_dups = count_duplicate_records(df_cleaned)
        delta_score = round(after_score - before_score, 1)

        if delta_score > 0:
            message = f"Pipeline executed successfully. Health Score improved by {delta_score} points."
        elif delta_score == 0:
            message = "Pipeline executed successfully. Health Score is unchanged."
        else:
            message = f"Pipeline executed, but the Health Score dropped by {abs(delta_score)} points. Review the approved fixes."

        response = ExecutionResponse(
            job_id=job.id,
            status="completed",
            message=message,
            before_health_score=before_score,
            after_health_score=after_score,
            health_score_delta=delta_score,
            transformations_applied=result.applied_steps,
            metric_deltas=[
                MetricDelta(metric_name="Data Health Score", before_value=f"{before_score} / 100",
                            after_value=f"{after_score} / 100", improvement=_signed(delta_score, " pts")),
                MetricDelta(metric_name="Missing Values", before_value=f"{raw_missing} cells",
                            after_value=f"{cleaned_missing} cells", improvement=_signed(cleaned_missing - raw_missing, " cells")),
                MetricDelta(metric_name="Duplicate Rows", before_value=f"{raw_dups} rows",
                            after_value=f"{cleaned_dups} rows", improvement=_signed(cleaned_dups - raw_dups, " rows")),
                MetricDelta(metric_name="Dataset Dimensions",
                            before_value=f"{df_raw.shape[0]} rows × {df_raw.shape[1]} cols",
                            after_value=f"{df_final.shape[0]} rows × {df_final.shape[1]} cols",
                            improvement=f"{_signed(df_final.shape[0] - df_raw.shape[0], ' rows')}, {_signed(df_final.shape[1] - df_raw.shape[1], ' cols')}"),
            ],
            cleaned_rows=int(df_final.shape[0]),
            cleaned_cols=int(df_final.shape[1]),
            quality_audit=json_safe(result.audit),
            data_dictionary=json_safe(result.data_dictionary),
        )
        get_snapshot(db, dataset_id).execution = json_safe(response.model_dump(mode="json"))
        db.commit()

    return response


# ─── Benchmark ───────────────────────────────────────────────────────────────

def _imbalance_strategy(db: Session, dataset_id: str) -> Optional[str]:
    recs = db.query(Recommendation).filter(
        Recommendation.dataset_id == dataset_id,
        Recommendation.issue_type == "class_imbalance",
        Recommendation.is_approved == True,  # noqa: E712
    ).all()
    for r in recs:
        key = (r.params or {}).get("method_key")
        if key in ("class_weights", "smote_class_weights"):
            return key
    return None


def benchmark(db: Session, db_dataset: Dataset) -> ModelBenchmarkLeaderboardResponse:
    """Cross-validates candidate models on the cleaned dataset, using the approved evaluation plan if any."""
    dataset_id = db_dataset.id
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise ServiceError(400, "Objective must be set before benchmarking.")
    if not db_dataset.cleaned_file_path or not os.path.exists(db_dataset.cleaned_file_path):
        raise ServiceError(400, "Execute the cleaning pipeline before benchmarking models.")

    pol = P.get_policy(db, dataset_id)
    plan = dict(pol.eval_plan) if pol and pol.eval_plan else None

    with dataset_lock(dataset_id):
        df = read_dataset(db_dataset.cleaned_file_path)
        assignments_path = assignments_file(dataset_id)
        remove_files(assignments_path)
        try:
            if is_unsupervised(obj.problem_type):
                lineage = {}
                if os.path.exists(data_dictionary_path(dataset_id)):
                    with open(data_dictionary_path(dataset_id), encoding="utf-8") as f:
                        lineage = {r["column"]: r.get("source_column") or r["column"] for r in json.load(f).get("columns", [])}
                results, primary_metric, summary_text, cv_folds, evaluation, assignments = run_unsupervised(df, obj.problem_type, lineage)
                assignments.to_csv(assignments_path, index=False)
                evaluation["has_assignments"] = True
            else:
                results, primary_metric, summary_text, cv_folds, evaluation = benchmark_models(
                    df_cleaned=df,
                    target_col=obj.target_column,
                    problem_type=obj.problem_type,
                    imbalance_strategy=_imbalance_strategy(db, dataset_id),
                    plan=plan,
                )
        except BenchmarkError as e:
            raise ServiceError(400, str(e))
        except Exception as e:
            logger.exception("Benchmark failed")
            raise ServiceError(500, f"Model benchmarking failed: {e}")

        db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).delete()
        items = []
        for r in results:
            bench_id = str(uuid.uuid4())
            db.add(ModelBenchmark(
                id=bench_id,
                dataset_id=dataset_id,
                model_name=r["model_name"],
                problem_type=r["problem_type"],
                metric_name=r["metric_name"],
                metric_value=r["metric_value"],
                training_time_sec=r["training_time_sec"],
                rank=r["rank"],
                is_recommended=r["is_recommended"],
                details=json_safe({
                    "description": r["description"], "suitability": r["suitability"],
                    "metric_display": r["metric_display"], "metric_std": r.get("metric_std"),
                    "is_baseline": r.get("is_baseline", False), "extra": r.get("extra"),
                }),
            ))
            items.append(ModelBenchmarkItem(id=bench_id, **{k: r[k] for k in (
                "model_name", "problem_type", "metric_name", "metric_value", "metric_display",
                "training_time_sec", "rank", "is_recommended", "description", "suitability",
            )}, metric_std=r.get("metric_std"), is_baseline=r.get("is_baseline", False), extra=json_safe(r.get("extra"))))

        response = ModelBenchmarkLeaderboardResponse(
            dataset_id=dataset_id,
            problem_type=obj.problem_type,
            primary_metric=primary_metric,
            cv_folds=cv_folds,
            models=items,
            best_model_summary=summary_text,
            evaluation=evaluation,
        )
        get_snapshot(db, dataset_id).leaderboard = json_safe(response.model_dump(mode="json"))
        db.commit()

    return response
