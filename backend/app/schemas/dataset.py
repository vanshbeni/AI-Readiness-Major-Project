from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Dataset Schemas ---
class DatasetCreate(BaseModel):
    filename: str


class DatasetResponse(BaseModel):
    id: str
    filename: str
    file_size_bytes: int
    row_count: int
    col_count: int
    uploaded_at: datetime
    has_objective: bool = False
    has_profile: bool = False
    has_health_score: bool = False
    has_cleaned_data: bool = False

    class Config:
        from_attributes = True


class DatasetSampleResponse(BaseModel):
    id: str
    filename: str
    total_rows: int
    total_cols: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]


# --- Objective Schemas ---
class ObjectiveCreate(BaseModel):
    problem_type: str = Field(..., description="'classification' or 'regression'")
    target_column: str


class ObjectiveResponse(BaseModel):
    id: str
    dataset_id: str
    problem_type: str
    target_column: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Profiling Schemas ---
class NumericStats(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    iqr: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None


class CategoricalStats(BaseModel):
    top_categories: List[Dict[str, Any]] = []
    cardinality: int = 0


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str  # numeric, categorical, datetime, boolean, id, text
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_target: bool = False
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None


class DatasetSummary(BaseModel):
    row_count: int
    col_count: int
    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    duplicate_rows_count: int
    duplicate_rows_pct: float
    memory_usage_mb: float
    type_breakdown: Dict[str, int]


class ProfileReportResponse(BaseModel):
    dataset_id: str
    summary: DatasetSummary
    columns: List[ColumnProfile]


# --- Issue Detection Schemas ---
class IssueItem(BaseModel):
    id: str
    issue_type: str
    column: Optional[str] = None
    severity: str  # critical, high, medium, low, info
    title: str
    details: Dict[str, Any]


# --- Health Score Schemas ---
class SubScores(BaseModel):
    missingness_score: float
    duplicate_score: float
    outlier_score: float
    validity_score: float
    target_balance_score: float
    feature_quality_score: float


class HealthScoreResponse(BaseModel):
    stage: str
    composite_score: float
    grade: str  # A, B, C, D, F
    sub_scores: SubScores
    summary_text: str
    top_negative_drivers: List[str]


# --- Recommendation Schemas ---
class RecommendationItem(BaseModel):
    id: str
    issue_type: str
    column: Optional[str] = None
    method: str
    reason_title: str
    explanation_text: str
    severity: str
    is_destructive: bool
    is_approved: bool


class RecommendationApprovalRequest(BaseModel):
    approvals: Dict[str, bool]  # recommendation_id -> is_approved


# --- Pipeline Execution & Diff Schemas ---
class MetricDelta(BaseModel):
    metric_name: str
    before_value: Any
    after_value: Any
    improvement: Optional[str] = None


class ExecutionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    before_health_score: float
    after_health_score: float
    health_score_delta: float
    transformations_applied: List[str]
    metric_deltas: List[MetricDelta]
    cleaned_rows: int
    cleaned_cols: int


# --- Model Benchmark Schemas ---
class ModelBenchmarkItem(BaseModel):
    id: str
    model_name: str
    problem_type: str
    metric_name: str
    metric_value: float
    metric_display: str
    training_time_sec: float
    rank: int
    is_recommended: bool
    description: str
    suitability: str


class ModelBenchmarkLeaderboardResponse(BaseModel):
    dataset_id: str
    problem_type: str
    primary_metric: str
    models: List[ModelBenchmarkItem]
    best_model_summary: str


# --- Combined Diagnostics Response ---
class FullDiagnosticResponse(BaseModel):
    dataset: DatasetResponse
    objective: Optional[ObjectiveResponse] = None
    profile: ProfileReportResponse
    health_score: HealthScoreResponse
    issues: List[IssueItem]
    recommendations: List[RecommendationItem]
