from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator


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
    modality: Optional[str] = None
    warnings: List[str] = []

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
    problem_type: str = Field(..., description="'classification' | 'regression' | 'clustering' | 'anomaly_detection'")
    target_column: Optional[str] = ""  # empty for unsupervised tasks


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
    signal_score: float = 100.0


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
    explanation_source: Optional[str] = "statistical"
    severity: str
    is_destructive: bool
    is_approved: bool
    agent_note: Optional[str] = None

    class Config:
        from_attributes = True


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
    # Post-cleaning self-audit of the exported data: status fixed | open | advice
    quality_audit: List[Dict[str, Any]] = []
    data_dictionary: List[Dict[str, Any]] = []


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
    metric_std: Optional[float] = None
    is_baseline: bool = False
    extra: Optional[Dict[str, Any]] = None  # task-specific metrics (clusters, Davies-Bouldin, flagged rows, ...)


class ModelBenchmarkLeaderboardResponse(BaseModel):
    dataset_id: str
    problem_type: str
    primary_metric: str
    cv_folds: int = 5
    models: List[ModelBenchmarkItem]
    best_model_summary: str
    # How the benchmark was run: metric, split strategy, rows used / available, and who chose it
    evaluation: Optional[Dict[str, Any]] = None


# --- Combined Diagnostics Response ---
class FullDiagnosticResponse(BaseModel):
    dataset: DatasetResponse
    objective: Optional[ObjectiveResponse] = None
    profile: ProfileReportResponse
    health_score: HealthScoreResponse
    issues: List[IssueItem]
    recommendations: List[RecommendationItem]


# --- Session Restore ---
class DatasetSessionResponse(BaseModel):
    dataset: DatasetResponse
    sample: DatasetSampleResponse
    objective: Optional[ObjectiveResponse] = None
    diagnostics: Optional[FullDiagnosticResponse] = None
    execution: Optional[ExecutionResponse] = None
    leaderboard: Optional[ModelBenchmarkLeaderboardResponse] = None


# --- Augmentation (training folds only; never written into the cleaned CSV) ---
class TextAugmentOp(BaseModel):
    name: Literal["swap", "deletion", "insertion", "sentence_shuffle"]
    alpha: float = Field(0.1, ge=0, le=1)

    model_config = {"extra": "forbid"}


class AugmentationRunRequest(BaseModel):
    modality: Literal["text", "image"]
    text_column: Optional[str] = None
    strategy: Literal["balance", "multiply"] = "balance"
    n_variants: int = Field(2, ge=0, le=10)
    ops: List[TextAugmentOp] = []
    seed: int = Field(42, ge=0, le=2_147_483_647)
    horizontal_flip: bool = True
    vertical_flip: bool = False
    max_deg: float = Field(15, ge=0, le=45)
    brightness: float = Field(0.15, ge=0, le=0.5)
    contrast: float = Field(0.15, ge=0, le=0.5)
    noise_std: float = Field(6, ge=0, le=30)
    crop_scale: float = Field(0.85, ge=0.5, le=1)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _complete(self):
        if self.modality == "text":
            if not (self.text_column or "").strip():
                raise ValueError("text_column is required for text augmentation")
            if not self.ops:
                raise ValueError("at least one text operation is required")
        else:
            enabled = any([
                self.horizontal_flip,
                self.vertical_flip,
                self.max_deg > 0,
                self.brightness > 0,
                self.contrast > 0,
                self.noise_std > 0,
                self.crop_scale < 1,
            ])
            if not enabled:
                raise ValueError("enable at least one image operation")
        return self


class AugmentModelScore(BaseModel):
    model: str
    baseline_mean: float
    baseline_std: float
    augmented_mean: float
    augmented_std: float
    delta: float
    verdict: str
    is_baseline: bool = False


class AugmentFoldAudit(BaseModel):
    fold: int
    validation_row_ids: List[int]
    synthetic_source_ids: List[int]


class AugmentationResult(BaseModel):
    modality: str
    metric: str
    seed: int
    rows_used: int
    rows_available: int
    folds: int
    synthetic_rows: int
    synthetic_rows_per_fold: List[int]
    models: List[AugmentModelScore]
    verdict: str
    verdict_detail: str
    evaluation_note: str
    leakage_audit: List[AugmentFoldAudit]
    warnings: List[str] = []


class AugmentSuggestionItem(BaseModel):
    id: str
    title: str
    detail: str
    strategy: Optional[str] = None
    text_column: Optional[str] = None


class AugmentTextColumn(BaseModel):
    name: str
    non_empty: int
    median_words: float
    unique_pct: float


class AugmentSuggestions(BaseModel):
    modality: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    row_count: int
    class_counts: Dict[str, int] = {}
    imbalance_ratio: Optional[float] = None
    text_columns: List[AugmentTextColumn] = []
    suggestions: List[AugmentSuggestionItem]
    latest_result: Optional[AugmentationResult] = None
    latest_config: Optional[Dict[str, Any]] = None


class AugmentPreviewItem(BaseModel):
    source_row_id: int
    label: str
    original: Optional[str] = None
    augmented: Optional[str] = None
    aug_ops: str = ""
    original_png: Optional[str] = None
    augmented_png: Optional[str] = None
    note: Optional[str] = None


class AugmentPreviewResponse(BaseModel):
    modality: str
    samples: List[AugmentPreviewItem]
