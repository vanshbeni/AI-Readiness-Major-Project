const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface DatasetSummary {
  id: string;
  filename: string;
  file_size_bytes: number;
  row_count: number;
  col_count: number;
  uploaded_at: string;
  has_objective: boolean;
  has_profile: boolean;
  has_health_score: boolean;
  has_cleaned_data: boolean;
}

export interface DatasetSample {
  id: string;
  filename: string;
  total_rows: number;
  total_cols: number;
  columns: string[];
  sample_data: Record<string, any>[];
}

export interface SubScores {
  missingness_score: number;
  duplicate_score: number;
  outlier_score: number;
  validity_score: number;
  target_balance_score: number;
  feature_quality_score: number;
}

export interface HealthScore {
  stage: string;
  composite_score: number;
  grade: string;
  sub_scores: SubScores;
  summary_text: string;
  top_negative_drivers: string[];
}

export interface ColumnProfile {
  name: string;
  inferred_type: string;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  unique_pct: number;
  is_target: boolean;
  numeric_stats?: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
    skewness: number;
    kurtosis: number;
    iqr: number;
  };
  categorical_stats?: {
    top_categories: { value: string; count: number; percentage: number }[];
    cardinality: number;
  };
}

export interface IssueItem {
  id: string;
  issue_type: string;
  column?: string;
  severity: string;
  title: string;
  details: Record<string, any>;
}

export interface RecommendationItem {
  id: string;
  issue_type: string;
  column?: string;
  method: string;
  reason_title: string;
  explanation_text: string;
  severity: string;
  is_destructive: boolean;
  is_approved: boolean;
}

export interface FullDiagnostic {
  dataset: DatasetSummary;
  objective?: {
    id: string;
    dataset_id: string;
    problem_type: string;
    target_column: string;
  };
  profile: {
    dataset_id: string;
    summary: {
      row_count: number;
      col_count: number;
      total_cells: number;
      total_missing_cells: number;
      overall_missing_pct: number;
      duplicate_rows_count: number;
      duplicate_rows_pct: number;
      memory_usage_mb: number;
      type_breakdown: Record<string, number>;
    };
    columns: ColumnProfile[];
  };
  health_score: HealthScore;
  issues: IssueItem[];
  recommendations: RecommendationItem[];
}

export interface MetricDelta {
  metric_name: string;
  before_value: any;
  after_value: any;
  improvement?: string;
}

export interface ExecutionResult {
  job_id: string;
  status: string;
  message: string;
  before_health_score: number;
  after_health_score: number;
  health_score_delta: number;
  transformations_applied: string[];
  metric_deltas: MetricDelta[];
  cleaned_rows: number;
  cleaned_cols: number;
}

export interface ModelBenchmarkItem {
  id: string;
  model_name: string;
  problem_type: string;
  metric_name: string;
  metric_value: number;
  metric_display: string;
  training_time_sec: number;
  rank: number;
  is_recommended: boolean;
  description: string;
  suitability: string;
}

export interface BenchmarkLeaderboard {
  dataset_id: string;
  problem_type: string;
  primary_metric: string;
  models: ModelBenchmarkItem[];
  best_model_summary: string;
}

export const api = {
  async uploadDataset(file: File): Promise<DatasetSummary> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE_URL}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async loadDemoDataset(demoKey: string): Promise<DatasetSummary> {
    const res = await fetch(`${API_BASE_URL}/datasets/demo/${demoKey}`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to load demo dataset');
    }
    return res.json();
  },

  async getSample(datasetId: string): Promise<DatasetSample> {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/sample`);
    if (!res.ok) throw new Error('Failed to fetch sample');
    return res.json();
  },

  async setObjective(datasetId: string, problemType: string, targetColumn: string) {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/objective`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ problem_type: problemType, target_column: targetColumn }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to set objective');
    }
    return res.json();
  },

  async runDiagnostics(datasetId: string): Promise<FullDiagnostic> {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/diagnose`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Diagnostic analysis failed');
    }
    return res.json();
  },

  async updateApprovals(datasetId: string, approvals: Record<string, boolean>): Promise<RecommendationItem[]> {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/recommendations/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approvals }),
    });
    if (!res.ok) throw new Error('Failed to update recommendation approvals');
    return res.json();
  },

  async executePipeline(datasetId: string): Promise<ExecutionResult> {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/execute`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Pipeline execution failed');
    }
    return res.json();
  },

  async runBenchmarks(datasetId: string): Promise<BenchmarkLeaderboard> {
    const res = await fetch(`${API_BASE_URL}/datasets/${datasetId}/benchmark`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Model benchmarking failed');
    }
    return res.json();
  },

  getCleanedCsvUrl(datasetId: string): string {
    return `${API_BASE_URL}/datasets/${datasetId}/export/cleaned-csv`;
  },

  getPipelineScriptUrl(datasetId: string): string {
    return `${API_BASE_URL}/datasets/${datasetId}/export/pipeline-script`;
  },

  getPdfReportUrl(datasetId: string): string {
    return `${API_BASE_URL}/datasets/${datasetId}/export/pdf-report`;
  },

  getManifestJsonUrl(datasetId: string): string {
    return `${API_BASE_URL}/datasets/${datasetId}/export/manifest-json`;
  },
};
