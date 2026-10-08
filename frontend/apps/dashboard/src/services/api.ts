const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
const CLIENT_ID_KEY = 'drp_client_id';

export const MAX_UPLOAD_MB = 200;
export const SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];

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
  warnings?: string[];
}

export interface DatasetSample {
  id: string;
  filename: string;
  total_rows: number;
  total_cols: number;
  columns: string[];
  sample_data: Record<string, any>[];
}

export interface Objective {
  id: string;
  dataset_id: string;
  problem_type: string;
  target_column: string;
}

export interface TargetSuggestion {
  suitable: boolean;
  problem_type: 'classification' | 'regression' | null;
  reason: string;
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
  } | null;
  categorical_stats?: {
    top_categories: { value: string; count: number; percentage: number }[];
    cardinality: number;
  } | null;
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
  explanation_source?: 'ai' | 'statistical' | null;
  severity: string;
  is_destructive: boolean;
  is_approved: boolean;
}

export interface FullDiagnostic {
  dataset: DatasetSummary;
  objective?: Objective | null;
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
  cv_folds: number;
  models: ModelBenchmarkItem[];
  best_model_summary: string;
}

export interface DatasetSession {
  dataset: DatasetSummary;
  sample: DatasetSample;
  objective?: Objective | null;
  diagnostics?: FullDiagnostic | null;
  execution?: ExecutionResult | null;
  leaderboard?: BenchmarkLeaderboard | null;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function generateId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export function getClientId(): string {
  if (typeof window === 'undefined') return '';
  let id = window.localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = generateId();
    window.localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  const text = await res.text().catch(() => '');
  if (text) {
    try {
      const body = JSON.parse(text);
      const detail = body?.detail;
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d: any) => (d?.loc ? `${d.loc.filter((p: any) => p !== 'body').join('.')}: ${d.msg}` : d?.msg || String(d)))
          .join('; ');
      }
    } catch {
      // non-JSON body (e.g. proxy or server error page)
    }
  }
  return `${fallback} (HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''})`;
}

async function request(path: string, init: RequestInit, fallbackError: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'X-Client-Id': getClientId(), ...(init.headers || {}) },
    });
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}. Make sure the API server is running.`, 0);
  }
  if (!res.ok) throw new ApiError(await extractErrorMessage(res, fallbackError), res.status);
  return res;
}

async function requestJson<T>(path: string, init: RequestInit, fallbackError: string): Promise<T> {
  const res = await request(path, init, fallbackError);
  return res.json() as Promise<T>;
}

function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header);
  if (star?.[1]) return decodeURIComponent(star[1].replace(/"/g, ''));
  const plain = /filename="?([^";]+)"?/i.exec(header);
  return plain?.[1] ?? null;
}

const jsonHeaders = { 'Content-Type': 'application/json' };

export const api = {
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
      return res.ok;
    } catch {
      return false;
    }
  },

  uploadDataset(file: File): Promise<DatasetSummary> {
    const formData = new FormData();
    formData.append('file', file);
    return requestJson('/datasets/upload', { method: 'POST', body: formData }, 'Upload failed');
  },

  loadDemoDataset(demoKey: string): Promise<DatasetSummary> {
    return requestJson(`/datasets/demo/${demoKey}`, { method: 'POST' }, 'Failed to load demo dataset');
  },

  getSample(datasetId: string): Promise<DatasetSample> {
    return requestJson(`/datasets/${datasetId}/sample`, {}, 'Failed to fetch sample');
  },

  getSession(datasetId: string): Promise<DatasetSession> {
    return requestJson(`/datasets/${datasetId}/session`, { cache: 'no-store' }, 'Failed to restore session');
  },

  async deleteDataset(datasetId: string): Promise<void> {
    await request(`/datasets/${datasetId}`, { method: 'DELETE' }, 'Failed to delete dataset');
  },

  getTargetSuggestions(datasetId: string): Promise<Record<string, TargetSuggestion>> {
    return requestJson(`/datasets/${datasetId}/target-suggestions`, {}, 'Failed to analyse target columns');
  },

  setObjective(datasetId: string, problemType: string, targetColumn: string): Promise<Objective> {
    return requestJson(
      `/datasets/${datasetId}/objective`,
      { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ problem_type: problemType, target_column: targetColumn }) },
      'Failed to set objective',
    );
  },

  runDiagnostics(datasetId: string): Promise<FullDiagnostic> {
    return requestJson(`/datasets/${datasetId}/diagnose`, { method: 'POST' }, 'Diagnostic analysis failed');
  },

  updateApprovals(datasetId: string, approvals: Record<string, boolean>): Promise<RecommendationItem[]> {
    return requestJson(
      `/datasets/${datasetId}/recommendations/approve`,
      { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ approvals }) },
      'Failed to update recommendation approvals',
    );
  },

  regenerateExplanations(datasetId: string): Promise<RecommendationItem[]> {
    return requestJson(
      `/datasets/${datasetId}/recommendations/explain`,
      { method: 'POST' },
      'Failed to regenerate AI explanations',
    );
  },

  executePipeline(datasetId: string): Promise<ExecutionResult> {
    return requestJson(`/datasets/${datasetId}/execute`, { method: 'POST' }, 'Pipeline execution failed');
  },

  runBenchmarks(datasetId: string): Promise<BenchmarkLeaderboard> {
    return requestJson(`/datasets/${datasetId}/benchmark`, { method: 'POST' }, 'Model benchmarking failed');
  },

  async downloadExport(
    datasetId: string,
    kind: 'cleaned-csv' | 'pipeline-script' | 'pdf-report' | 'manifest-json',
    fallbackName: string,
  ): Promise<void> {
    const res = await request(`/datasets/${datasetId}/export/${kind}`, {}, 'Download failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filenameFromDisposition(res.headers.get('Content-Disposition')) || fallbackName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};
