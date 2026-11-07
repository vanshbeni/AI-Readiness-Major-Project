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

export type ProblemType = 'classification' | 'regression' | 'clustering' | 'anomaly_detection';
export const UNSUPERVISED: ProblemType[] = ['clustering', 'anomaly_detection'];
export const isUnsupervised = (p?: string | null) => !!p && (UNSUPERVISED as string[]).includes(p);
export const TASK_LABEL: Record<string, string> = {
  classification: 'classification',
  regression: 'regression',
  clustering: 'clustering',
  anomaly_detection: 'anomaly detection',
};

export interface Objective {
  id: string;
  dataset_id: string;
  problem_type: string;
  target_column: string; // empty for unsupervised tasks
}

export interface TargetSuggestion {
  suitable: boolean;
  problem_type: ProblemType | null;
  reason: string;
  warning?: string;
}

export interface SubScores {
  missingness_score: number;
  duplicate_score: number;
  outlier_score: number;
  validity_score: number;
  target_balance_score: number;
  feature_quality_score: number;
  signal_score?: number;
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
  agent_note?: string | null;
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
  quality_audit?: QualityFinding[];
  data_dictionary?: DataDictionaryRow[];
}

export interface QualityFinding {
  check: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  status: 'fixed' | 'open' | 'advice';
  column?: string | null;
  detail: string;
}

export interface DataDictionaryRow {
  column: string;
  dtype: string;
  kind: string;
  source_column: string;
  origin: 'original' | 'derived';
  meaning: string;
  transformations: string[];
  missing: number;
  unique: number;
  min?: number | null;
  max?: number | null;
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
  metric_std?: number | null;
  is_baseline?: boolean;
  extra?: Record<string, any> | null;
}

export interface ClusterProfile {
  cluster: number;
  size: number;
  share: number;
  noise: boolean;
  traits: { feature: string; z: number; cluster: number; overall: number; unit: string }[];
}

export interface AnomalyRow {
  row: number | string;
  score: number;
  votes: number;
  of: number;
  reasons: { feature: string; value: number; robust_z: number }[];
}

export interface BenchmarkEvaluation {
  metric: string;
  metric_label: string;
  higher_is_better: boolean;
  cv: 'stratified' | 'kfold' | 'time' | 'none';
  n_splits: number;
  time_column?: string | null;
  rows_used: number;
  rows_available: number;
  skipped_models: string[];
  source: string;
  reason?: string | null;
  verdict?: 'useful' | 'weak' | 'no_signal' | null;
  lift_over_baseline?: number | null;
  task?: string;
  clusters?: ClusterProfile[];
  k?: number;
  anomalies?: AnomalyRow[];
  contamination?: number;
  consensus_count?: number;
  detectors?: string[];
  has_assignments?: boolean;
}

export interface BenchmarkLeaderboard {
  dataset_id: string;
  problem_type: string;
  primary_metric: string;
  cv_folds: number;
  models: ModelBenchmarkItem[];
  best_model_summary: string;
  evaluation?: BenchmarkEvaluation | null;
}

// ─── Summary page ────────────────────────────────────────────────────────────

export interface SummaryHist { discrete: boolean; edges?: number[]; values?: number[]; counts: number[] }
export interface SummaryBox { q1: number; median: number; q3: number; lo: number; hi: number; min: number; max: number; outliers: number; extreme: number }
export interface SummaryColumn {
  name: string;
  kind: 'numeric' | 'categorical' | 'datetime' | 'text' | 'id';
  inferred_type: string;
  is_target: boolean;
  missing_pct: number;
  unique: number;
  stats?: { min: number; max: number; mean: number; median: number; std: number; q1: number; q3: number; skew: number } | null;
  range?: [string, string] | null;
  hist?: SummaryHist | null;
  box?: SummaryBox | null;
  timeline?: { granularity: string; points: { t: string; count: number }[]; start: string; end: string; span_days: number } | null;
  top?: { value: string; count: number }[];
  other_count?: number;
  top_share?: number;
}
export interface SummaryInsight { kind: string; tone: 'good' | 'warn' | 'bad' | 'info'; title: string; text: string; column?: string }
export interface SummaryDriver {
  feature: string;
  association: number;
  breakdown?: { measure: string; overall: number; focus?: string; groups: { group: string; value: number; n: number }[] } | null;
}
export interface SummaryPayload {
  dataset_id: string;
  filename: string;
  source: 'raw' | 'cleaned';
  has_cleaned: boolean;
  rows: number;
  cols: number;
  kinds: Record<string, number>;
  target: string | null;
  problem_type: string | null;
  columns: SummaryColumn[];
  missing: { cells_pct: number; complete_rows_pct: number; by_column: { column: string; pct: number; count: number }[] };
  duplicates: number;
  correlation: { pairs: { a: string; b: string; r: number }[]; heatmap: { columns: string[]; matrix: (number | null)[][] } | null };
  target_info: {
    name: string; problem_type: string; missing: number;
    classes?: { value: string; count: number }[]; hist?: SummaryHist | null; box?: SummaryBox | null; drivers: SummaryDriver[];
  } | null;
  projection: {
    coords: number[][]; explained: number[]; loadings: { feature: string; weight: number }[][]; features: string[];
    color: (string | null)[] | null; color_label: string | null;
  } | null;
  insights: SummaryInsight[];
  health: HealthScore | null;
  issues: { severity: string; title: string; type: string }[];
  execution: { before_health_score: number; after_health_score: number; health_score_delta: number; cleaned_rows: number;
    cleaned_cols: number; transformations_applied: string[]; quality_audit?: QualityFinding[] } | null;
  leaderboard: BenchmarkLeaderboard | null;
  report: { markdown: string; source: string; generated_at: string } | null;
}

// ─── Agents ──────────────────────────────────────────────────────────────────

export type AgentKind = 'understand' | 'clean' | 'evaluate' | 'report' | 'ask' | 'autopilot';
export type AgentRunStatus = 'queued' | 'running' | 'waiting' | 'completed' | 'failed' | 'cancelled';

export interface AgentStep {
  i: number;
  t: string;
  actor: string;
  kind: 'plan' | 'tool' | 'llm' | 'critic' | 'gate' | 'result' | 'warn' | 'error';
  text: string;
  data?: any;
}

export interface AgentGateItem {
  id: string;
  label: string;
  reason?: string;
  confidence?: number;
  source?: string;
  destructive?: boolean;
  default?: boolean;
  column?: string | null;
  action?: string;
  from?: string;
  to?: string;
}

export interface AgentGateQuestion {
  id: string;
  column: string;
  question: string;
  options: { label: string; action?: string | null }[];
}

export interface AgentGate {
  type: 'understand' | 'clean' | 'evaluate';
  title: string;
  stage?: string;
  items: AgentGateItem[];
  questions?: AgentGateQuestion[];
  notes?: Record<string, string>;
  plan?: Record<string, any>;
}

export interface AgentChart {
  id: string;
  type: 'bar' | 'line' | 'scatter';
  title: string;
  table: string;
  x: string;
  y: string[];
  columns: string[];
  rows: (string | number | null)[][];
}

export interface AgentTable {
  id: string;
  title: string;
  columns: string[];
  rows: (string | number | null)[][];
}

export interface AgentReport {
  markdown: string;
  source: 'agent' | 'rules';
  generated_at: string;
  facts?: { id: string; label: string; value: any }[];
}

export interface AgentRun {
  id: string;
  dataset_id: string;
  kind: AgentKind;
  status: AgentRunStatus;
  input?: Record<string, any> | null;
  steps: AgentStep[];
  step_count: number;
  gate: AgentGate | null;
  result: any;
  usage?: { calls: number; prompt_tokens: number; completion_tokens: number; models: string[] } | null;
  error?: string | null;
  created_at?: string;
}

export interface AgentStatus {
  configured: boolean;
  provider: string;
  model: string;
  fallbacks: string[];
  privacy_mode: string;
  agents: AgentKind[];
}

export interface AgentPolicy {
  column_overrides: Record<string, Record<string, any>>;
  eval_plan: Record<string, any> | null;
  report: AgentReport | null;
}

export type ExploreKind = 'numeric' | 'categorical' | 'datetime' | 'text' | 'id';

export interface ExploreColumn {
  name: string;
  kind: ExploreKind;
  inferred_type: string;
  is_target: boolean;
  missing_pct: number;
  unique: number;
  included: boolean;
  numeric_like: boolean;
  stats?: { min: number; max: number; mean: number; median: number; std: number; q1: number; q3: number; skew: number } | null;
  top?: { value: string; count: number }[];
  range?: [string, string];
}

export interface ExploreProjection {
  features: string[];
  dims: number;
  explained: number[];
  loadings: { feature: string; weight: number }[][];
  coords: number[][];
  clusters: string[] | null;
  k: number | null;
  silhouette: number | null;
}

export interface ExplorePayload {
  source: 'raw' | 'cleaned';
  filename: string;
  has_cleaned: boolean;
  total_rows: number;
  sample_rows: number;
  total_cols: number;
  target: string | null;
  columns: ExploreColumn[];
  data: Record<string, (number | string | null)[]>;
  correlation: { columns: string[]; pearson: (number | null)[][]; spearman: (number | null)[][] } | null;
  projection: ExploreProjection | null;
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

  summary(datasetId: string, source: 'raw' | 'cleaned'): Promise<SummaryPayload> {
    return requestJson(`/datasets/${datasetId}/summary?source=${source}`, { cache: 'no-store' }, 'Failed to build the summary');
  },

  explore(datasetId: string, source: 'raw' | 'cleaned', maxRows = 2000): Promise<ExplorePayload> {
    return requestJson(
      `/datasets/${datasetId}/explore?source=${source}&max_rows=${maxRows}`,
      { cache: 'no-store' },
      'Failed to load the dataset for visualization',
    );
  },

  agentStatus(): Promise<AgentStatus> {
    return requestJson('/agents/status', { cache: 'no-store' }, 'Failed to read agent status');
  },

  startAgent(
    datasetId: string,
    kind: AgentKind,
    body: { question?: string; history?: { q: string; a: string }[]; source?: 'raw' | 'cleaned' } = {},
  ): Promise<AgentRun> {
    return requestJson(
      `/datasets/${datasetId}/agents/${kind}`,
      { method: 'POST', headers: jsonHeaders, body: JSON.stringify(body) },
      'Failed to start the agent',
    );
  },

  getAgentRun(runId: string, since = 0): Promise<AgentRun> {
    return requestJson(`/agent-runs/${runId}?since=${since}`, { cache: 'no-store' }, 'Failed to read the agent run');
  },

  resumeAgentRun(runId: string, approvals: Record<string, boolean>, answers: Record<string, number>): Promise<AgentRun> {
    return requestJson(
      `/agent-runs/${runId}/resume`,
      { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ approvals, answers }) },
      'Failed to send your decision to the agent',
    );
  },

  cancelAgentRun(runId: string): Promise<AgentRun> {
    return requestJson(`/agent-runs/${runId}/cancel`, { method: 'POST' }, 'Failed to cancel the agent run');
  },

  listAgentRuns(datasetId: string, kind?: AgentKind, limit = 10): Promise<AgentRun[]> {
    const q = new URLSearchParams({ limit: String(limit), ...(kind ? { kind } : {}) });
    return requestJson(`/datasets/${datasetId}/agents/runs?${q}`, { cache: 'no-store' }, 'Failed to list agent runs');
  },

  getAgentPolicy(datasetId: string): Promise<AgentPolicy> {
    return requestJson(`/datasets/${datasetId}/agents/policy`, { cache: 'no-store' }, 'Failed to read agent decisions');
  },

  resetAgentPolicy(datasetId: string, part: 'all' | 'columns' | 'eval_plan' | 'report' = 'all'): Promise<AgentPolicy> {
    return requestJson(`/datasets/${datasetId}/agents/policy?part=${part}`, { method: 'DELETE' }, 'Failed to reset agent decisions');
  },

  runBenchmarks(datasetId: string): Promise<BenchmarkLeaderboard> {
    return requestJson(`/datasets/${datasetId}/benchmark`, { method: 'POST' }, 'Model benchmarking failed');
  },

  async downloadExport(
    datasetId: string,
    kind: 'cleaned-csv' | 'pipeline-script' | 'data-dictionary' | 'assignments' | 'pdf-report' | 'manifest-json',
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
