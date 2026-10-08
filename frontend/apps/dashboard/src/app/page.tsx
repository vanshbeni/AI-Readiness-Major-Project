'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Header } from '../components/Header';
import { Stepper, StepKey } from '../components/Stepper';
import { UploadStep } from '../components/UploadStep';
import { ObjectiveStep } from '../components/ObjectiveStep';
import { HealthScoreGauge } from '../components/HealthScoreGauge';
import { ProfileTable } from '../components/ProfileTable';
import { IssueReceipt } from '../components/IssueReceipt';
import { RecommendationChecklist } from '../components/RecommendationChecklist';
import { BeforeAfterDiff } from '../components/BeforeAfterDiff';
import { ModelLeaderboard } from '../components/ModelLeaderboard';
import { ExportHub } from '../components/ExportHub';
import { Banner, Eyebrow, Note, Scramble, SkyBand, Underlined } from '../components/ui';
import { BenchmarkTerminal } from '../components/Terminal';
import {
  api,
  ApiError,
  DatasetSummary,
  DatasetSample,
  FullDiagnostic,
  ExecutionResult,
  BenchmarkLeaderboard,
  Objective,
} from '../services/api';
import { ArrowRight, ArrowLeft, RefreshCw } from 'lucide-react';

const DATASET_PARAM = 'dataset';

function approvalsFrom(diag: FullDiagnostic | null): Record<string, boolean> {
  const map: Record<string, boolean> = {};
  diag?.recommendations.forEach((r) => {
    map[r.id] = r.is_approved;
  });
  return map;
}

function setDatasetInUrl(datasetId: string | null) {
  const url = new URL(window.location.href);
  if (datasetId) url.searchParams.set(DATASET_PARAM, datasetId);
  else url.searchParams.delete(DATASET_PARAM);
  window.history.replaceState(null, '', url.toString());
}

export default function Home() {
  const [currentStep, setCurrentStep] = useState<StepKey>('upload');
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [sample, setSample] = useState<DatasetSample | null>(null);
  const [objective, setObjective] = useState<Objective | null>(null);
  const [diagnostics, setDiagnostics] = useState<FullDiagnostic | null>(null);
  const [approvals, setApprovals] = useState<Record<string, boolean>>({});
  const [executedApprovals, setExecutedApprovals] = useState<Record<string, boolean> | null>(null);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [leaderboard, setLeaderboard] = useState<BenchmarkLeaderboard | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notices, setNotices] = useState<string[]>([]);

  const resetAll = useCallback(() => {
    setDataset(null);
    setSample(null);
    setObjective(null);
    setDiagnostics(null);
    setApprovals({});
    setExecutedApprovals(null);
    setExecutionResult(null);
    setLeaderboard(null);
    setBenchmarkError(null);
    setBenchmarkLoading(false);
    setNotices([]);
    setError(null);
    setCurrentStep('upload');
  }, []);

  // Restore the session from ?dataset=<id> after a refresh
  useEffect(() => {
    const datasetId = new URLSearchParams(window.location.search).get(DATASET_PARAM);
    if (!datasetId) {
      setRestoring(false);
      return;
    }
    api.getSession(datasetId)
      .then((s) => {
        setDataset(s.dataset);
        setSample(s.sample);
        setObjective(s.objective ?? null);
        const diag = s.diagnostics ?? null;
        setDiagnostics(diag);
        setApprovals(approvalsFrom(diag));
        setExecutionResult(s.execution ?? null);
        setExecutedApprovals(s.execution ? approvalsFrom(diag) : null);
        setLeaderboard(s.leaderboard ?? null);
        setCurrentStep(s.execution ? 'results' : diag ? 'diagnostic' : 'objective');
      })
      .catch((err: unknown) => {
        setDatasetInUrl(null);
        const status = err instanceof ApiError ? err.status : 0;
        setError(
          status === 404 || status === 410
            ? 'Your previous dataset session is no longer available. Please upload the dataset again.'
            : `Could not restore your previous session: ${(err as Error).message}`,
        );
      })
      .finally(() => setRestoring(false));
  }, []);

  const completedSteps = useMemo<StepKey[]>(() => {
    const steps: StepKey[] = [];
    if (dataset) steps.push('upload');
    if (diagnostics) steps.push('objective', 'diagnostic', 'recommendations');
    if (executionResult) steps.push('results');
    return steps;
  }, [dataset, diagnostics, executionResult]);

  const approvalsChangedSinceRun = useMemo(() => {
    if (!executedApprovals) return false;
    return Object.keys(approvals).some((id) => approvals[id] !== executedApprovals[id]);
  }, [approvals, executedApprovals]);

  const goTo = (step: StepKey) => {
    setError(null);
    setCurrentStep(step);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 1. Dataset loaded
  const handleDatasetLoaded = (d: DatasetSummary, s: DatasetSample) => {
    resetAll();
    setDataset(d);
    setSample(s);
    setNotices(d.warnings ?? []);
    setDatasetInUrl(d.id);
    setCurrentStep('objective');
  };

  // 2. Objective submitted -> diagnostics
  const handleObjectiveSubmit = async (problemType: string, targetColumn: string) => {
    if (!dataset) return;
    setError(null);
    const unchanged = objective && diagnostics
      && objective.problem_type === problemType && objective.target_column === targetColumn;
    if (unchanged) {
      setCurrentStep('diagnostic');
      return;
    }
    try {
      const obj = await api.setObjective(dataset.id, problemType, targetColumn);
      setObjective(obj);
      setDiagnostics(null);
      setExecutionResult(null);
      setExecutedApprovals(null);
      setLeaderboard(null);
      setBenchmarkError(null);
      const diag = await api.runDiagnostics(dataset.id);
      setDiagnostics(diag);
      setApprovals(approvalsFrom(diag));
      setCurrentStep('diagnostic');
    } catch (err: any) {
      setError(err.message || 'Diagnostic profiling failed');
      throw err;
    }
  };

  const runBenchmarks = async (datasetId: string) => {
    setBenchmarkLoading(true);
    setBenchmarkError(null);
    try {
      setLeaderboard(await api.runBenchmarks(datasetId));
    } catch (err: any) {
      setBenchmarkError(err.message || 'Model benchmarking failed');
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const handleRegenerateExplanations = async () => {
    if (!dataset) return;
    const updated = await api.regenerateExplanations(dataset.id);
    setDiagnostics((prev) => (prev ? { ...prev, recommendations: updated } : prev));
  };

  // 3. Execute approved fixes, then benchmark on the results page
  const handleExecutePipeline = async () => {
    if (!dataset || !diagnostics) return;
    setError(null);
    try {
      const updated = await api.updateApprovals(dataset.id, approvals);
      const saved: Record<string, boolean> = {};
      updated.forEach((r) => { saved[r.id] = r.is_approved; });
      setApprovals(saved);
      setDiagnostics({ ...diagnostics, recommendations: updated });

      const exec = await api.executePipeline(dataset.id);
      setExecutionResult(exec);
      setExecutedApprovals(saved);
      setLeaderboard(null);
      setCurrentStep('results');
      window.scrollTo({ top: 0 });
    } catch (err: any) {
      setError(err.message || 'Pipeline execution failed');
      throw err;
    }
    runBenchmarks(dataset.id);
  };

  const handleReset = async () => {
    if (dataset && !window.confirm('Start over with a new dataset? The current dataset and its results will be deleted from the server.')) {
      return;
    }
    const oldId = dataset?.id;
    resetAll();
    setDatasetInUrl(null);
    if (oldId) api.deleteDataset(oldId).catch(() => undefined);
  };

  if (restoring) {
    return (
      <div className="app-shell">
        <div className="grain" aria-hidden />
        <SkyBand />
        <div style={{ position: 'relative', zIndex: 2, minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.8rem' }}>
            <Eyebrow label="restore-session" caret />
            <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
              <Scramble text="re-hydrating your dataset" />
            </span>
          </div>
        </div>
      </div>
    );
  }

  const issueCount = diagnostics?.issues.length ?? 0;
  const recCount = diagnostics?.recommendations.length ?? 0;

  return (
    <div className="app-shell">
      <div className="grain" aria-hidden />
      <div className="guides" aria-hidden />
      <SkyBand />

      <Header datasetName={dataset?.filename} onReset={handleReset} />

      {dataset && <Stepper currentStep={currentStep} onStepClick={goTo} completedSteps={completedSteps} />}

      <main className="app-main">
        <div className="frame">
          {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}
          {notices.length > 0 && (
            <Banner tone="info" onClose={() => setNotices([])}>
              {notices.map((n, i) => <div key={i}>{n}</div>)}
            </Banner>
          )}

          {currentStep === 'upload' && <UploadStep onDatasetLoaded={handleDatasetLoaded} />}

          {currentStep === 'objective' && dataset && sample && (
            <ObjectiveStep dataset={dataset} sample={sample} initialObjective={objective} onSubmit={handleObjectiveSubmit} />
          )}

          {currentStep === 'diagnostic' && diagnostics && (
            <div key="diagnostic" className="step-enter section-gap">
              <div className="step-head">
                <Eyebrow label="health-score" index="03/05" />
                <div className="step-head-row">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
                    <h1 className="h-title">
                      <Scramble text="Here's what's" /> <Underlined>wrong.</Underlined>
                    </h1>
                    <p className="h-sub">
                      Predicting <strong>{diagnostics.objective?.target_column}</strong> ({diagnostics.objective?.problem_type})
                      {' '}across {diagnostics.profile.summary.row_count.toLocaleString()} rows × {diagnostics.profile.summary.col_count} columns.
                      {' '}{issueCount} issue{issueCount === 1 ? '' : 's'} found.
                    </p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    {recCount > 0 && <Note tone="purple" arrow="down" rot={-6}>you approve each one</Note>}
                    <button className="btn btn-dark btn-lg" onClick={() => goTo('recommendations')}>
                      {recCount > 0 ? `Review ${recCount} fix${recCount === 1 ? '' : 'es'}` : 'No fixes needed, continue'}
                      <ArrowRight size={16} className="arrow" />
                    </button>
                  </div>
                </div>
              </div>

              <HealthScoreGauge score={diagnostics.health_score} stageName="raw dataset" />
              {issueCount > 0 && <IssueReceipt issues={diagnostics.issues} />}
              <ProfileTable columns={diagnostics.profile.columns} />
            </div>
          )}

          {currentStep === 'recommendations' && diagnostics && (
            <div key="recommendations" className="step-enter section-gap">
              <div>
                <button className="btn btn-ghost btn-sm" onClick={() => goTo('diagnostic')}>
                  <ArrowLeft size={14} className="arrow arrow-left" /> back to health score
                </button>
              </div>
              {approvalsChangedSinceRun && (
                <Banner tone="info">You changed fix approvals since the last run. Apply the fixes again to update the cleaned dataset and model results.</Banner>
              )}
              <RecommendationChecklist
                recommendations={diagnostics.recommendations}
                approvals={approvals}
                onToggle={(id) => setApprovals((prev) => ({ ...prev, [id]: !prev[id] }))}
                onSetAll={(state) => setApprovals(Object.fromEntries(diagnostics.recommendations.map((r) => [r.id, state])))}
                onExecute={handleExecutePipeline}
                onRegenerateExplanations={handleRegenerateExplanations}
              />
            </div>
          )}

          {currentStep === 'results' && dataset && (
            <div key="results" className="step-enter section-gap">
              <div className="step-head">
                <Eyebrow label="clean-and-benchmark" index="05/05" />
                <div className="step-head-row">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
                    <h1 className="h-title">
                      <Scramble text="Model-ready." /> <Underlined>Proven.</Underlined>
                    </h1>
                    <p className="h-sub">
                      The before/after quality change, a cross-validated model leaderboard, and everything you need to take it home.
                    </p>
                  </div>
                  <button className="btn btn-light" onClick={() => goTo('recommendations')}>
                    <ArrowLeft size={14} className="arrow arrow-left" /> Modify fix approvals
                  </button>
                </div>
              </div>

              {approvalsChangedSinceRun && (
                <Banner tone="info">These results reflect the previous approvals. Go back and apply the fixes again to include your latest changes.</Banner>
              )}

              {executionResult ? <BeforeAfterDiff result={executionResult} /> : (
                <Banner tone="info">The cleaning pipeline has not been executed yet.</Banner>
              )}

              {benchmarkLoading && <BenchmarkTerminal problemType={diagnostics?.objective?.problem_type} />}
              {benchmarkError && !benchmarkLoading && (
                <Banner tone="error">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                    <span>Model benchmarking failed: {benchmarkError}</span>
                    <button className="btn btn-light btn-sm" onClick={() => runBenchmarks(dataset.id)}>
                      <RefreshCw size={13} /> Retry
                    </button>
                  </div>
                </Banner>
              )}
              {!leaderboard && !benchmarkLoading && !benchmarkError && executionResult && (
                <div>
                  <button className="btn btn-dark" onClick={() => runBenchmarks(dataset.id)}>
                    <RefreshCw size={14} /> Run model benchmarks
                  </button>
                </div>
              )}
              {leaderboard && !benchmarkLoading && <ModelLeaderboard leaderboard={leaderboard} />}

              <ExportHub datasetId={dataset.id} hasExecution={!!executionResult} hasBenchmarks={!!leaderboard} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
