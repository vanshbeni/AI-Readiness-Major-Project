'use client';

import React, { useState } from 'react';
import { Header } from '../components/Header';
import { Stepper, StepKey } from '../components/Stepper';
import { UploadStep } from '../components/UploadStep';
import { ObjectiveStep } from '../components/ObjectiveStep';
import { HealthScoreGauge } from '../components/HealthScoreGauge';
import { ProfileTable } from '../components/ProfileTable';
import { RecommendationChecklist } from '../components/RecommendationChecklist';
import { BeforeAfterDiff } from '../components/BeforeAfterDiff';
import { ModelLeaderboard } from '../components/ModelLeaderboard';
import { ExportHub } from '../components/ExportHub';
import {
  api,
  DatasetSummary,
  DatasetSample,
  FullDiagnostic,
  ExecutionResult,
  BenchmarkLeaderboard,
} from '../services/api';
import { ArrowRight, ArrowLeft, AlertTriangle } from 'lucide-react';

export default function Home() {
  const [currentStep, setCurrentStep] = useState<StepKey>('upload');
  const [completedSteps, setCompletedSteps] = useState<StepKey[]>([]);

  // State entities
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [sample, setSample] = useState<DatasetSample | null>(null);
  const [diagnostics, setDiagnostics] = useState<FullDiagnostic | null>(null);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [leaderboard, setLeaderboard] = useState<BenchmarkLeaderboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Dataset Loaded handler
  const handleDatasetLoaded = (d: DatasetSummary, s: DatasetSample) => {
    setDataset(d);
    setSample(s);
    setCompletedSteps(['upload']);
    setCurrentStep('objective');
  };

  // 2. Objective Selected handler
  const handleObjectiveSubmit = async (problemType: string, targetColumn: string) => {
    if (!dataset) return;
    setError(null);
    try {
      await api.setObjective(dataset.id, problemType, targetColumn);
      const diag = await api.runDiagnostics(dataset.id);
      setDiagnostics(diag);
      setCompletedSteps((prev) => Array.from(new Set([...prev, 'objective'])));
      setCurrentStep('diagnostic');
    } catch (err: any) {
      setError(err.message || 'Diagnostic profiling failed');
    }
  };

  // 3. Execution Pipeline handler
  const handleExecutePipeline = async (approvals: Record<string, boolean>) => {
    if (!dataset) return;
    setError(null);
    try {
      await api.updateApprovals(dataset.id, approvals);
      const exec = await api.executePipeline(dataset.id);
      setExecutionResult(exec);
      
      // Auto-trigger model benchmarking
      const bench = await api.runBenchmarks(dataset.id);
      setLeaderboard(bench);

      setCompletedSteps((prev) => Array.from(new Set([...prev, 'recommendations', 'results'])));
      setCurrentStep('results');
    } catch (err: any) {
      setError(err.message || 'Pipeline execution or benchmarking failed');
    }
  };

  // Reset flow
  const handleReset = () => {
    setDataset(null);
    setSample(null);
    setDiagnostics(null);
    setExecutionResult(null);
    setLeaderboard(null);
    setCompletedSteps([]);
    setCurrentStep('upload');
    setError(null);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header datasetName={dataset?.filename} onReset={handleReset} />

      {dataset && (
        <Stepper
          currentStep={currentStep}
          onStepClick={(step) => setCurrentStep(step)}
          completedSteps={completedSteps}
        />
      )}

      <main style={{ flex: 1, padding: '1.5rem 2rem 4rem 2rem', maxWidth: 1200, width: '100%', margin: '0 auto' }}>
        {error && (
          <div style={{
            padding: '1rem 1.25rem',
            borderRadius: 10,
            background: 'rgba(244, 63, 94, 0.12)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            color: '#fb7185',
            fontSize: '0.9rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
          }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* STEP 1: Upload & Ingestion */}
        {currentStep === 'upload' && (
          <UploadStep onDatasetLoaded={handleDatasetLoaded} />
        )}

        {/* STEP 2: Objective & Target Column */}
        {currentStep === 'objective' && dataset && sample && (
          <ObjectiveStep
            dataset={dataset}
            sample={sample}
            onSubmit={handleObjectiveSubmit}
          />
        )}

        {/* STEP 3: Diagnostic Profiling & Health Score Dashboard */}
        {currentStep === 'diagnostic' && diagnostics && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.85rem', fontWeight: 800 }}>
                  Pre-ML <span className="gradient-text">Diagnostic Assessment</span>
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Target: <strong>{diagnostics.objective?.target_column}</strong> ({diagnostics.objective?.problem_type}) • {diagnostics.profile.summary.row_count} rows × {diagnostics.profile.summary.col_count} columns
                </p>
              </div>

              <button
                className="btn-primary"
                onClick={() => {
                  setCompletedSteps((prev) => Array.from(new Set([...prev, 'diagnostic'])));
                  setCurrentStep('recommendations');
                }}
              >
                <span>Review {diagnostics.recommendations.length} Explainable Fixes</span>
                <ArrowRight size={16} />
              </button>
            </div>

            {/* Health Score Dial */}
            <HealthScoreGauge score={diagnostics.health_score} stageName="Raw Dataset Profile" />

            {/* Feature Matrix Table */}
            <ProfileTable columns={diagnostics.profile.columns} />
          </div>
        )}

        {/* STEP 4: Explainable Recommendations Checklist */}
        {currentStep === 'recommendations' && diagnostics && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <button
                className="btn-secondary"
                onClick={() => setCurrentStep('diagnostic')}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                <ArrowLeft size={14} /> Back to Health Score
              </button>
            </div>

            <RecommendationChecklist
              recommendations={diagnostics.recommendations}
              onExecute={handleExecutePipeline}
            />
          </div>
        )}

        {/* STEP 5: Cleaned Results, Benchmarks & Export Hub */}
        {currentStep === 'results' && dataset && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.85rem', fontWeight: 800 }}>
                  Model-Ready <span className="gradient-text">Dataset & Candidate Models</span>
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Transformations executed cleanly. Review before/after improvements, model benchmark leaderboard, and download artifacts.
                </p>
              </div>

              <button
                className="btn-secondary"
                onClick={() => setCurrentStep('recommendations')}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                <ArrowLeft size={14} /> Modify Fix Approvals
              </button>
            </div>

            {/* Before vs After Quality Diff */}
            {executionResult && <BeforeAfterDiff result={executionResult} />}

            {/* Candidate ML Model Leaderboard */}
            {leaderboard && <ModelLeaderboard leaderboard={leaderboard} />}

            {/* Export & Download Hub */}
            <ExportHub datasetId={dataset.id} />
          </div>
        )}
      </main>
    </div>
  );
}
