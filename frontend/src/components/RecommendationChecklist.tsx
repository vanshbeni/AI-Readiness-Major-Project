import React, { useState, useEffect } from 'react';
import { RecommendationItem } from '../services/api';
import { Sparkles, CheckSquare, XCircle, AlertTriangle, ArrowRight, Play, Loader2, Info, CheckCircle2, ShieldCheck, Cpu, FileCheck } from 'lucide-react';

interface RecommendationChecklistProps {
  recommendations: RecommendationItem[];
  onExecute: (approvals: Record<string, boolean>) => Promise<void>;
}

export const RecommendationChecklist: React.FC<RecommendationChecklistProps> = ({
  recommendations,
  onExecute,
}) => {
  const [approvals, setApprovals] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    recommendations.forEach((r) => {
      initial[r.id] = r.is_approved;
    });
    return initial;
  });
  const [executing, setExecuting] = useState(false);
  const [execStep, setExecStep] = useState(0);

  const toggleApproval = (id: string) => {
    setApprovals((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleApproveAll = (state: boolean) => {
    const next: Record<string, boolean> = {};
    recommendations.forEach((r) => {
      next[r.id] = state;
    });
    setApprovals(next);
  };

  const approvedCount = Object.values(approvals).filter(Boolean).length;

  const handleExecute = async () => {
    setExecuting(true);
    setExecStep(1);

    // Simulated visual step progression while async backend executes
    const timer1 = setTimeout(() => setExecStep(2), 600);
    const timer2 = setTimeout(() => setExecStep(3), 1400);
    const timer3 = setTimeout(() => setExecStep(4), 2200);

    try {
      await onExecute(approvals);
      setExecStep(5);
    } catch (e) {
      setExecuting(false);
      throw e;
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    }
  };

  const EXECUTION_STEPS = [
    { num: 1, label: 'Submitting approved remediation rules & parameters', icon: ShieldCheck },
    { num: 2, label: 'Executing safe-order pipeline (Impute, Clean, Winsorize, Encode, Scale)', icon: Loader2 },
    { num: 3, label: 'Re-profiling cleaned dataset & computing Post-Cleaning Health Score', icon: CheckCircle2 },
    { num: 4, label: 'Cross-validating candidate ML models (XGBoost, Random Forest, LightGBM)', icon: Cpu },
    { num: 5, label: 'Compiling quality improvement diffs & generating download artifacts', icon: FileCheck },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', position: 'relative' }}>
      {/* Execution Progress Modal Overlay */}
      {executing && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(7, 9, 14, 0.85)',
          backdropFilter: 'blur(16px)',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem',
        }}>
          <div className="glass-panel glow-cyan" style={{
            maxWidth: 550,
            width: '100%',
            padding: '2.5rem 2rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
            background: 'rgba(14, 20, 31, 0.95)',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.8)',
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: 56,
                height: 56,
                borderRadius: '50%',
                background: 'rgba(56, 189, 248, 0.15)',
                margin: '0 auto 1rem auto',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#38bdf8',
              }}>
                <Loader2 size={30} className="animate-spin" />
              </div>
              <h3 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: '0.35rem' }}>
                Executing Data Pipeline & ML Benchmarking
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Transforming {approvedCount} approved rules into a clean, model-ready dataset...
              </p>
            </div>

            {/* Step by step list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
              {EXECUTION_STEPS.map((s) => {
                const Icon = s.icon;
                const isCurrent = execStep === s.num;
                const isDone = execStep > s.num;

                return (
                  <div
                    key={s.num}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      padding: '0.65rem 0.9rem',
                      borderRadius: 8,
                      background: isCurrent ? 'rgba(56, 189, 248, 0.1)' : isDone ? 'rgba(16, 185, 129, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                      border: isCurrent ? '1px solid #38bdf8' : isDone ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid var(--border)',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    <div style={{
                      width: 22,
                      height: 22,
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: isDone ? '#10b981' : isCurrent ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)',
                      color: isDone || isCurrent ? '#07090e' : 'var(--text-dim)',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      flexShrink: 0,
                    }}>
                      {isDone ? <CheckCircle2 size={14} color="#07090e" /> : isCurrent ? <Loader2 size={14} className="animate-spin" /> : s.num}
                    </div>
                    <span style={{
                      fontSize: '0.825rem',
                      fontWeight: isCurrent ? 600 : 400,
                      color: isDone ? '#34d399' : isCurrent ? '#38bdf8' : 'var(--text-dim)',
                    }}>
                      {s.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Progress bar */}
            <div style={{ width: '100%', height: 6, background: 'rgba(255, 255, 255, 0.08)', borderRadius: 4, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${(execStep / 5) * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #38bdf8 0%, #10b981 100%)',
                  borderRadius: 4,
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Header controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={20} color="#38bdf8" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800 }}>
              Explainable Remediation Recommendations
            </h3>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Each fix is chosen by statistical heuristics and justified with Gemini. Only approved fixes will be executed.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <button
            onClick={() => handleApproveAll(true)}
            className="btn-secondary"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.775rem' }}
          >
            Approve All
          </button>
          <button
            onClick={() => handleApproveAll(false)}
            className="btn-secondary"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.775rem' }}
          >
            Reject All
          </button>
        </div>
      </div>

      {/* Recommendations Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {recommendations.map((rec) => {
          const isApproved = approvals[rec.id] ?? rec.is_approved;
          const severityClass =
            rec.severity === 'critical'
              ? 'badge-critical'
              : rec.severity === 'high'
              ? 'badge-high'
              : 'badge-medium';

          return (
            <div
              key={rec.id}
              className="glass-panel"
              style={{
                padding: '1.25rem 1.5rem',
                borderLeft: isApproved ? '4px solid #10b981' : '4px solid #475569',
                background: isApproved ? 'rgba(14, 20, 31, 0.85)' : 'rgba(14, 20, 31, 0.4)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                  <span className={`badge ${severityClass}`}>
                    {rec.severity}
                  </span>
                  {rec.column && (
                    <span className="badge badge-low" style={{ textTransform: 'none', fontWeight: 600 }}>
                      Feature: {rec.column}
                    </span>
                  )}
                  {rec.is_destructive && (
                    <span className="badge badge-critical" style={{ fontSize: '0.65rem' }}>
                      Destructive Action
                    </span>
                  )}
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, color: isApproved ? 'var(--text-main)' : 'var(--text-dim)' }}>
                    {rec.reason_title}
                  </h4>
                </div>

                {/* Toggle switch */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexShrink: 0 }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isApproved ? '#34d399' : 'var(--text-dim)' }}>
                    {isApproved ? 'Approved' : 'Skipped'}
                  </span>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={isApproved}
                      onChange={() => toggleApproval(rec.id)}
                    />
                    <span className="slider" />
                  </label>
                </div>
              </div>

              {/* Method badge */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Remediation Method:</span>
                <span style={{
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  color: '#38bdf8',
                  padding: '0.15rem 0.5rem',
                  borderRadius: 6,
                  background: 'rgba(56, 189, 248, 0.1)',
                  border: '1px solid rgba(56, 189, 248, 0.2)',
                }}>
                  {rec.method}
                </span>
              </div>

              {/* Gemini Reasoning Callout */}
              <div style={{
                padding: '0.75rem 1rem',
                borderRadius: 8,
                background: 'rgba(56, 189, 248, 0.04)',
                border: '1px solid rgba(56, 189, 248, 0.15)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.6rem',
                fontSize: '0.825rem',
                color: 'var(--text-muted)',
                lineHeight: 1.5,
              }}>
                <Sparkles size={16} color="#38bdf8" style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <strong style={{ color: '#38bdf8' }}>AI Justification: </strong>
                  {rec.explanation_text}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Execute Button */}
      <div className="glass-panel" style={{ padding: '1.25rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'sticky', bottom: 15, zIndex: 40, background: 'rgba(7, 9, 14, 0.95)', border: '1px solid var(--border-glow)' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Ready to transform dataset with <strong>{approvedCount}</strong> approved fixes
          </span>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Executes safe-order scikit-learn pipeline and recalculates Data Health Score
          </p>
        </div>

        <button
          className="btn-primary"
          onClick={handleExecute}
          disabled={executing || approvedCount === 0}
          style={{ padding: '0.75rem 1.75rem', fontSize: '0.95rem' }}
        >
          {executing ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              <span>Executing Preprocessing Pipeline...</span>
            </>
          ) : (
            <>
              <Play size={16} fill="currentColor" />
              <span>Apply {approvedCount} Fixes & Verify</span>
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
    </div>
  );
};
