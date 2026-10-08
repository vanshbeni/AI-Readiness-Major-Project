import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { RecommendationItem } from '../services/api';
import { ArrowRight, Loader2, RefreshCw } from 'lucide-react';
import { Eyebrow, MagneticButton, Note, Panel, Reveal, Scramble, Underlined } from './ui';
import { Terminal } from './Terminal';

interface RecommendationChecklistProps {
  recommendations: RecommendationItem[];
  approvals: Record<string, boolean>;
  onToggle: (id: string) => void;
  onSetAll: (state: boolean) => void;
  onExecute: () => Promise<void>;
  onRegenerateExplanations?: () => Promise<void>;
}

const EXECUTION_LINES = [
  'saving approved rules & parameters',
  'deduplicating rows',
  'fixing invalid values',
  'capping outliers',
  'imputing missing values',
  'encoding categoricals',
  're-profiling cleaned dataset',
  'computing post-cleaning health score',
  'compiling diffs & export artifacts',
];

export const RecommendationChecklist: React.FC<RecommendationChecklistProps> = ({
  recommendations,
  approvals,
  onToggle,
  onSetAll,
  onExecute,
  onRegenerateExplanations,
}) => {
  const [executing, setExecuting] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [regenError, setRegenError] = useState<string | null>(null);

  const approvedCount = recommendations.filter((r) => approvals[r.id] ?? r.is_approved).length;
  const totalCount = recommendations.length;
  const statisticalCount = recommendations.filter((r) => r.explanation_source !== 'ai').length;

  const handleRegenerate = async () => {
    if (!onRegenerateExplanations) return;
    setRegenerating(true);
    setRegenError(null);
    try {
      await onRegenerateExplanations();
    } catch (err: any) {
      setRegenError(err?.message || 'AI explanations are unavailable right now.');
    } finally {
      setRegenerating(false);
    }
  };

  const handleExecute = async () => {
    setExecuting(true);
    try {
      await onExecute();
    } catch {
      // The page shows the error banner; close the overlay so the user can act on it.
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div>
      {executing && createPortal(
        <div className="overlay">
          <Terminal
            title="pipeline.log"
            command={`apply --fixes ${approvedCount} --safe-order`}
            lines={EXECUTION_LINES}
            stepMs={650}
          />
        </div>,
        document.body,
      )}

      <div className="step-head" style={{ marginTop: '0.5rem' }}>
        <Eyebrow label="approve-fixes" index="04/05" />
        <div className="step-head-row">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            <h1 className="h-title">
              <Scramble text="You're the" /> <Underlined>reviewer.</Underlined>
            </h1>
            <p className="h-sub">
              Each fix is picked by statistical rules and explained from the numbers. Only approved fixes run; scaling and
              class rebalancing happen inside model training so nothing leaks.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button className="btn btn-light btn-sm" onClick={() => onSetAll(true)} disabled={totalCount === 0}>
              approve all
            </button>
            <button className="btn btn-light btn-sm" onClick={() => onSetAll(false)} disabled={totalCount === 0}>
              reject all
            </button>
          </div>
        </div>
      </div>

      {onRegenerateExplanations && statisticalCount > 0 && (
        <div className="banner info" style={{ alignItems: 'center', marginBottom: '1.25rem' }}>
          <span className="tag">[ai]</span>
          <span style={{ flex: 1 }}>
            {regenError ?? `${statisticalCount} of ${totalCount} explanations are statistical (AI was unavailable or not configured).`}
          </span>
          <button className="btn btn-light btn-sm" onClick={handleRegenerate} disabled={regenerating || executing}>
            {regenerating ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            {regenerating ? 'asking gemini…' : 'retry ai explanations'}
          </button>
        </div>
      )}

      <div className="rec-list">
        {totalCount === 0 && (
          <Panel className="panel-pad" style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '1.05rem', fontWeight: 600, letterSpacing: '-0.02em', marginBottom: '0.4rem' }}>
              Nothing to fix. Clean as a whistle.
            </p>
            <p style={{ color: 'var(--muted)', fontSize: '0.86rem' }}>
              Continue to generate the model-ready dataset and benchmark models.
            </p>
          </Panel>
        )}

        {recommendations.map((rec, i) => {
          const isApproved = approvals[rec.id] ?? rec.is_approved;
          const isAi = rec.explanation_source === 'ai';
          return (
            <Reveal key={rec.id} delay={Math.min(i, 6) * 60}>
              <Panel lift className={`rec ${isApproved ? '' : 'off'}`}>
                <span className="rec-num">{String(i + 1).padStart(2, '0')}</span>
                <div style={{ minWidth: 0 }}>
                  <div className="rec-meta">
                    <span className={`sev ${rec.severity.toLowerCase()}`}><i />{rec.severity.toLowerCase()}</span>
                    {rec.column && <span className="chip">{rec.column}</span>}
                    {rec.is_destructive && <span className="chip bad">destructive</span>}
                  </div>
                  <h4 className="rec-title">{rec.reason_title}</h4>
                  <div className="rec-method">method → <b>{rec.method}</b></div>
                  <div className="rec-why">
                    <span className={`src ${isAi ? 'ai' : 'stat'}`}>{isAi ? '[ai]' : '[stats]'}</span>
                    <span>{rec.explanation_text}</span>
                  </div>
                </div>
                <span className="rec-toggle">
                  <span style={{ color: isApproved ? 'var(--good)' : 'var(--dim)', minWidth: 58, textAlign: 'right' }}>
                    {isApproved ? 'approved' : 'skipped'}
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={isApproved}
                    aria-label={`Approve ${rec.reason_title}`}
                    className={`switch-btn ${isApproved ? 'on' : ''}`}
                    onClick={() => onToggle(rec.id)}
                  />
                </span>
              </Panel>
            </Reveal>
          );
        })}
      </div>

      <div className="action-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span className="count">
            {approvedCount}<span style={{ color: 'rgba(255,255,255,.4)', fontSize: '1rem' }}>/{totalCount}</span>
          </span>
          <div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
              {approvedCount === 0 ? 'No fixes selected' : `fix${approvedCount === 1 ? '' : 'es'} ready to apply`}
            </div>
            <p>
              {approvedCount === 0
                ? 'The dataset is exported as-is and models are benchmarked on it.'
                : 'Runs in a safe order, then recalculates the health score.'}
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {approvedCount > 0 && !executing && (
            <Note tone="purple" rot={-5} delay={0.4} style={{ color: '#c4a7ff' }}>let's go</Note>
          )}
          <MagneticButton className="btn btn-light btn-lg" onClick={handleExecute} disabled={executing}>
            {executing ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Running pipeline…
              </>
            ) : (
              <>
                {approvedCount === 0 ? 'Continue without fixes' : `Apply ${approvedCount} fix${approvedCount === 1 ? '' : 'es'}`}
                <ArrowRight size={16} className="arrow" />
              </>
            )}
          </MagneticButton>
        </div>
      </div>
    </div>
  );
};
