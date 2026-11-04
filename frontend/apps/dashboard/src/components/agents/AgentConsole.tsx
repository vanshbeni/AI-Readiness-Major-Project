'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Loader2, X } from 'lucide-react';
import { AgentRun } from '../../services/api';
import { Markdown } from './Markdown';

export const AGENT_TITLES: Record<string, string> = {
  autopilot: 'Autopilot',
  understand: 'Understand agent',
  clean: 'Clean agent',
  evaluate: 'Evaluate agent',
  report: 'Report agent',
  ask: 'Analyst agent',
};

const KIND_LABEL: Record<string, string> = {
  plan: 'plan', tool: 'tool', llm: 'model', critic: 'critic', gate: 'gate', result: 'done', warn: 'note', error: 'error',
};

function elapsed(from: string, to: string) {
  const s = (new Date(to).getTime() - new Date(from).getTime()) / 1000;
  return Number.isFinite(s) ? `+${s.toFixed(1)}s` : '';
}

function GateCard({ run, onResume, onCancel }: { run: AgentRun; onResume: (a: Record<string, boolean>, q: Record<string, number>) => void; onCancel: () => void }) {
  const gate = run.gate!;
  const [checked, setChecked] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(gate.items.map((it) => [it.id, it.default ?? true])),
  );
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [sending, setSending] = useState(false);
  const count = Object.values(checked).filter(Boolean).length;

  return (
    <div className="gate">
      <div className="gate-head">
        <span className="gate-tag">human gate{gate.stage ? ` · ${gate.stage}` : ''}</span>
        <h4>{gate.title}</h4>
        <p>Nothing changes until you approve. Unchecked items are skipped.</p>
      </div>
      <div className="gate-items">
        {gate.items.map((it) => (
          <label key={it.id} className={`gate-item ${checked[it.id] ? 'on' : ''}`}>
            <input
              type="checkbox"
              checked={!!checked[it.id]}
              onChange={() => setChecked((c) => ({ ...c, [it.id]: !c[it.id] }))}
            />
            <span className="gate-box">{checked[it.id] && <Check size={11} strokeWidth={3} />}</span>
            <span className="gate-body">
              <span className="gate-label">{it.label}</span>
              {it.from && it.to && <span className="gate-diff"><s>{it.from}</s> → <b>{it.to}</b></span>}
              {it.reason && <span className="gate-reason">{it.reason}</span>}
              <span className="gate-meta">
                {it.source && <span className="chip">{it.source}</span>}
                {typeof it.confidence === 'number' && (
                  <span className="conf" title="Confidence">
                    <i><b style={{ width: `${Math.round(it.confidence * 100)}%` }} /></i>
                    <em>{Math.round(it.confidence * 100)}%</em>
                  </span>
                )}
                {it.destructive && <span className="chip bad">destructive</span>}
              </span>
            </span>
          </label>
        ))}
      </div>
      {gate.questions && gate.questions.length > 0 && (
        <div className="gate-questions">
          {gate.questions.map((q) => (
            <div key={q.id} className="gate-q">
              <span className="gate-label">{q.question}</span>
              <div className="gate-opts">
                {q.options.map((o, i) => (
                  <button key={i} type="button" className={`toggle-chip ${answers[q.id] === i ? 'on' : ''}`}
                    onClick={() => setAnswers((a) => ({ ...a, [q.id]: i }))}>
                    <i />{o.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="gate-actions">
        <button className="btn btn-ghost btn-sm" onClick={onCancel} disabled={sending}>cancel run</button>
        <button
          className="btn btn-light btn-sm"
          disabled={sending}
          onClick={() => { setSending(true); onResume(checked, answers); }}
        >
          {sending ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
          Approve {count} of {gate.items.length}{gate.questions?.length ? ' + answers' : ''}
        </button>
      </div>
    </div>
  );
}

function ResultCard({ run }: { run: AgentRun }) {
  const r = run.result || {};
  if (run.kind === 'understand') {
    return (
      <div className="agent-result">
        <h4>{r.applied?.length ? `${r.applied.length} column decision(s) saved` : 'No changes needed'}</h4>
        <ul>{(r.applied || []).map((a: string) => <li key={a}>{a}</li>)}</ul>
        {r.notes && Object.keys(r.notes).length > 0 && (
          <div className="agent-notes">
            {Object.entries(r.notes).slice(0, 8).map(([c, n]) => <div key={c}><b>{c}</b> {String(n)}</div>)}
          </div>
        )}
        {r.needs_rediagnose && <p className="dim">Diagnostics will re-run with these decisions.</p>}
      </div>
    );
  }
  if (run.kind === 'clean') {
    return (
      <div className="agent-result">
        <h4>{r.changed?.length ? `${r.changed.length} fix(es) updated` : 'Plan unchanged'}</h4>
        <ul>{(r.changed || []).map((a: string) => <li key={a}>{a}</li>)}</ul>
      </div>
    );
  }
  if (run.kind === 'evaluate') {
    const p = r.plan || {};
    return (
      <div className="agent-result">
        <h4>{r.saved ? 'Evaluation plan saved' : 'Plan not saved'}</h4>
        <p><b>{p.metric_label}</b> · {p.cv} split · up to {Number(p.max_rows || 0).toLocaleString()} rows</p>
        {p.reason && <p className="dim">{p.reason}</p>}
      </div>
    );
  }
  if (run.kind === 'report' || (run.kind === 'autopilot' && r.report)) {
    return (
      <div className="agent-result">
        {run.kind === 'autopilot' && (
          <h4>Health {r.health_before} → {r.health_after} · best {r.best_model} ({r.metric} {r.score})</h4>
        )}
        <Markdown text={(r.report || {}).markdown || ''} />
      </div>
    );
  }
  return null;
}

export function AgentConsole({
  run,
  error,
  onResume,
  onCancel,
  onClose,
}: {
  run: AgentRun | null;
  error: string | null;
  onResume: (approvals: Record<string, boolean>, answers: Record<string, number>) => void;
  onCancel: () => void;
  onClose: () => void;
}) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' });
  }, [run?.steps.length, run?.status]);

  const live = run && (run.status === 'queued' || run.status === 'running');
  const first = run?.steps[0]?.t;
  const usage = run?.usage;
  const statusText = useMemo(() => {
    if (!run) return '';
    return { queued: 'queued', running: 'working', waiting: 'needs you', completed: 'done', failed: 'failed', cancelled: 'cancelled' }[run.status];
  }, [run]);

  if (!mounted || (!run && !error)) return null;
  return createPortal(
    <div className="agent-overlay" onClick={(e) => e.target === e.currentTarget && !live && onClose()}>
      <aside className="agent-console" role="dialog" aria-label="Agent console">
        <header className="agent-top">
          <span className="dots"><i /><i /><i /></span>
          <span className="agent-name">{run ? AGENT_TITLES[run.kind] : 'Agent'}</span>
          {run && <span className={`agent-status ${run.status}`}><i />{statusText}</span>}
          <span style={{ flex: 1 }} />
          {usage && usage.calls > 0 && (
            <span className="agent-usage" title={usage.models.join(', ')}>
              {usage.calls} call{usage.calls === 1 ? '' : 's'} · {(usage.prompt_tokens + usage.completion_tokens).toLocaleString()} tok
            </span>
          )}
          <button className="agent-x" onClick={live ? onCancel : onClose} aria-label={live ? 'Cancel' : 'Close'} title={live ? 'Cancel run' : 'Close'}>
            <X size={15} />
          </button>
        </header>

        <div className="agent-body" ref={bodyRef}>
          {run?.steps.map((s) => (
            <div key={s.i} className={`agent-step k-${s.kind}`}>
              <span className="agent-t">{first ? elapsed(first, s.t) : ''}</span>
              <span className="agent-actor">{s.actor}</span>
              <span className="agent-kind">{KIND_LABEL[s.kind] || s.kind}</span>
              <span className="agent-text">
                {s.text}
                {s.data != null && (
                  <details>
                    <summary>data</summary>
                    <pre>{JSON.stringify(s.data, null, 2).slice(0, 4000)}</pre>
                  </details>
                )}
              </span>
            </div>
          ))}
          {live && (
            <div className="agent-step k-live">
              <span className="agent-t" />
              <span className="agent-actor">…</span>
              <span className="agent-kind" />
              <span className="agent-text"><span className="term-caret" /></span>
            </div>
          )}
          {run?.status === 'waiting' && run.gate && <GateCard key={run.step_count} run={run} onResume={onResume} onCancel={onCancel} />}
          {run?.status === 'completed' && <ResultCard run={run} />}
          {run?.status === 'failed' && <div className="agent-error">Run failed: {run.error}</div>}
          {error && <div className="agent-error">{error}</div>}
        </div>

        {run && !live && run.status !== 'waiting' && (
          <footer className="agent-foot">
            <button className="btn btn-light btn-sm" onClick={onClose}>{run.status === 'completed' ? 'Done' : 'Close'}</button>
          </footer>
        )}
      </aside>
    </div>,
    document.body,
  );
}
