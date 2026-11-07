'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, Loader2, ShieldCheck, ShieldAlert } from 'lucide-react';
import { api, AgentRun, AgentStatus, ExploreColumn } from '../../services/api';
import { Panel } from '../ui';
import { AgentChart } from './AgentChart';
import { Markdown } from './Markdown';
import { useAgentRun } from './useAgentRun';

interface Turn {
  question: string;
  run: AgentRun;
}

function suggestions(columns: ExploreColumn[], target: string | null): string[] {
  const num = columns.filter((c) => c.kind === 'numeric' && !c.is_target).map((c) => c.name);
  const cat = columns.filter((c) => c.kind === 'categorical' && !c.is_target).map((c) => c.name);
  const out: string[] = [];
  if (target) out.push(`What drives ${target}? Show the strongest factors.`);
  if (num[0] && cat[0]) out.push(`How does ${num[0]} differ across ${cat[0]}?`);
  if (target && cat[0]) out.push(`What is the ${target} rate for each ${cat[0]}?`);
  out.push('Which columns have the most missing values, and does it matter?');
  if (num[0] && num[1]) out.push(`Is ${num[0]} related to ${num[1]}?`);
  return out.slice(0, 4);
}

function Answer({ run }: { run: AgentRun }) {
  const r = run.result || {};
  const v = r.verification;
  const [showTrace, setShowTrace] = useState(false);
  if (r.error) return <div className="agent-error">{r.answer}</div>;
  return (
    <div className="ask-answer">
      <Markdown text={r.answer || ''} />
      <div className="ask-meta">
        {v && (
          <span className={`ask-verify ${v.unverified_numbers?.length ? 'warn' : 'ok'}`}>
            {v.unverified_numbers?.length ? <ShieldAlert size={12} /> : <ShieldCheck size={12} />}
            {v.unverified_numbers?.length
              ? `${v.unverified_numbers.length} number(s) not verified: ${v.unverified_numbers.slice(0, 3).join(', ')}`
              : `${v.verified_numbers} number(s) verified against query results`}
          </span>
        )}
        {run.usage && run.usage.calls > 0 && <span className="dim">{run.usage.calls} model calls</span>}
        <button type="button" className="ask-trace-toggle" onClick={() => setShowTrace((s) => !s)}>
          {showTrace ? 'hide' : 'show'} reasoning trace ({run.steps.length})
        </button>
      </div>
      {(r.charts || []).map((c: any) => (
        <div key={c.id} className="ask-chart">
          <div className="ask-chart-title"><span className="cite">{c.id}</span> {c.title} <span className="dim">from {c.table}</span></div>
          <AgentChart spec={c} />
        </div>
      ))}
      {(r.tables || []).length > 0 && (
        <details className="ask-tables">
          <summary>source tables ({r.tables.length})</summary>
          {r.tables.map((t: any) => (
            <div key={t.id} className="table-wrap" style={{ marginTop: '.5rem' }}>
              <div className="dim mono" style={{ fontSize: '.64rem', marginBottom: '.25rem' }}><span className="cite">{t.id}</span> {t.title}</div>
              <table className="mono-table">
                <thead><tr>{t.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>
                  {t.rows.slice(0, 15).map((row: any[], i: number) => (
                    <tr key={i}>{row.map((cell, j) => <td key={j}>{cell === null ? '—' : String(cell)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </details>
      )}
      {showTrace && (
        <div className="ask-trace">
          {run.steps.map((s) => (
            <div key={s.i} className={`agent-step k-${s.kind}`}>
              <span className="agent-actor">{s.actor}</span>
              <span className="agent-kind">{s.kind}</span>
              <span className="agent-text">{s.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function AskPanel({
  datasetId,
  columns,
  target,
  source,
}: {
  datasetId: string;
  columns: ExploreColumn[];
  target: string | null;
  source: 'raw' | 'cleaned';
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState('');
  const [pending, setPending] = useState<string | null>(null);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const agent = useAgentRun((run) => {
    setTurns((t) => [...t, { question: run.input?.question || pending || '', run }]);
    setPending(null);
  });

  useEffect(() => {
    api.agentStatus().then(setStatus).catch(() => undefined);
    api.listAgentRuns(datasetId, 'ask', 6)
      .then((runs) => setTurns(runs.filter((r) => r.status === 'completed').reverse().map((r) => ({ question: r.input?.question || '', run: r }))))
      .catch(() => undefined);
  }, [datasetId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns.length, agent.run?.steps.length]);

  const ideas = useMemo(() => suggestions(columns, target), [columns, target]);
  const busy = !!pending;

  const ask = (q: string) => {
    const text = q.trim();
    if (!text || busy) return;
    setPending(text);
    setQuestion('');
    const history = turns.slice(-3).map((t) => ({ q: t.question, a: String(t.run.result?.answer || '').slice(0, 600) }));
    agent.start(datasetId, 'ask', { question: text, history, source });
  };

  const live = agent.run && pending ? agent.run : null;

  return (
    <Panel className="ask-panel">
      <div className="panel-head">
        <div className="left"><span className="dots"><i /><i /><i /></span><span>analyst agent · ask your data</span></div>
        <span>{status?.configured ? `openrouter · ${status.model}` : 'needs OPENROUTER_API_KEY'} · read-only SQL · numbers verified</span>
      </div>
      <div className="ask-thread">
        {turns.length === 0 && !pending && (
          <div className="ask-empty">
            <p>Ask anything about this dataset. The agent plans queries, runs them in a read-only sandbox, and every number in the answer is checked against the results.</p>
            <div className="ask-ideas">
              {ideas.map((s) => <button key={s} type="button" className="toggle-chip" onClick={() => ask(s)}><i />{s}</button>)}
            </div>
          </div>
        )}
        {turns.map((t) => (
          <div key={t.run.id} className="ask-turn">
            <div className="ask-q">{t.question}</div>
            <Answer run={t.run} />
          </div>
        ))}
        {pending && (
          <div className="ask-turn">
            <div className="ask-q">{pending}</div>
            <div className="ask-live">
              {(live?.steps || []).slice(-6).map((s) => (
                <div key={s.i} className={`agent-step k-${s.kind}`}>
                  <span className="agent-actor">{s.actor}</span>
                  <span className="agent-kind">{s.kind}</span>
                  <span className="agent-text">{s.text}</span>
                </div>
              ))}
              <div className="agent-step k-live"><span className="agent-actor"><Loader2 size={12} className="animate-spin" /></span><span className="agent-kind" /><span className="agent-text"><span className="term-caret" /></span></div>
            </div>
          </div>
        )}
        {agent.error && <div className="agent-error">{agent.error}</div>}
        <div ref={endRef} />
      </div>
      <form className="ask-input" onSubmit={(e) => { e.preventDefault(); ask(question); }}>
        <textarea
          value={question}
          rows={1}
          placeholder={busy ? 'The agent is working…' : 'Ask a question, e.g. "Which segment has the highest churn?"'}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              ask(question);
            }
          }}
          disabled={busy}
        />
        <button className="btn btn-dark btn-sm" type="submit" disabled={busy || !question.trim()} aria-label="Ask">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <ArrowUp size={14} />}
        </button>
      </form>
    </Panel>
  );
}
