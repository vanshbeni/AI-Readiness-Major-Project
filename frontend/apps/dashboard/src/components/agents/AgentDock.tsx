import React from 'react';
import { ArrowRight, Bot, FileText, Gauge, ListChecks, ScanSearch, Sparkles } from 'lucide-react';
import { AgentKind, AgentPolicy, AgentStatus } from '../../services/api';
import { Eyebrow, Note, Panel } from '../ui';
import { Markdown } from './Markdown';

type Step = 'diagnostic' | 'recommendations' | 'results';

const ACTIONS: Record<Step, { kind: AgentKind; icon: React.ElementType; title: string; body: string; primary?: boolean }[]> = {
  diagnostic: [
    { kind: 'autopilot', icon: Sparkles, title: 'Autopilot', body: 'Agents run the whole pipeline and stop only for your approval.', primary: true },
    { kind: 'understand', icon: ScanSearch, title: 'Understand columns', body: 'Find placeholders, number formats, day-first dates, variants, PII.' },
  ],
  recommendations: [
    { kind: 'clean', icon: ListChecks, title: 'Review the plan', body: 'The Clean agent picks better methods per issue, using target signal.', primary: true },
  ],
  results: [
    { kind: 'evaluate', icon: Gauge, title: 'Plan evaluation', body: 'Choose the metric and split that fit this data, then re-benchmark.' },
    { kind: 'report', icon: FileText, title: 'Write report', body: 'Executive summary where every number is verified.', primary: true },
  ],
};

export function AgentDock({
  step,
  status,
  policy,
  busy,
  onRun,
}: {
  step: Step;
  status: AgentStatus | null;
  policy?: AgentPolicy | null;
  busy: boolean;
  onRun: (kind: AgentKind) => void;
}) {
  const decided = policy ? Object.keys(policy.column_overrides || {}).length : 0;
  return (
    <Panel className="agent-dock">
      <div className="agent-dock-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: '.7rem' }}>
          <span className="agent-dock-icon"><Bot size={16} /></span>
          <div>
            <Eyebrow label="agents" />
            <p className="agent-dock-sub">
              {status?.configured
                ? <>Running on <b>{status.model}</b> via OpenRouter · privacy: {status.privacy_mode}</>
                : <>No LLM key: agents use their deterministic rules. Add <code>OPENROUTER_API_KEY</code> for full reasoning.</>}
            </p>
          </div>
        </div>
        {step === 'diagnostic' && decided > 0 && <span className="chip accent">{decided} column decision{decided === 1 ? '' : 's'} active</span>}
      </div>
      <div className="agent-actions">
        {ACTIONS[step].map((a) => (
          <button key={a.kind} type="button" className={`agent-action ${a.primary ? 'primary' : ''}`} disabled={busy} onClick={() => onRun(a.kind)}>
            <a.icon size={16} />
            <span>
              <b>{a.title}</b>
              <em>{a.body}</em>
            </span>
            <ArrowRight size={14} className="arrow" />
          </button>
        ))}
      </div>
      {step === 'results' && policy?.eval_plan && (
        <div className="agent-plan">
          <span className="tag">[evaluation]</span>
          <span>
            <b>{policy.eval_plan.metric_label}</b> · {policy.eval_plan.cv} split · up to {Number(policy.eval_plan.max_rows).toLocaleString()} rows
            <span className="dim"> · chosen by {policy.eval_plan.source}</span>
          </span>
        </div>
      )}
      {step === 'results' && policy?.report && (
        <div className="agent-report">
          <div className="agent-report-head">
            <Eyebrow label="report" />
            <Note tone={policy.report.source === 'agent' ? 'green' : undefined} rot={-2} delay={0.2}>
              {policy.report.source === 'agent' ? 'every number verified' : 'template report'}
            </Note>
          </div>
          <Markdown text={policy.report.markdown} />
        </div>
      )}
    </Panel>
  );
}
