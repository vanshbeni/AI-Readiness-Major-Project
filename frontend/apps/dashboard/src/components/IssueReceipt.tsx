import React from 'react';
import { IssueItem } from '../services/api';
import { Note, Panel, Reveal, useInView } from './ui';

const SEVERITY_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };

export const IssueReceipt: React.FC<{ issues: IssueItem[] }> = ({ issues }) => {
  const [ref, inView] = useInView<HTMLDivElement>(0.1);
  const sorted = [...issues].sort(
    (a, b) => (SEVERITY_ORDER[a.severity.toLowerCase()] ?? 9) - (SEVERITY_ORDER[b.severity.toLowerCase()] ?? 9),
  );
  const counts = sorted.reduce<Record<string, number>>((acc, i) => {
    const k = i.severity.toLowerCase();
    acc[k] = (acc[k] ?? 0) + 1;
    return acc;
  }, {});
  const serious = (counts.critical ?? 0) + (counts.high ?? 0);

  return (
    <Reveal>
      <Panel>
        <div className="panel-head">
          <div className="left">
            <span className="dots"><i /><i /><i /></span>
            <span>issues.receipt</span>
          </div>
          <span>{issues.length} line item{issues.length === 1 ? '' : 's'}</span>
        </div>
        <div ref={ref} className="receipt">
          {sorted.map((issue, i) => (
            <div
              key={issue.id}
              className="receipt-row"
              style={{
                opacity: inView ? 1 : 0,
                transform: inView ? 'none' : 'translateX(-10px)',
                transition: `opacity .5s ease ${i * 45}ms, transform .5s var(--ease-out) ${i * 45}ms`,
              }}
            >
              <span className={`sev ${issue.severity.toLowerCase()}`}><i />{issue.severity.toLowerCase()}</span>
              <span className="what" title={issue.title}>{issue.title}</span>
              <span className="col">{issue.column ?? 'dataset'}</span>
            </div>
          ))}
        </div>
        <div className="receipt-total">
          <span style={{ color: 'var(--muted)' }}>
            {Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(' · ')}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {serious > 0 && <Note tone="red" rot={-3} delay={0.2} style={{ fontSize: '1.1rem' }}>fix these first</Note>}
            <b>total {issues.length}</b>
          </span>
        </div>
      </Panel>
    </Reveal>
  );
};
