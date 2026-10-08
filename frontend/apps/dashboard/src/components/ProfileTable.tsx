import React from 'react';
import { ColumnProfile } from '../services/api';
import { Panel, Reveal, useInView } from './ui';

interface ProfileTableProps {
  columns: ColumnProfile[];
}

const TYPE_TONE: Record<string, string> = {
  numeric: 'accent',
  categorical: '',
  boolean: 'good',
  datetime: 'warn',
  id: '',
};

const fmt = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: Math.abs(v) >= 1000 ? 0 : 3 });

export const ProfileTable: React.FC<ProfileTableProps> = ({ columns }) => {
  const [ref, inView] = useInView<HTMLDivElement>(0.05);

  return (
    <Reveal>
      <Panel>
        <div className="panel-head">
          <div className="left">
            <span className="dots"><i /><i /><i /></span>
            <span>profile.columns</span>
          </div>
          <span>{columns.length} features</span>
        </div>
        <div ref={ref} className={`table-wrap ${inView ? 'in' : ''}`}>
          <table className="mono-table">
            <thead>
              <tr>
                <th>column</th>
                <th>type</th>
                <th>missing</th>
                <th>unique</th>
                <th>distribution</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((col, i) => {
                const missTone = col.missing_pct > 20 ? 'var(--bad)' : col.missing_pct > 0 ? 'var(--warn)' : 'var(--good)';
                return (
                  <tr key={col.name} style={col.is_target ? { background: 'var(--accent-soft)' } : undefined}>
                    <td style={{ fontWeight: 600, color: col.is_target ? 'var(--accent)' : 'var(--ink)' }}>
                      {col.name}
                      {col.is_target && <span className="chip accent" style={{ marginLeft: 8 }}>★ target</span>}
                    </td>
                    <td>
                      <span className={`chip ${TYPE_TONE[col.inferred_type] ?? ''}`}>{col.inferred_type}</span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <div className="meter" style={{ width: 54 }}>
                          <i style={{ ['--w' as string]: `${Math.max(col.missing_pct, col.missing_pct > 0 ? 4 : 0)}%`, ['--d' as string]: `${i * 30}ms`, background: missTone }} />
                        </div>
                        <span style={{ color: col.missing_pct > 0 ? missTone : 'var(--dim)' }}>
                          {col.missing_count} <span style={{ color: 'var(--dim)' }}>({col.missing_pct}%)</span>
                        </span>
                      </div>
                    </td>
                    <td style={{ color: 'var(--muted)' }}>
                      {col.unique_count} <span style={{ color: 'var(--dim)' }}>({col.unique_pct}%)</span>
                    </td>
                    <td style={{ color: 'var(--muted)', fontSize: '0.7rem' }}>
                      {col.numeric_stats ? (
                        <span>
                          μ <b style={{ color: 'var(--ink-2)', fontWeight: 500 }}>{fmt(col.numeric_stats.mean)}</b>
                          {' · '}med {fmt(col.numeric_stats.median)}
                          {' · '}[{fmt(col.numeric_stats.min)}, {fmt(col.numeric_stats.max)}]
                          {' · '}skew <span style={{ color: Math.abs(col.numeric_stats.skewness) > 1 ? 'var(--warn)' : undefined }}>{col.numeric_stats.skewness}</span>
                        </span>
                      ) : col.categorical_stats && col.categorical_stats.top_categories.length > 0 ? (
                        <span>
                          top: {col.categorical_stats.top_categories.slice(0, 3).map((c) => `${c.value} ${c.percentage}%`).join(' · ')}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--dim)' }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </Reveal>
  );
};
