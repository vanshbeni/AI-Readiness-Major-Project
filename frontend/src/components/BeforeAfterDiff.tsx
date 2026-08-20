import React from 'react';
import { ExecutionResult } from '../services/api';
import { TrendingUp, CheckCircle, ListOrdered, ShieldCheck } from 'lucide-react';

interface BeforeAfterDiffProps {
  result: ExecutionResult;
}

export const BeforeAfterDiff: React.FC<BeforeAfterDiffProps> = ({ result }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Hero Improvement Badge */}
      <div
        className="glass-panel glow-cyan"
        style={{
          padding: '1.75rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(56, 189, 248, 0.08) 100%)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: 48,
            height: 48,
            borderRadius: '50%',
            background: 'rgba(16, 185, 129, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#10b981',
          }}>
            <TrendingUp size={24} />
          </div>
          <div>
            <span style={{ fontSize: '0.8rem', color: '#34d399', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Quality Optimization Verified
            </span>
            <h3 style={{ fontSize: '1.5rem', fontWeight: 800 }}>
              Health Score: {result.before_health_score} <span style={{ color: 'var(--text-dim)' }}>→</span> <span style={{ color: '#34d399' }}>{result.after_health_score} / 100</span>
            </h3>
          </div>
        </div>

        <div style={{
          padding: '0.6rem 1.25rem',
          borderRadius: 12,
          background: 'rgba(16, 185, 129, 0.2)',
          border: '1px solid rgba(16, 185, 129, 0.4)',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: '1.35rem', fontWeight: 800, color: '#34d399', display: 'block' }}>
            +{result.health_score_delta} pts
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            Quality Improvement
          </span>
        </div>
      </div>

      {/* Metric Deltas Table */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h4 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={18} color="#38bdf8" />
          <span>Before vs. After Quality Metric Diff</span>
        </h4>

        <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid var(--border)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border)' }}>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Quality Dimension</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Raw Ingested State</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Cleaned Pipeline Output</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Impact Delta</th>
              </tr>
            </thead>
            <tbody>
              {result.metric_deltas.map((delta, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-main)' }}>
                    {delta.metric_name}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#fb7185', fontFamily: 'monospace' }}>
                    {String(delta.before_value)}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#34d399', fontWeight: 600, fontFamily: 'monospace' }}>
                    {String(delta.after_value)}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className="badge badge-success" style={{ textTransform: 'none' }}>
                      {delta.improvement}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Applied Transformations Ordered List */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <h4 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ListOrdered size={18} color="#10b981" />
          <span>Executed Transformation Pipeline Sequence</span>
        </h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {result.transformations_applied.map((step, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                padding: '0.6rem 0.85rem',
                borderRadius: 8,
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border)',
                fontSize: '0.825rem',
                color: 'var(--text-muted)',
              }}
            >
              <CheckCircle size={15} color="#10b981" style={{ flexShrink: 0 }} />
              <span>{step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
