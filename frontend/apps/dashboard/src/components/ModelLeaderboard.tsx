import React from 'react';
import { BenchmarkLeaderboard } from '../services/api';
import { Eyebrow, Note, Panel, Reveal, useInView } from './ui';

interface ModelLeaderboardProps {
  leaderboard: BenchmarkLeaderboard;
}

const barWidth = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;

export const ModelLeaderboard: React.FC<ModelLeaderboardProps> = ({ leaderboard }) => {
  const [ref, inView] = useInView<HTMLDivElement>(0.1);
  const topModel = leaderboard.models[0];

  if (!topModel) {
    return (
      <Panel className="panel-pad">
        <p style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>No candidate model could be trained on this dataset.</p>
      </Panel>
    );
  }

  return (
    <div className="section-gap">
      <Reveal>
        <div className="dark-card">
          <span className="blob a" />
          <span className="blob b" />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: '2rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', maxWidth: 560 }}>
              <Eyebrow label="winner" index="#1" />
              <h2 style={{ fontSize: 'clamp(1.8rem, 3.6vw, 2.8rem)', letterSpacing: '-0.05em', fontWeight: 600, lineHeight: 1 }}>
                {topModel.model_name}
              </h2>
              <p className="muted" style={{ fontSize: '0.9rem', lineHeight: 1.6 }}>{topModel.description}</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="mono muted" style={{ fontSize: '0.68rem', letterSpacing: '0.08em' }}>
                {leaderboard.cv_folds}-FOLD CV · {leaderboard.primary_metric.toUpperCase()}
              </div>
              <div style={{ fontSize: 'clamp(2.8rem, 6vw, 4.4rem)', fontWeight: 600, letterSpacing: '-0.06em', lineHeight: 1 }}>
                {topModel.metric_display}
              </div>
              <div className="mono muted" style={{ fontSize: '0.7rem', marginTop: '0.4rem' }}>
                fit in {topModel.training_time_sec}s
              </div>
            </div>
          </div>
          <p className="muted" style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,.12)', fontSize: '0.82rem', lineHeight: 1.6 }}>
            {leaderboard.best_model_summary.replace(/\*\*/g, '')}
          </p>
        </div>
      </Reveal>

      <Reveal>
        <Panel>
          <div className="panel-head">
            <div className="left">
              <span className="dots"><i /><i /><i /></span>
              <span>leaderboard · {leaderboard.problem_type}</span>
            </div>
            <span>scaling & rebalancing fitted per fold</span>
          </div>
          <div ref={ref} className={`table-wrap ${inView ? 'in' : ''}`}>
            <table className="mono-table">
              <thead>
                <tr>
                  <th style={{ width: 56 }}>rank</th>
                  <th>model</th>
                  <th>{leaderboard.primary_metric}</th>
                  <th>fit time</th>
                  <th>suitability</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.models.map((model, i) => {
                  const best = model.rank === 1;
                  return (
                    <tr key={model.id} style={best ? { background: 'var(--accent-soft)' } : undefined}>
                      <td style={{ color: best ? 'var(--accent)' : 'var(--dim)', fontWeight: 600 }}>#{model.rank}</td>
                      <td style={{ color: 'var(--ink)', fontWeight: best ? 600 : 400 }}>
                        {model.model_name}
                        {best && <span className="chip accent" style={{ marginLeft: 8 }}>best</span>}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <span style={{ minWidth: 52, fontWeight: 600, color: best ? 'var(--good)' : 'var(--ink-2)' }}>{model.metric_display}</span>
                          <span className="lb-bar">
                            <i style={{ ['--w' as string]: barWidth(model.metric_value), ['--d' as string]: `${i * 120}ms`, background: best ? 'var(--accent)' : undefined }} />
                          </span>
                        </div>
                      </td>
                      <td style={{ color: 'var(--muted)' }}>{model.training_time_sec}s</td>
                      <td>
                        <span className={`chip ${model.suitability === 'High' ? 'good' : ''}`}>{model.suitability.toLowerCase()}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </Reveal>
      {leaderboard.models.length > 1 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '-0.75rem' }}>
          <Note rot={-2} delay={0.3}>all scored on held-out folds, no peeking</Note>
        </div>
      )}
    </div>
  );
};
