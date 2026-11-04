import React from 'react';
import { BenchmarkLeaderboard, ClusterProfile, AnomalyRow, ModelBenchmarkItem } from '../services/api';
import { Eyebrow, Note, Panel, Reveal, useInView } from './ui';
import { PALETTE } from './viz/lib';

interface ModelLeaderboardProps {
  leaderboard: BenchmarkLeaderboard;
}

const SPLIT_LABEL: Record<string, string> = { stratified: 'stratified', kfold: 'K-fold', time: 'time-ordered', none: 'no split' };

const VERDICT: Record<string, { tone: string; title: string; body: (task: string) => string }> = {
  useful: { tone: 'good', title: 'Real, usable signal', body: (t) => (t === 'clustering' ? 'The groups separate clearly better than a no-structure reference.' : t === 'anomaly_detection' ? 'The detectors agree on which rows are unusual.' : 'The best model clearly beats no-skill guessing.') },
  weak: { tone: 'warn', title: 'Weak signal', body: (t) => (t === 'clustering' ? 'Groups exist but overlap: treat them as soft segments.' : t === 'anomaly_detection' ? 'Detectors only partly agree: review flags before acting.' : 'Better than guessing, but only slightly: validate on fresh data.') },
  no_signal: { tone: 'bad', title: 'No usable signal', body: (t) => (t === 'clustering' ? 'No natural groups: any segmentation would be arbitrary.' : t === 'anomaly_detection' ? 'Detectors disagree: there are no clear anomalies.' : 'No model beats guessing: do not deploy.') },
};

function extraCells(task: string, m: ModelBenchmarkItem) {
  const e = m.extra || {};
  if (task === 'clustering') {
    return [
      e.clusters ?? '—',
      e.davies_bouldin != null ? e.davies_bouldin : '—',
      e.noise_pct != null ? `${e.noise_pct}%` : '—',
    ];
  }
  if (task === 'anomaly_detection') return [e.flagged ?? '—', e.overlap_with_consensus ?? '—'];
  return [];
}

function ClusterCards({ clusters, features }: { clusters: ClusterProfile[]; features?: number }) {
  const total = clusters.reduce((s, c) => s + c.size, 0) || 1;
  return (
    <Panel>
      <div className="panel-head">
        <div className="left"><span className="dots"><i /><i /><i /></span><span>segments found · what sets each apart</span></div>
        <span>difference from the overall average, in standard deviations</span>
      </div>
      <div className="seg-bar" aria-hidden>
        {clusters.map((c, i) => (
          <i key={c.cluster} style={{ width: `${(c.size / total) * 100}%`, background: c.noise ? 'var(--line-strong)' : PALETTE[i % PALETTE.length], ['--d' as string]: `${i * 120}ms` }} />
        ))}
      </div>
      <div className="cluster-grid">
        {clusters.map((c, i) => (
          <Reveal key={c.cluster} delay={i * 70}>
            <div className="cluster-card">
              <div className="cluster-top">
                <span className="cluster-dot" style={{ background: c.noise ? 'var(--line-strong)' : PALETTE[i % PALETTE.length] }} />
                <b>{c.noise ? 'noise / outliers' : `cluster ${c.cluster + 1}`}</b>
                <span className="dim mono">{c.size.toLocaleString()} rows · {c.share}%</span>
              </div>
              {c.traits.length ? c.traits.map((t) => (
                <div key={t.feature} className="trait">
                  <span className="trait-name" title={t.feature}>{t.feature}</span>
                  <span className="trait-bar"><i className={t.z > 0 ? 'up' : 'down'} style={{ width: `${Math.min(100, Math.abs(t.z) * 33)}%` }} /></span>
                  <span className="trait-val mono">
                    {t.z > 0 ? '+' : ''}{t.z}σ
                    <span className="dim"> · {t.unit === '%' ? `${t.cluster}% vs ${t.overall}%` : `${+t.cluster.toPrecision(4)} vs ${+t.overall.toPrecision(4)}`}</span>
                  </span>
                </div>
              )) : <p className="dim" style={{ fontSize: '.78rem' }}>Close to the overall average on every feature.</p>}
            </div>
          </Reveal>
        ))}
      </div>
      {features != null && <p className="dim mono" style={{ fontSize: '.66rem', padding: '0 1.25rem 1rem' }}>clustered on {features} encoded features</p>}
    </Panel>
  );
}

function AnomalyTable({ rows, consensus, contamination, detectors }: { rows: AnomalyRow[]; consensus?: number; contamination?: number; detectors?: string[] }) {
  return (
    <Panel>
      <div className="panel-head">
        <div className="left"><span className="dots"><i /><i /><i /></span><span>most anomalous rows · why they stand out</span></div>
        <span>{consensus ?? 0} row(s) flagged by a majority · expected {contamination != null ? (contamination * 100).toFixed(1) : '?'}%</span>
      </div>
      <div className="table-wrap">
        <table className="mono-table">
          <thead><tr><th>row</th><th>votes</th><th>score</th><th>what is unusual (robust z = distance from the median)</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={String(r.row)}>
                <td style={{ fontWeight: 600 }}>#{r.row}</td>
                <td><span className="chip bad">{r.votes}/{r.of}</span></td>
                <td>{(r.score * 100).toFixed(1)}</td>
                <td>
                  <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap' }}>
                    {r.reasons.length ? r.reasons.map((x) => (
                      <span key={x.feature} className="chip" title={`value ${x.value}`}>
                        {x.feature} = {+x.value.toPrecision(5)} <b style={{ color: Math.abs(x.robust_z) > 3.5 ? 'var(--bad)' : 'var(--warn)' }}>{x.robust_z > 0 ? '+' : ''}{x.robust_z}z</b>
                      </span>
                    )) : <span className="dim">unusual combination of values</span>}
                  </div>
                </td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={4} className="dim">No row was flagged by most detectors.</td></tr>}
          </tbody>
        </table>
      </div>
      {detectors && <p className="dim mono" style={{ fontSize: '.66rem', padding: '.6rem 1.25rem 1rem' }}>detectors: {detectors.join(' · ')}</p>}
    </Panel>
  );
}

export const ModelLeaderboard: React.FC<ModelLeaderboardProps> = ({ leaderboard }) => {
  const [ref, inView] = useInView<HTMLDivElement>(0.1);
  const topModel = leaderboard.models[0];
  const ev = leaderboard.evaluation;
  const task = leaderboard.problem_type;
  const unsupervised = task === 'clustering' || task === 'anomaly_detection';
  const lowerIsBetter = ev ? !ev.higher_is_better : false;
  const values = leaderboard.models.map((m) => m.metric_value).filter((v) => v > 0);
  const bestErr = values.length ? Math.min(...values) : 1;
  const barWidth = (v: number) =>
    `${Math.max(0, Math.min(1, lowerIsBetter ? (v > 0 ? bestErr / v : 0) : v)) * 100}%`;
  const extraHeads = task === 'clustering' ? ['clusters', 'davies-bouldin ↓', 'noise'] : task === 'anomaly_detection' ? ['flagged', 'agree w/ majority'] : [];
  const verdict = ev?.verdict ? VERDICT[ev.verdict] : null;

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
              <Eyebrow label={ev?.verdict === 'no_signal' ? 'best of a weak field' : 'winner'} index="#1" />
              <h2 style={{ fontSize: 'clamp(1.8rem, 3.6vw, 2.8rem)', letterSpacing: '-0.05em', fontWeight: 600, lineHeight: 1 }}>
                {topModel.model_name}
              </h2>
              <p className="muted" style={{ fontSize: '0.9rem', lineHeight: 1.6 }}>{topModel.description}</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="mono muted" style={{ fontSize: '0.68rem', letterSpacing: '0.08em' }}>
                {unsupervised
                  ? `${task === 'clustering' ? `${topModel.extra?.clusters ?? '?'} CLUSTERS` : `${topModel.extra?.flagged ?? '?'} FLAGGED`} · ${leaderboard.primary_metric.toUpperCase()}`
                  : `${leaderboard.cv_folds}-FOLD ${ev ? SPLIT_LABEL[ev.cv]?.toUpperCase() + ' ' : ''}CV · ${leaderboard.primary_metric.toUpperCase()}`}
              </div>
              <div style={{ fontSize: 'clamp(2.8rem, 6vw, 4.4rem)', fontWeight: 600, letterSpacing: '-0.06em', lineHeight: 1 }}>
                {topModel.metric_display}
              </div>
              <div className="mono muted" style={{ fontSize: '0.7rem', marginTop: '0.4rem' }}>
                {topModel.metric_std != null && <>± {topModel.metric_std} across folds · </>}fit in {topModel.training_time_sec}s
              </div>
            </div>
          </div>
          <p className="muted" style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,.12)', fontSize: '0.82rem', lineHeight: 1.6 }}>
            {leaderboard.best_model_summary.replace(/\*\*/g, '')}
          </p>
        </div>
      </Reveal>

      {verdict && (
        <Reveal>
          <div className={`verdict-band ${verdict.tone}`}>
            <span className="verdict-dot" />
            <b>{verdict.title}</b>
            <span>{verdict.body(task)}</span>
            {ev?.lift_over_baseline != null && <span className="mono dim">lift over reference {ev.lift_over_baseline > 0 ? '+' : ''}{ev.lift_over_baseline}</span>}
          </div>
        </Reveal>
      )}

      <Reveal>
        <Panel>
          <div className="panel-head">
            <div className="left">
              <span className="dots"><i /><i /><i /></span>
              <span>leaderboard · {task.replace('_', ' ')}</span>
            </div>
            <span>
              {ev
                ? unsupervised
                  ? <>{ev.rows_used.toLocaleString()} of {ev.rows_available.toLocaleString()} rows fitted · every row labelled</>
                  : <>{SPLIT_LABEL[ev.cv]} split · {ev.rows_used.toLocaleString()} of {ev.rows_available.toLocaleString()} rows · plan by {ev.source}</>
                : 'scaling & rebalancing fitted per fold'}
            </span>
          </div>
          {ev?.reason && (
            <div className="summary-line" style={{ borderTop: 'none', borderBottom: '1px solid var(--line)' }}>
              <span className="tag">[evaluation]</span>
              <span>{ev.reason}{ev.skipped_models.length ? ` Skipped for speed: ${ev.skipped_models.join(', ')}.` : ''}</span>
            </div>
          )}
          <div ref={ref} className={`table-wrap ${inView ? 'in' : ''}`}>
            <table className="mono-table">
              <thead>
                <tr>
                  <th style={{ width: 56 }}>rank</th>
                  <th>{unsupervised ? 'method' : 'model'}</th>
                  <th>{leaderboard.primary_metric}</th>
                  {extraHeads.map((h) => <th key={h}>{h}</th>)}
                  <th>fit time</th>
                  <th>suitability</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.models.map((model, i) => {
                  const best = model.rank === 1 && !model.is_baseline;
                  const base = !!model.is_baseline;
                  return (
                    <tr key={model.id} style={best ? { background: 'var(--accent-soft)' } : base ? { opacity: 0.6 } : undefined}>
                      <td style={{ color: best ? 'var(--accent)' : 'var(--dim)', fontWeight: 600 }}>{base ? '—' : `#${model.rank}`}</td>
                      <td style={{ color: 'var(--ink)', fontWeight: best ? 600 : 400 }}>
                        {model.model_name}
                        {best && <span className="chip accent" style={{ marginLeft: 8 }}>best</span>}
                        {base && <span className="chip" style={{ marginLeft: 8 }}>{unsupervised ? 'reference' : 'baseline'}</span>}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <span style={{ minWidth: 52, fontWeight: 600, color: best ? 'var(--good)' : 'var(--ink-2)' }}>
                            {model.metric_display}
                            {model.metric_std != null && <span className="dim" style={{ fontWeight: 400, fontSize: '.66rem' }}> ±{model.metric_std}</span>}
                          </span>
                          <span className="lb-bar">
                            <i style={{ ['--w' as string]: barWidth(model.metric_value), ['--d' as string]: `${i * 120}ms`, background: best ? 'var(--accent)' : undefined }} />
                          </span>
                        </div>
                      </td>
                      {extraCells(task, model).map((v, j) => <td key={j} style={{ color: 'var(--muted)' }}>{String(v)}</td>)}
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

      {task === 'clustering' && ev?.clusters?.length ? <Reveal><ClusterCards clusters={ev.clusters} /></Reveal> : null}
      {task === 'anomaly_detection' && ev ? (
        <Reveal><AnomalyTable rows={ev.anomalies || []} consensus={ev.consensus_count} contamination={ev.contamination} detectors={ev.detectors} /></Reveal>
      ) : null}

      {leaderboard.models.length > 1 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '-0.75rem' }}>
          <Note rot={-2} delay={0.3}>{unsupervised ? 'judged against a no-structure reference' : 'all scored on held-out folds'}</Note>
        </div>
      )}
    </div>
  );
};
