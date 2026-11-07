import React from 'react';
import { HealthScore } from '../services/api';
import { CountUp, Note, Panel, useInView } from './ui';

interface HealthScoreGaugeProps {
  score: HealthScore;
  stageName?: string;
  problemType?: string;
}

const toneFor = (v: number) => (v >= 85 ? 'var(--good)' : v >= 65 ? 'var(--accent)' : v >= 50 ? 'var(--warn)' : 'var(--bad)');

export const HealthScoreGauge: React.FC<HealthScoreGaugeProps> = ({ score, stageName = 'raw dataset', problemType }) => {
  const [ref, inView] = useInView<HTMLDivElement>(0.25);
  const comp = score.composite_score;
  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const offset = inView ? circumference - (comp / 100) * circumference : circumference;
  const color = toneFor(comp);

  // Mirrors backend scorer.task_weights: unsupervised tasks have no target, anomaly detection has no signal check.
  const all = [
    { key: 'signal', label: problemType === 'clustering' ? 'cluster structure' : 'predictive signal', weight: 20, val: score.sub_scores.signal_score ?? 100 },
    { key: 'completeness', label: 'completeness', weight: 20, val: score.sub_scores.missingness_score },
    { key: 'validity', label: 'validity & consistency', weight: 20, val: score.sub_scores.validity_score },
    { key: 'usefulness', label: 'feature usefulness', weight: 15, val: score.sub_scores.feature_quality_score },
    { key: 'target', label: 'target health', weight: 10, val: score.sub_scores.target_balance_score },
    { key: 'uniqueness', label: 'uniqueness', weight: 10, val: score.sub_scores.duplicate_score },
    { key: 'outliers', label: 'extreme values', weight: 5, val: score.sub_scores.outlier_score },
  ];
  const dropped = problemType === 'clustering' ? ['target'] : problemType === 'anomaly_detection' ? ['target', 'signal'] : [];
  const kept = all.filter((s) => !dropped.includes(s.key));
  const total = kept.reduce((sum, s) => sum + s.weight, 0);
  const subScoreList = kept.map((s) => ({ ...s, weight: Math.round((s.weight / total) * 100) }));

  const verdict = comp >= 85 ? 'ship it' : comp >= 65 ? 'needs a tune-up' : comp >= 50 ? 'rough shape' : 'yikes.';

  return (
    <Panel>
      <div className="panel-head">
        <div className="left">
          <span className="dots"><i /><i /><i /></span>
          <span>health_score · {stageName}</span>
        </div>
        <span>{subScoreList.length} checks · weighted geometric mean</span>
      </div>
      <div ref={ref} className={`score-grid ${inView ? 'in' : ''}`}>
        <div className="score-dial">
          <svg width="180" height="180" viewBox="0 0 180 180" style={{ transform: 'rotate(-90deg)' }} aria-hidden>
            <circle cx="90" cy="90" r={radius} stroke="var(--line-soft)" strokeWidth="10" fill="none" />
            <circle cx="90" cy="90" r="88" stroke="var(--line-strong)" strokeWidth="1" fill="none" strokeDasharray="2 6" />
            <circle
              className="ring"
              cx="90"
              cy="90"
              r={radius}
              stroke={color}
              strokeWidth="10"
              fill="none"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="score-num">
            <b style={{ color }}><CountUp value={comp} decimals={Number.isInteger(comp) ? 0 : 1} duration={1600} /></b>
            <span>/ 100</span>
          </div>
          <span className="grade-chip">grade <b>{score.grade}</b></span>
          <Note tone={comp >= 65 ? 'green' : 'red'} rot={-5} delay={1.4}>
            {verdict}
          </Note>
        </div>

        <div className="subscores">
          {subScoreList.map((sub, i) => (
            <div key={sub.label} className="sub-row">
              <div className="top">
                <span>{sub.label} <span style={{ color: 'var(--dim)' }}>·{sub.weight}%</span></span>
                <b style={{ color: toneFor(sub.val) }}>{sub.val}</b>
              </div>
              <div className="meter">
                <i style={{ ['--w' as string]: `${sub.val}%`, ['--d' as string]: `${200 + i * 90}ms`, background: toneFor(sub.val) }} />
              </div>
            </div>
          ))}
        </div>

        <div className="summary-line">
          <span className="tag">[verdict]</span>
          <span>{score.summary_text}</span>
        </div>
      </div>
    </Panel>
  );
};
