import React from 'react';
import { HealthScore } from '../services/api';
import { Info } from 'lucide-react';

interface HealthScoreGaugeProps {
  score: HealthScore;
  stageName?: string;
}

export const HealthScoreGauge: React.FC<HealthScoreGaugeProps> = ({ score, stageName = 'Raw Ingested Dataset' }) => {
  const comp = score.composite_score;
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (comp / 100) * circumference;

  // Determine color theme
  let scoreColor = '#10b981'; // Green
  if (comp < 60) scoreColor = '#f43f5e'; // Rose
  else if (comp < 75) scoreColor = '#f59e0b'; // Amber
  else if (comp < 85) scoreColor = '#38bdf8'; // Cyan

  const subScoreList = [
    { label: 'Missing Values (25%)', val: score.sub_scores.missingness_score },
    { label: 'Duplication (15%)', val: score.sub_scores.duplicate_score },
    { label: 'Outlier Quality (15%)', val: score.sub_scores.outlier_score },
    { label: 'Domain Validity (15%)', val: score.sub_scores.validity_score },
    { label: 'Target Balance (15%)', val: score.sub_scores.target_balance_score },
    { label: 'Feature Quality (15%)', val: score.sub_scores.feature_quality_score },
  ];

  return (
    <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
            {stageName}
          </span>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Data Health Score</h3>
        </div>
        <span className={`badge ${comp >= 80 ? 'badge-success' : comp >= 60 ? 'badge-high' : 'badge-critical'}`}>
          Grade: {score.grade}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2rem', alignItems: 'center' }}>
        {/* Radial Meter Dial */}
        <div style={{ position: 'relative', width: 170, height: 170, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="170" height="170" style={{ transform: 'rotate(-90deg)' }}>
            {/* Background Track */}
            <circle
              cx="85"
              cy="85"
              r={radius}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeWidth="12"
              fill="transparent"
            />
            {/* Active Gauge */}
            <circle
              cx="85"
              cy="85"
              r={radius}
              stroke={scoreColor}
              strokeWidth="12"
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 1s ease-in-out' }}
            />
          </svg>
          <div style={{ position: 'absolute', textAlign: 'center' }}>
            <span style={{ fontSize: '2.5rem', fontWeight: 800, lineHeight: 1, color: scoreColor }}>
              {comp}
            </span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)', display: 'block', fontWeight: 600 }}>
              / 100
            </span>
          </div>
        </div>

        {/* 6 Sub-Scores Bars */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
          {subScoreList.map((sub, i) => {
            const barColor = sub.val >= 85 ? '#10b981' : sub.val >= 65 ? '#38bdf8' : sub.val >= 50 ? '#f59e0b' : '#f43f5e';
            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{sub.label}</span>
                  <span style={{ fontWeight: 700, color: barColor }}>{sub.val}%</span>
                </div>
                <div style={{ width: '100%', height: 6, background: 'rgba(255, 255, 255, 0.08)', borderRadius: 4, overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${sub.val}%`,
                      height: '100%',
                      background: barColor,
                      borderRadius: 4,
                      transition: 'width 0.8s ease',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Summary Narrative */}
      <div style={{
        padding: '0.85rem 1rem',
        borderRadius: 10,
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.6rem',
        fontSize: '0.85rem',
        color: 'var(--text-muted)',
      }}>
        <Info size={18} color="#38bdf8" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong style={{ color: 'var(--text-main)' }}>Diagnostic Assessment: </strong>
          {score.summary_text}
        </div>
      </div>
    </div>
  );
};
