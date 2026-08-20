import React from 'react';
import { BenchmarkLeaderboard } from '../services/api';
import { Trophy, Zap, Clock, Sparkles, CheckCircle2, ChevronRight } from 'lucide-react';

interface ModelLeaderboardProps {
  leaderboard: BenchmarkLeaderboard;
}

export const ModelLeaderboard: React.FC<ModelLeaderboardProps> = ({ leaderboard }) => {
  const topModel = leaderboard.models[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Model Recommendation Spotlight */}
      {topModel && (
        <div
          className="glass-panel glow-cyan"
          style={{
            padding: '1.75rem',
            background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%)',
            border: '1px solid rgba(56, 189, 248, 0.35)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{
              width: 50,
              height: 50,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(56, 189, 248, 0.4)',
            }}>
              <Trophy size={26} color="#07090e" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                <span className="badge badge-success">Top Candidate Recommendation</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Rank #1</span>
              </div>
              <h3 style={{ fontSize: '1.4rem', fontWeight: 800 }}>
                {topModel.model_name}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: 550 }}>
                {topModel.description}
              </p>
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
              Cross-Validated {leaderboard.primary_metric}
            </span>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#38bdf8', lineHeight: 1.1 }}>
              {topModel.metric_display}
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Fit latency: {topModel.training_time_sec}s
            </span>
          </div>
        </div>
      )}

      {/* Full Leaderboard Table */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Zap size={18} color="#f59e0b" />
            <h4 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              Model Candidate Leaderboard ({leaderboard.problem_type})
            </h4>
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            Evaluated using 3-Fold Cross Validation on Cleaned Data
          </span>
        </div>

        <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid var(--border)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border)' }}>
                <th style={{ padding: '0.75rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)', width: 50 }}>Rank</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Algorithm</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>{leaderboard.primary_metric}</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Fit Time</th>
                <th style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>Suitability</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.models.map((model) => (
                <tr
                  key={model.id}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                    background: model.rank === 1 ? 'rgba(56, 189, 248, 0.05)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.75rem 0.85rem', fontWeight: 700, color: model.rank === 1 ? '#38bdf8' : 'var(--text-dim)' }}>
                    #{model.rank}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span style={{ color: model.rank === 1 ? '#38bdf8' : 'var(--text-main)' }}>{model.model_name}</span>
                      {model.rank === 1 && <span className="badge badge-medium" style={{ fontSize: '0.65rem' }}>Best</span>}
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 700, color: model.rank === 1 ? '#34d399' : 'var(--text-main)', fontFamily: 'monospace' }}>
                    {model.metric_display}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Clock size={12} />
                      <span>{model.training_time_sec}s</span>
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className={`badge ${model.suitability === 'High' ? 'badge-success' : 'badge-low'}`}>
                      {model.suitability}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
