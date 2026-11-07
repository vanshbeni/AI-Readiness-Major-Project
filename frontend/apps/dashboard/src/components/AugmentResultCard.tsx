import React from 'react';
import { AugmentResult } from '../services/api';

interface AugmentResultCardProps {
  result: AugmentResult;
}

function fmt(value: number): string {
  return value.toFixed(3);
}

export const AugmentResultCard: React.FC<AugmentResultCardProps> = ({ result }) => {
  return (
    <div className="panel panel-pad" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <div className="panel-head" style={{ margin: '-1.5rem -1.5rem 0', padding: '0.9rem 1.1rem' }}>
        <span className="panel-title">{result.verdict}</span>
        <span className="panel-idx">{result.metric}</span>
      </div>
      <p style={{ margin: 0 }}>{result.verdict_detail}</p>
      <p className="mono" style={{ margin: 0, fontSize: '0.75rem', color: 'var(--muted)' }}>
        {result.evaluation_note} Rows used: {result.rows_used} of {result.rows_available}. Synthetic rows per training fold: {result.synthetic_rows}. Seed {result.seed}. {result.folds}-fold macro F1.
      </p>
      {result.warnings.map((warning) => <p key={warning} style={{ margin: 0, color: 'var(--muted)' }}>{warning}</p>)}
      <div className="table-wrap">
        <table className="mono-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Baseline</th>
              <th>Augmented</th>
              <th>Delta</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {result.models.map((model) => (
              <tr key={model.model}>
                <td>{model.model}{model.is_baseline ? ' (dummy)' : ''}</td>
                <td>{fmt(model.baseline_mean)} ± {fmt(model.baseline_std)}</td>
                <td>{fmt(model.augmented_mean)} ± {fmt(model.augmented_std)}</td>
                <td>{model.delta >= 0 ? '+' : ''}{fmt(model.delta)}</td>
                <td>{model.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
