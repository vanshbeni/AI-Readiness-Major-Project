import React, { useState } from 'react';
import { Target, Layers, TrendingUp, Sparkles, ArrowRight, Loader2, Table } from 'lucide-react';
import { DatasetSummary, DatasetSample } from '../services/api';

interface ObjectiveStepProps {
  dataset: DatasetSummary;
  sample: DatasetSample;
  onSubmit: (problemType: string, targetColumn: string) => Promise<void>;
}

export const ObjectiveStep: React.FC<ObjectiveStepProps> = ({ dataset, sample, onSubmit }) => {
  // Infer smart default target column
  const cols = sample.columns;
  const defaultTarget = cols.find(c => ['survived', 'churn', 'target', 'label', 'price', 'medianhousevalue', 'income', 'class', 'status'].includes(c.toLowerCase())) || cols[cols.length - 1];
  
  const [problemType, setProblemType] = useState<string>(
    ['price', 'medianhousevalue', 'value', 'amount', 'salary'].some(k => defaultTarget.toLowerCase().includes(k))
      ? 'regression'
      : 'classification'
  );
  const [targetColumn, setTargetColumn] = useState<string>(defaultTarget);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await onSubmit(problemType, targetColumn);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 850, margin: '1rem auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>
          Define Your <span className="gradient-text">ML Objective</span>
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Select the supervised machine learning problem type and specify which column your models should predict.
        </p>
      </div>

      <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
        {/* Problem Type Cards */}
        <div>
          <label style={{ display: 'block', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.75rem' }}>
            1. Select Problem Type
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div
              onClick={() => setProblemType('classification')}
              style={{
                padding: '1.25rem',
                borderRadius: 12,
                cursor: 'pointer',
                background: problemType === 'classification' ? 'rgba(56, 189, 248, 0.12)' : 'rgba(255, 255, 255, 0.02)',
                border: problemType === 'classification' ? '2px solid #38bdf8' : '1px solid var(--border)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
                <Layers size={20} color={problemType === 'classification' ? '#38bdf8' : 'var(--text-dim)'} />
                <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Classification</h4>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Predicting discrete categories or binary outcomes (e.g. Churn vs Retain, Survived vs Perished).
              </p>
            </div>

            <div
              onClick={() => setProblemType('regression')}
              style={{
                padding: '1.25rem',
                borderRadius: 12,
                cursor: 'pointer',
                background: problemType === 'regression' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(255, 255, 255, 0.02)',
                border: problemType === 'regression' ? '2px solid #10b981' : '1px solid var(--border)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
                <TrendingUp size={20} color={problemType === 'regression' ? '#10b981' : 'var(--text-dim)'} />
                <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Regression</h4>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Predicting continuous numeric quantities (e.g. Housing Price, Temperature, Revenue).
              </p>
            </div>
          </div>
        </div>

        {/* Target Column Selector */}
        <div>
          <label style={{ display: 'block', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.75rem' }}>
            2. Choose Target (Prediction) Column
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                borderRadius: 10,
                background: '#0d131f',
                border: '1px solid var(--border)',
                color: 'var(--text-main)',
                fontSize: '0.95rem',
                fontWeight: 600,
                outline: 'none',
              }}
            >
              {cols.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <span className="badge badge-medium" style={{ height: 'fit-content' }}>
              Target: {targetColumn}
            </span>
          </div>
        </div>

        {/* Dataset Preview Table */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            <Table size={14} />
            <span>Dataset Preview (Top 5 rows of {dataset.row_count} total):</span>
          </div>
          <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid var(--border)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border)' }}>
                  {sample.columns.map((c) => (
                    <th key={c} style={{
                      padding: '0.5rem 0.75rem',
                      fontWeight: 600,
                      color: c === targetColumn ? '#38bdf8' : 'var(--text-muted)',
                      background: c === targetColumn ? 'rgba(56, 189, 248, 0.08)' : 'transparent',
                    }}>
                      {c} {c === targetColumn && '★'}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample.sample_data.slice(0, 5).map((row, rIdx) => (
                  <tr key={rIdx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                    {sample.columns.map((c) => (
                      <td key={c} style={{
                        padding: '0.45rem 0.75rem',
                        color: row[c] === null ? '#fb7185' : 'var(--text-main)',
                        background: c === targetColumn ? 'rgba(56, 189, 248, 0.04)' : 'transparent',
                      }}>
                        {row[c] === null ? '<null>' : String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Run Diagnostic Button */}
        <button
          className="btn-primary"
          onClick={handleSubmit}
          disabled={loading}
          style={{ width: '100%', padding: '0.9rem', fontSize: '1rem' }}
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              <span>Running Deep Profiling & Gemini Diagnostic Engine...</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>Run AI Diagnostic & Compute Health Score</span>
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
    </div>
  );
};
