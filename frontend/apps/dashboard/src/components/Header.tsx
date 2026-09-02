import React from 'react';
import { ShieldCheck, RefreshCw, Database } from 'lucide-react';

interface HeaderProps {
  datasetName?: string;
  onReset?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ datasetName, onReset }) => {
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '1.25rem 2rem',
      borderBottom: '1px solid var(--border)',
      background: 'rgba(7, 9, 14, 0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
        <div style={{
          width: 38,
          height: 38,
          borderRadius: 10,
          background: 'linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(56, 189, 248, 0.4)',
        }}>
          <ShieldCheck size={22} color="#07090e" strokeWidth={2.5} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h1 style={{ fontSize: '1.2rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              AI Data Readiness Platform
            </h1>
            <span style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              padding: '0.15rem 0.45rem',
              borderRadius: 4,
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
            }}>
              v1.0
            </span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Explainable Pre-ML Diagnostic & Remediation Engine
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {datasetName && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.35rem 0.85rem',
            borderRadius: 8,
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--border)',
            fontSize: '0.825rem',
            color: 'var(--text-muted)',
          }}>
            <Database size={14} color="#38bdf8" />
            <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{datasetName}</span>
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: '#10b981' }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', display: 'inline-block', boxShadow: '0 0 8px #10b981' }} />
          <span>Engine Active</span>
        </div>

        {onReset && datasetName && (
          <button
            onClick={onReset}
            className="btn-secondary"
            style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
            title="Upload new dataset"
          >
            <RefreshCw size={14} />
            <span>New Dataset</span>
          </button>
        )}
      </div>
    </header>
  );
};
