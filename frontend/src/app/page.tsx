import React from 'react';
import { Activity, ShieldCheck, Sparkles, Database, FileSpreadsheet, ArrowRight } from 'lucide-react';

export default function HomePage() {
  return (
    <main style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}>
      <div style={{ maxWidth: '800px', width: '100%', textAlign: 'center' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 1rem', borderRadius: '9999px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.3)', color: '#38bdf8', fontSize: '0.875rem', fontWeight: 500, marginBottom: '1.5rem' }}>
          <Sparkles size={16} />
          <span>Explainable Pre-ML Diagnostic Platform</span>
        </div>

        <h1 style={{ fontSize: '3rem', fontWeight: 800, lineHeight: 1.15, marginBottom: '1rem', background: 'linear-gradient(135deg, #ffffff 0%, #94a3b8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          AI Data Readiness Platform
        </h1>

        <p style={{ fontSize: '1.125rem', color: '#94a3b8', lineHeight: 1.6, marginBottom: '2.5rem' }}>
          Before asking <em>"Which model should I train?"</em>, ask <em>"Is my data ready for ML?"</em>. Profile quality, get explainable fix recommendations, and verify dataset health.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2.5rem', textAlign: 'left' }}>
          <div style={{ padding: '1.25rem', borderRadius: '12px', background: 'rgba(20, 27, 41, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
            <Activity size={24} color="#38bdf8" style={{ marginBottom: '0.75rem' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '0.25rem' }}>Data Health Score</h3>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Composite 0–100 health metrics based on statistical profiling.</p>
          </div>

          <div style={{ padding: '1.25rem', borderRadius: '12px', background: 'rgba(20, 27, 41, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
            <ShieldCheck size={24} color="#10b981" style={{ marginBottom: '0.75rem' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '0.25rem' }}>Explainable Fixes</h3>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>AI-grounded remediation reasoning with human-in-the-loop approvals.</p>
          </div>

          <div style={{ padding: '1.25rem', borderRadius: '12px', background: 'rgba(20, 27, 41, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
            <Database size={24} color="#8b5cf6" style={{ marginBottom: '0.75rem' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '0.25rem' }}>Model Benchmark</h3>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Candidate model ranking and downloadable scikit-learn pipelines.</p>
          </div>
        </div>

        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1.5rem', borderRadius: '8px', background: '#38bdf8', color: '#0a0d14', fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer' }}>
          <FileSpreadsheet size={18} />
          <span>Upload Dataset to Diagnose</span>
          <ArrowRight size={16} />
        </div>
      </div>
    </main>
  );
}
