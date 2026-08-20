import React, { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Sparkles, Database, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react';
import { api, DatasetSummary, DatasetSample } from '../services/api';

interface UploadStepProps {
  onDatasetLoaded: (dataset: DatasetSummary, sample: DatasetSample) => void;
}

export const UploadStep: React.FC<UploadStepProps> = ({ onDatasetLoaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    try {
      setLoading('Uploading & parsing dataset...');
      setError(null);
      const dataset = await api.uploadDataset(file);
      const sample = await api.getSample(dataset.id);
      onDatasetLoaded(dataset, sample);
    } catch (err: any) {
      setError(err.message || 'Failed to upload dataset');
    } finally {
      setLoading(null);
    }
  };

  const handleDemoLoad = async (demoKey: string) => {
    try {
      setLoading(`Loading ${demoKey} demo dataset...`);
      setError(null);
      const dataset = await api.loadDemoDataset(demoKey);
      const sample = await api.getSample(dataset.id);
      onDatasetLoaded(dataset, sample);
    } catch (err: any) {
      setError(err.message || 'Failed to load demo dataset');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div style={{ maxWidth: 850, margin: '1rem auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>
          Upload Your <span className="gradient-text">Tabular Dataset</span>
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Upload any CSV or Excel file to profile quality defects, calculate Health Score, and generate AI remediation fixes.
        </p>
      </div>

      {error && (
        <div style={{
          padding: '0.85rem 1.25rem',
          borderRadius: 10,
          background: 'rgba(244, 63, 94, 0.12)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          color: '#fb7185',
          fontSize: '0.875rem',
        }}>
          {error}
        </div>
      )}

      {/* Main Drag and Drop Area */}
      <div
        className="glass-panel"
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
        }}
        onClick={() => fileInputRef.current?.click()}
        style={{
          padding: '3.5rem 2rem',
          textAlign: 'center',
          cursor: 'pointer',
          borderColor: isDragging ? '#38bdf8' : 'var(--border)',
          background: isDragging ? 'rgba(56, 189, 248, 0.05)' : 'var(--bg-card)',
          transition: 'all 0.2s ease',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
        }}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept=".csv,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
          }}
        />

        <div style={{
          width: 64,
          height: 64,
          borderRadius: '50%',
          background: 'rgba(56, 189, 248, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#38bdf8',
        }}>
          {loading ? <Loader2 size={32} className="animate-spin" /> : <UploadCloud size={32} />}
        </div>

        <div>
          <p style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.35rem' }}>
            {loading ? loading : 'Drag & drop your CSV or Excel file here'}
          </p>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>
            Supports CSV, XLSX up to 200 MB with automatic data type inference
          </p>
        </div>

        <button className="btn-primary" disabled={!!loading} style={{ pointerEvents: 'none' }}>
          <FileSpreadsheet size={16} />
          <span>Browse File</span>
        </button>
      </div>

      {/* Quick Curated Demo Datasets */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>
          <Sparkles size={16} color="#38bdf8" />
          <span>Or test with a pre-configured capstone benchmark dataset:</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
          <div
            className="glass-panel"
            onClick={() => !loading && handleDemoLoad('titanic')}
            style={{
              padding: '1.25rem',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-medium">Classification</span>
              <ArrowRight size={14} color="var(--text-dim)" />
            </div>
            <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Titanic Passenger Survival</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
              100 rows • Missing age/cabin, negative values, fare outliers, categorical encoding.
            </p>
          </div>

          <div
            className="glass-panel"
            onClick={() => !loading && handleDemoLoad('churn')}
            style={{
              padding: '1.25rem',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-high">Imbalance</span>
              <ArrowRight size={14} color="var(--text-dim)" />
            </div>
            <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Telco Customer Churn</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
              100 rows • Class imbalance ratio (80:20), missing charges, categoricals.
            </p>
          </div>

          <div
            className="glass-panel"
            onClick={() => !loading && handleDemoLoad('housing')}
            style={{
              padding: '1.25rem',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-success">Regression</span>
              <ArrowRight size={14} color="var(--text-dim)" />
            </div>
            <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>California Housing Prices</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
              100 rows • Continuous target, multi-feature collinearity, non-normal skew.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
