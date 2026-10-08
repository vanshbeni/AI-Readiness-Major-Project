import React, { useState, useRef } from 'react';
import { ArrowUpRight, Loader2, UploadCloud } from 'lucide-react';
import { api, DatasetSummary, DatasetSample, MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS } from '../services/api';
import { Banner, Eyebrow, Note, Panel, Reveal, Scramble, Underlined, trackPointer } from './ui';

interface UploadStepProps {
  onDatasetLoaded: (dataset: DatasetSummary, sample: DatasetSample) => void;
}

const DEMOS = [
  {
    key: 'titanic',
    file: 'titanic.csv',
    tag: 'classification',
    tone: 'accent',
    title: 'Titanic passenger survival',
    body: 'Missing age and cabin, invalid negative ages, fare outliers, duplicates, ID and name columns.',
    chips: ['nulls', 'outliers', 'dupes'],
  },
  {
    key: 'churn',
    file: 'churn.csv',
    tag: 'imbalance',
    tone: 'warn',
    title: 'Telco customer churn',
    body: 'Roughly 89:11 class imbalance, missing charges, collinear billing features.',
    chips: ['imbalance', 'collinear', 'nulls'],
  },
  {
    key: 'housing',
    file: 'housing.csv',
    tag: 'regression',
    tone: 'good',
    title: 'California housing prices',
    body: 'Continuous target, collinear room features, outliers, duplicate records.',
    chips: ['skew', 'collinear', 'dupes'],
  },
] as const;

export const UploadStep: React.FC<UploadStepProps> = ({ onDatasetLoaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) return `Unsupported file type '${ext}'. Upload a CSV, XLSX or XLS file.`;
    if (file.size === 0) return 'The selected file is empty.';
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `File is larger than the ${MAX_UPLOAD_MB} MB limit.`;
    return null;
  };

  const handleFileUpload = async (file: File) => {
    if (loading) return;
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    try {
      setLoading(`uploading & parsing ${file.name}`);
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
    if (loading) return;
    try {
      setLoading(`loading ${demoKey} demo dataset`);
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
    <div className="step-enter" style={{ maxWidth: 920, margin: '0 auto' }}>
      <div className="step-head center" style={{ marginTop: '3.5rem' }}>
        <Eyebrow label="ingest" index="01/05" caret />
        <h1 className="h-title">
          <Scramble text="Drop a messy" /> <Underlined>dataset.</Underlined>
        </h1>
        <p className="h-sub">
          CSV or Excel. We profile every column, score its health out of 100 and draft fixes you approve one by one.
        </p>
      </div>

      {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}

      <div style={{ position: 'relative' }}>
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''} ${loading ? 'loading' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="Upload a CSV or Excel file"
          onPointerMove={trackPointer}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !loading) {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          onDragOver={(e) => { e.preventDefault(); if (!loading) setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
          }}
          onClick={() => !loading && fileInputRef.current?.click()}
        >
          <svg className="border" aria-hidden>
            <rect x="1" y="1" rx="15" ry="15" style={{ width: 'calc(100% - 2px)', height: 'calc(100% - 2px)' }} />
          </svg>
          {loading && <span className="drop-scan" aria-hidden />}

          <input
            type="file"
            ref={fileInputRef}
            accept=".csv,.xlsx,.xls"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = '';
              if (file) handleFileUpload(file);
            }}
          />

          <div className="drop-icon">
            {loading ? <Loader2 size={28} className="animate-spin" /> : <UploadCloud size={28} />}
          </div>

          <p style={{ fontSize: '1.2rem', fontWeight: 600, letterSpacing: '-0.03em', marginBottom: '0.4rem' }}>
            {loading ? <span className="mono" style={{ fontSize: '0.95rem' }}>{loading}<span className="caret" style={{ marginLeft: 4 }} /></span>
              : isDragging ? 'Let go, we got it.' : 'Drag & drop your file here'}
          </p>
          <p className="mono" style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: '1.4rem' }}>
            .csv · .xlsx · .xls &nbsp;/&nbsp; up to {MAX_UPLOAD_MB} MB &nbsp;/&nbsp; delimiter, encoding & types auto-detected
          </p>

          <span className="btn btn-dark" aria-hidden style={{ pointerEvents: 'none', opacity: loading ? 0.45 : 1 }}>
            Browse files <ArrowUpRight size={15} className="arrow" />
          </span>
        </div>

        <Note className="float-note" tone="purple" arrow="left" rot={-7} delay={1.2} style={{ position: 'absolute', right: '-2.5rem', top: '-1.6rem' }}>
          your file stays yours
        </Note>
      </div>

      <div style={{ marginTop: '3.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', gap: '1rem', flexWrap: 'wrap' }}>
          <Eyebrow label="or try a demo" />
          <Note rot={3} delay={1.5}>pre-broken on purpose</Note>
        </div>
        <div className="demo-grid">
          {DEMOS.map((d, i) => (
            <Reveal key={d.key} delay={i * 90}>
              <Panel
                as="button"
                type="button"
                tilt
                lift
                className="demo-tile"
                disabled={!!loading}
                onClick={() => !loading && handleDemoLoad(d.key)}
                style={{ width: '100%', height: '100%' }}
              >
                <span className="go">
                  <span>~/{d.file}</span>
                  <ArrowUpRight size={14} />
                </span>
                <span className={`chip ${d.tone}`} style={{ alignSelf: 'flex-start' }}>{d.tag}</span>
                <h4>{d.title}</h4>
                <p>~300 rows · {d.body}</p>
                <span className="chips">
                  {d.chips.map((c) => <span key={c} className="chip">{c}</span>)}
                </span>
              </Panel>
            </Reveal>
          ))}
        </div>
      </div>
    </div>
  );
};
