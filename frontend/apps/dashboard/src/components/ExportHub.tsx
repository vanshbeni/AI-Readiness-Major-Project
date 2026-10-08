import React, { useState } from 'react';
import { api } from '../services/api';
import { ArrowDown, Loader2 } from 'lucide-react';
import { Banner, Eyebrow, Panel, Reveal } from './ui';

interface ExportHubProps {
  datasetId: string;
  hasExecution: boolean;
  hasBenchmarks: boolean;
}

type ExportKind = 'cleaned-csv' | 'pipeline-script' | 'pdf-report' | 'manifest-json';

const EXPORTS: {
  kind: ExportKind;
  ext: string;
  title: string;
  description: string;
  fallbackName: string;
  color: string;
  requiresExecution: boolean;
}[] = [
  { kind: 'cleaned-csv', ext: '.csv', title: 'Cleaned dataset', description: 'Preprocessed, ML-ready CSV.', fallbackName: 'cleaned_dataset.csv', color: 'var(--accent)', requiresExecution: true },
  { kind: 'pipeline-script', ext: '.py', title: 'Pipeline script', description: 'Standalone Python that reproduces the cleaned CSV.', fallbackName: 'pipeline.py', color: 'var(--good)', requiresExecution: true },
  { kind: 'pdf-report', ext: '.pdf', title: 'Quality audit', description: 'Executive PDF diagnostic report.', fallbackName: 'data_readiness_report.pdf', color: 'var(--bad)', requiresExecution: false },
  { kind: 'manifest-json', ext: '.json', title: 'JSON manifest', description: 'Machine-readable audit manifest.', fallbackName: 'manifest.json', color: 'var(--ink)', requiresExecution: false },
];

export const ExportHub: React.FC<ExportHubProps> = ({ datasetId, hasExecution, hasBenchmarks }) => {
  const [busy, setBusy] = useState<ExportKind | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async (kind: ExportKind, fallbackName: string) => {
    setBusy(kind);
    setError(null);
    try {
      await api.downloadExport(datasetId, kind, fallbackName);
    } catch (err: any) {
      setError(err.message || 'Download failed');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div style={{ marginTop: '1.5rem' }}>
      <div className="step-head-row" style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem' }}>
          <Eyebrow label="export" />
          <h2 style={{ fontSize: '1.8rem', letterSpacing: '-0.045em', fontWeight: 600, lineHeight: 1.05 }}>Take it home.</h2>
        </div>
        <p className="h-sub" style={{ fontSize: '0.86rem', maxWidth: 380 }}>
          Audited files and reports, ready for your notebook, repo or report.
          {!hasBenchmarks && ' Reports generated now will not include model benchmarks.'}
        </p>
      </div>

      {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}

      <div className="export-grid">
        {EXPORTS.map((item, i) => {
          const locked = item.requiresExecution && !hasExecution;
          const disabled = locked || busy !== null;
          return (
            <Reveal key={item.kind} delay={i * 80}>
              <Panel
                as="button"
                type="button"
                tilt
                lift
                className="export-tile"
                onClick={() => handleDownload(item.kind, item.fallbackName)}
                disabled={disabled}
                title={locked ? 'Execute the cleaning pipeline first' : undefined}
                style={{ width: '100%', height: '100%', opacity: locked ? 0.5 : 1 }}
              >
                <span className="ext" style={{ color: item.color }}>{item.ext}</span>
                <span>
                  <h4>{item.title}</h4>
                  <p>{item.description}</p>
                </span>
                <span className="go">
                  {busy === item.kind ? <Loader2 size={12} className="animate-spin" /> : <ArrowDown size={12} />}
                  {locked ? 'run pipeline first' : busy === item.kind ? 'preparing…' : `download ${item.fallbackName}`}
                </span>
              </Panel>
            </Reveal>
          );
        })}
      </div>
    </div>
  );
};
