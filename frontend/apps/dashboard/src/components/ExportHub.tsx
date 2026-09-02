import React from 'react';
import { api } from '../services/api';
import { Download, FileSpreadsheet, FileCode, FileText, Code2 } from 'lucide-react';

interface ExportHubProps {
  datasetId: string;
}

export const ExportHub: React.FC<ExportHubProps> = ({ datasetId }) => {
  return (
    <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Download size={20} color="#38bdf8" />
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800 }}>
            Download & Export Hub
          </h3>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Export production-ready artifacts, audited dataset files, and executive reports.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        {/* Cleaned CSV */}
        <a
          href={api.getCleanedCsvUrl(datasetId)}
          download
          className="glass-panel"
          style={{
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
            transition: 'all 0.2s ease',
            border: '1px solid var(--border)',
          }}
        >
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'rgba(56, 189, 248, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8' }}>
            <FileSpreadsheet size={18} />
          </div>
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Cleaned Dataset</h4>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Preprocessed, ML-ready CSV format</p>
          </div>
          <span style={{ fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600, marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Download size={13} /> Download .CSV
          </span>
        </a>

        {/* Python Pipeline Script */}
        <a
          href={api.getPipelineScriptUrl(datasetId)}
          download
          className="glass-panel"
          style={{
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
            transition: 'all 0.2s ease',
            border: '1px solid var(--border)',
          }}
        >
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'rgba(168, 85, 247, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#10b981' }}>
            <FileCode size={18} />
          </div>
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Pipeline Script</h4>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Standalone Python & scikit-learn code</p>
          </div>
          <span style={{ fontSize: '0.8rem', color: '#10b981', fontWeight: 600, marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Download size={13} /> Download .PY
          </span>
        </a>

        {/* PDF Audit Report */}
        <a
          href={api.getPdfReportUrl(datasetId)}
          target="_blank"
          rel="noopener noreferrer"
          className="glass-panel"
          style={{
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
            transition: 'all 0.2s ease',
            border: '1px solid var(--border)',
          }}
        >
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'rgba(244, 63, 94, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#f43f5e' }}>
            <FileText size={18} />
          </div>
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Data Quality Audit</h4>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Executive PDF diagnostic report</p>
          </div>
          <span style={{ fontSize: '0.8rem', color: '#fb7185', fontWeight: 600, marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Download size={13} /> Download .PDF
          </span>
        </a>

        {/* JSON Manifest */}
        <a
          href={api.getManifestJsonUrl(datasetId)}
          download
          className="glass-panel"
          style={{
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
            transition: 'all 0.2s ease',
            border: '1px solid var(--border)',
          }}
        >
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'rgba(168, 85, 247, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a855f7' }}>
            <Code2 size={18} />
          </div>
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>JSON Manifest</h4>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Machine-readable audit manifest</p>
          </div>
          <span style={{ fontSize: '0.8rem', color: '#c084fc', fontWeight: 600, marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Download size={13} /> Download .JSON
          </span>
        </a>
      </div>
    </div>
  );
};
