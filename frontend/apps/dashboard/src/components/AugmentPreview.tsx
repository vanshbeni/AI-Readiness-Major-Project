import React from 'react';
import { AugmentPreviewSample } from '../services/api';

interface AugmentPreviewProps {
  samples: AugmentPreviewSample[];
  modality: string;
}

function Thumb({ png, alt }: { png?: string | null; alt: string }) {
  if (!png) return <p style={{ color: 'var(--muted)', margin: 0 }}>No image</p>;
  return <img src={`data:image/png;base64,${png}`} alt={alt} style={{ maxWidth: '100%', height: 'auto', borderRadius: 8 }} />;
}

export const AugmentPreview: React.FC<AugmentPreviewProps> = ({ samples, modality }) => {
  if (!samples.length) return null;
  return (
    <div className="panel panel-pad" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div className="panel-head" style={{ margin: '-1.5rem -1.5rem 0', padding: '0.9rem 1.1rem' }}>
        <span className="panel-title">Original vs augmented</span>
      </div>
      {samples.map((sample) => (
        <div key={sample.source_row_id} className="augment-pair">
          <div>
            <p className="mono" style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>original · row {sample.source_row_id} · {sample.label}</p>
            {modality === 'image' ? <Thumb png={sample.original_png} alt="Original image" /> : <p style={{ margin: 0 }}>{sample.original || '—'}</p>}
          </div>
          <div>
            <p className="mono" style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>augmented{sample.aug_ops ? ` · ${sample.aug_ops}` : ''}</p>
            {modality === 'image'
              ? <Thumb png={sample.augmented_png} alt="Augmented image" />
              : <p style={{ margin: 0 }}>{sample.augmented || sample.note || '—'}</p>}
            {sample.note && modality === 'image' && <p style={{ color: 'var(--muted)' }}>{sample.note}</p>}
          </div>
        </div>
      ))}
    </div>
  );
};
