import React from 'react';
import { AugmentManifest } from '../services/api';

interface AugmentExportPanelProps {
  manifest: AugmentManifest | null;
  busy: boolean;
  onTrain: () => void;
  onManifest: () => void;
}

export const AugmentExportPanel: React.FC<AugmentExportPanelProps> = ({ manifest, busy, onTrain, onManifest }) => {
  return (
    <div className="panel panel-pad" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <div className="panel-head" style={{ margin: '-1.5rem -1.5rem 0', padding: '0.9rem 1.1rem' }}>
        <span className="panel-title">Export the training split</span>
      </div>
      <p style={{ margin: 0 }}>
        The holdout stays original. Augmented rows are added only to the training split, and they are not written into the cleaned dataset.
      </p>
      {manifest && (
        <>
          <p className="mono" style={{ margin: 0, fontSize: '0.75rem' }}>
            Train originals {manifest.split.train_original} · synthetic {manifest.split.train_synthetic} · holdout {manifest.split.holdout}
          </p>
          {manifest.warnings.map((warning) => <p key={warning} style={{ margin: 0, color: 'var(--muted)' }}>{warning}</p>)}
        </>
      )}
      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
        <button className="btn btn-dark" type="button" disabled={busy} onClick={onTrain}>Download augmented train</button>
        <button className="btn btn-light" type="button" disabled={busy} onClick={onManifest}>Refresh manifest</button>
      </div>
    </div>
  );
};
