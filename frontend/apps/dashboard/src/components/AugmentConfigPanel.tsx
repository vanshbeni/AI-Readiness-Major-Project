import React from 'react';
import { AugmentConfig, AugmentOpName, AugmentSuggestions } from '../services/api';

const TEXT_OPS: { name: AugmentOpName; label: string }[] = [
  { name: 'swap', label: 'Swap words' },
  { name: 'deletion', label: 'Delete words' },
  { name: 'insertion', label: 'Insert a word from the same text' },
  { name: 'sentence_shuffle', label: 'Shuffle sentences' },
];

interface AugmentConfigPanelProps {
  config: AugmentConfig;
  suggestions: AugmentSuggestions;
  busy: boolean;
  onChange: (config: AugmentConfig) => void;
  onPreview: () => void;
  onRun: () => void;
}

export const AugmentConfigPanel: React.FC<AugmentConfigPanelProps> = ({ config, suggestions, busy, onChange, onPreview, onRun }) => {
  const image = config.modality === 'image';
  const set = (patch: Partial<AugmentConfig>) => onChange({ ...config, ...patch });
  const classification = suggestions.problem_type === 'classification';

  return (
    <div className="panel panel-pad" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div className="panel-head" style={{ margin: '-1.5rem -1.5rem 0', padding: '0.9rem 1.1rem' }}>
        <div className="left"><span className="panel-title">Configuration</span></div>
        <span className="panel-idx">seed {config.seed}</span>
      </div>

      {suggestions.suggestions.map((item) => (
        <p key={item.id} style={{ margin: 0, color: 'var(--ink-2)', fontSize: '0.92rem' }}>
          <strong>{item.title}. </strong>{item.detail}
        </p>
      ))}

      <label className="ctl">
        <span className="ctl-label">Strategy</span>
        <select className="select sm" value={config.strategy} onChange={(e) => set({ strategy: e.target.value as AugmentConfig['strategy'] })}>
          <option value="balance">Balance minority classes</option>
          <option value="multiply">Fixed variants per row</option>
        </select>
      </label>

      <label className="ctl">
        <span className="ctl-label">Variants per row (max 10, and 5 when balancing)</span>
        <input className="select sm" type="number" min={0} max={10} value={config.n_variants} onChange={(e) => set({ n_variants: Number(e.target.value) })} />
      </label>

      <label className="ctl">
        <span className="ctl-label">Seed</span>
        <input className="select sm" type="number" min={0} value={config.seed} onChange={(e) => set({ seed: Number(e.target.value) })} />
      </label>

      {!image && (
        <>
          <label className="ctl">
            <span className="ctl-label">Text column</span>
            <select className="select sm" value={config.text_column || ''} onChange={(e) => set({ text_column: e.target.value })}>
              <option value="">Choose a column</option>
              {suggestions.text_columns.map((column) => (
                <option key={column.name} value={column.name}>{column.name}</option>
              ))}
            </select>
          </label>
          {TEXT_OPS.map((op) => {
            const current = config.ops.find((item) => item.name === op.name);
            return (
              <label key={op.name} className="ctl" style={{ display: 'grid', gridTemplateColumns: '1.4rem 1fr 5rem', gap: '0.6rem', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={!!current}
                  onChange={(e) => {
                    const ops = e.target.checked
                      ? [...config.ops, { name: op.name, alpha: current?.alpha ?? 0.1 }]
                      : config.ops.filter((item) => item.name !== op.name);
                    set({ ops });
                  }}
                />
                <span>{op.label}</span>
                <input
                  className="select sm"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={!current}
                  value={current?.alpha ?? 0.1}
                  onChange={(e) => set({ ops: config.ops.map((item) => item.name === op.name ? { ...item, alpha: Number(e.target.value) } : item) })}
                />
              </label>
            );
          })}
        </>
      )}

      {image && (
        <div style={{ display: 'grid', gap: '0.7rem' }}>
          <label><input type="checkbox" checked={config.horizontal_flip} onChange={(e) => set({ horizontal_flip: e.target.checked })} /> Horizontal flip</label>
          <label><input type="checkbox" checked={config.vertical_flip} onChange={(e) => set({ vertical_flip: e.target.checked })} /> Vertical flip</label>
          <label className="ctl"><span className="ctl-label">Max rotation (degrees)</span>
            <input className="select sm" type="number" min={0} max={45} value={config.max_deg} onChange={(e) => set({ max_deg: Number(e.target.value) })} />
          </label>
          <label className="ctl"><span className="ctl-label">Brightness jitter</span>
            <input className="select sm" type="number" min={0} max={0.5} step={0.05} value={config.brightness} onChange={(e) => set({ brightness: Number(e.target.value) })} />
          </label>
          <label className="ctl"><span className="ctl-label">Contrast jitter</span>
            <input className="select sm" type="number" min={0} max={0.5} step={0.05} value={config.contrast} onChange={(e) => set({ contrast: Number(e.target.value) })} />
          </label>
          <label className="ctl"><span className="ctl-label">Noise standard deviation</span>
            <input className="select sm" type="number" min={0} max={30} value={config.noise_std} onChange={(e) => set({ noise_std: Number(e.target.value) })} />
          </label>
          <label className="ctl"><span className="ctl-label">Crop scale</span>
            <input className="select sm" type="number" min={0.5} max={1} step={0.05} value={config.crop_scale} onChange={(e) => set({ crop_scale: Number(e.target.value) })} />
          </label>
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
        <button className="btn btn-light" type="button" disabled={busy || !classification} onClick={onPreview}>Preview</button>
        <button className="btn btn-dark" type="button" disabled={busy || !classification} onClick={onRun}>Benchmark baseline vs augmented</button>
      </div>
      {!classification && (
        <p style={{ margin: 0, color: 'var(--muted)' }}>
          {suggestions.problem_type
            ? 'This objective is not classification. Augmentation is only benchmarked for classification.'
            : 'Set a classification objective before running augmentation.'}
        </p>
      )}
    </div>
  );
};
