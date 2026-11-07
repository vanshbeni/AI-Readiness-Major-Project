'use client';

import React, { useEffect, useState } from 'react';
import { Header } from '../../components/Header';
import { Banner, Eyebrow, Scramble, SkyBand, Underlined } from '../../components/ui';
import { AugmentConfigPanel } from '../../components/AugmentConfigPanel';
import { AugmentPreview } from '../../components/AugmentPreview';
import { AugmentResultCard } from '../../components/AugmentResultCard';
import { AugmentExportPanel } from '../../components/AugmentExportPanel';
import { api, AugmentConfig, AugmentManifest, AugmentPreview as Preview, AugmentResult, AugmentSuggestions } from '../../services/api';

const DATASET_PARAM = 'dataset';

function defaultConfig(modality: 'text' | 'image', textColumn: string | null): AugmentConfig {
  return {
    modality,
    text_column: textColumn,
    strategy: 'balance',
    n_variants: 2,
    ops: modality === 'text'
      ? [
          { name: 'swap', alpha: 0.1 },
          { name: 'deletion', alpha: 0.1 },
          { name: 'insertion', alpha: 0.1 },
        ]
      : [],
    seed: 42,
    horizontal_flip: true,
    vertical_flip: false,
    max_deg: 15,
    brightness: 0.15,
    contrast: 0.15,
    noise_std: 6,
    crop_scale: 0.85,
  };
}

export default function AugmentPage() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<AugmentSuggestions | null>(null);
  const [config, setConfig] = useState<AugmentConfig | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [result, setResult] = useState<AugmentResult | null>(null);
  const [manifest, setManifest] = useState<AugmentManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get(DATASET_PARAM);
    setDatasetId(id);
    if (!id) {
      setLoading(false);
      setError('No dataset selected. Open augmentation from the dashboard after loading a dataset.');
      return;
    }
    let active = true;
    api.augmentSuggestions(id)
      .then((payload) => {
        if (!active) return;
        setSuggestions(payload);
        const modality = payload.modality === 'image' ? 'image' : 'text';
        const column = payload.text_columns[0]?.name ?? null;
        const saved = payload.latest_config;
        setConfig(saved && saved.modality === modality ? { ...defaultConfig(modality, column), ...saved } : defaultConfig(modality, column));
        if (payload.latest_result) setResult(payload.latest_result);
      })
      .catch((err: Error) => active && setError(err.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, []);

  const runPreview = async () => {
    if (!datasetId || !config) return;
    setBusy('preview');
    setError(null);
    try {
      setPreview(await api.augmentPreview(datasetId, config, 4));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Preview failed');
    } finally {
      setBusy(null);
    }
  };

  const runBenchmark = async () => {
    if (!datasetId || !config) return;
    setBusy('run');
    setError(null);
    try {
      setResult(await api.augmentRun(datasetId, config));
      setManifest(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed');
    } finally {
      setBusy(null);
    }
  };

  const loadManifest = async () => {
    if (!datasetId) return;
    setBusy('manifest');
    setError(null);
    try {
      setManifest(await api.augmentManifest(datasetId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Manifest failed');
    } finally {
      setBusy(null);
    }
  };

  const downloadTrain = async () => {
    if (!datasetId) return;
    setBusy('download');
    setError(null);
    try {
      await api.downloadAugmentedTrain(datasetId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="app-shell">
      <div className="grain" aria-hidden />
      <SkyBand />
      <Header datasetName={suggestions?.target_column ? `target ${suggestions.target_column}` : undefined} />
      <main className="app-main">
        <div className="frame" style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          <div className="step-head">
            <Eyebrow label="augmentation" />
            <h1 className="h-title"><Scramble text="More training rows." /> <Underlined>Same holdout.</Underlined></h1>
            <p className="h-sub">Augmentation is applied only to training folds. Validation rows stay exactly as uploaded.</p>
          </div>
          <p><a href={datasetId ? `/?dataset=${datasetId}` : '/'}>Back to the dataset</a></p>
          {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}
          {loading && <p className="mono">Loading suggestions…</p>}
          {suggestions && config && (
            <AugmentConfigPanel
              config={config}
              suggestions={suggestions}
              busy={!!busy}
              onChange={setConfig}
              onPreview={runPreview}
              onRun={runBenchmark}
            />
          )}
          {preview && <AugmentPreview samples={preview.samples} modality={preview.modality} />}
          {result && <AugmentResultCard result={result} />}
          {result && (
            <AugmentExportPanel
              manifest={manifest}
              busy={!!busy}
              onTrain={downloadTrain}
              onManifest={loadManifest}
            />
          )}
          {busy && <p className="mono">{busy === 'run' ? 'Fitting both arms on the same folds…' : 'Working…'}</p>}
        </div>
      </main>
    </div>
  );
};
