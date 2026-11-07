import React, { useEffect, useState } from 'react';

interface TerminalProps {
  title: string;
  command: string;
  lines: string[];
  /** Milliseconds between log lines. The last line keeps "running" until unmounted. */
  stepMs?: number;
  footer?: React.ReactNode;
}

/** Fake shell log that prints one line at a time while a long request is in flight. */
export function Terminal({ title, command, lines, stepMs = 1400, footer }: TerminalProps) {
  const [shown, setShown] = useState(1);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const lineTimer = setInterval(() => setShown((n) => Math.min(n + 1, lines.length)), stepMs);
    const clock = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => {
      clearInterval(lineTimer);
      clearInterval(clock);
    };
  }, [lines.length, stepMs]);

  return (
    <div className="terminal" role="status" aria-live="polite">
      <div className="term-top">
        <span className="dots"><i /><i /><i /></span>
        <span>{title}</span>
        <span>{elapsed.toFixed(1)}s</span>
      </div>
      <div className="term-body">
        <div className="term-line">
          <span className="p">$</span>
          <span><span className="k">nebulus</span> {command}</span>
        </div>
        {lines.slice(0, shown).map((line, i) => {
          const running = i === shown - 1;
          return (
            <div key={i} className="term-line">
              <span className={running ? 'run' : 'ok'}>{running ? '…' : '✓'}</span>
              <span style={{ color: running ? '#e7e5e0' : '#8a8781' }}>{line}</span>
              {running && <span className="term-caret" />}
            </div>
          );
        })}
        {footer}
      </div>
    </div>
  );
}

const CLASSIFICATION_MODELS = ['random forest', 'xgboost', 'lightgbm', 'logistic regression', 'gradient boosting', 'decision tree', 'svc'];
const REGRESSION_MODELS = ['random forest', 'xgboost', 'lightgbm', 'ridge regression', 'gradient boosting', 'decision tree', 'svr'];

const CLUSTERING_MODELS = ['k-means', 'gaussian mixture', 'agglomerative (ward)', 'birch', 'dbscan', 'hdbscan'];
const ANOMALY_MODELS = ['isolation forest', 'local outlier factor', 'one-class svm', 'robust z-score', 'elliptic envelope'];

export function BenchmarkTerminal({ problemType }: { problemType?: string }) {
  let lines: string[];
  if (problemType === 'clustering') {
    lines = [
      'loading cleaned dataset',
      'encoding + scaling every usable feature',
      ...CLUSTERING_MODELS.map((m) => `fitting ${m}  (choosing k by silhouette / BIC)`),
      'shuffling columns to build a no-structure reference',
      'profiling the winning clusters',
    ];
  } else if (problemType === 'anomaly_detection') {
    lines = [
      'loading cleaned dataset',
      'estimating contamination from robust z-scores',
      ...ANOMALY_MODELS.map((m) => `scoring rows with ${m}`),
      'voting: rows most detectors agree on',
      'explaining the top anomalies',
    ];
  } else {
    const models = problemType === 'regression' ? REGRESSION_MODELS : CLASSIFICATION_MODELS;
    lines = [
      'loading cleaned dataset',
      'building leak-free folds  (impute → scale → resample → fit)',
      ...models.map((m) => `cross-validating ${m}`),
      'ranking candidates by primary metric',
    ];
  }
  return <Terminal title="benchmark.log" command={`benchmark --cv auto --task ${problemType ?? 'auto'}`} lines={lines} stepMs={1800} />;
}
