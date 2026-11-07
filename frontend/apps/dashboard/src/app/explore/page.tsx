'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Loader2, RefreshCw } from 'lucide-react';
import { Header } from '../../components/Header';
import { Banner, Eyebrow, Note, Panel, Reveal, Scramble, SkyBand, Underlined } from '../../components/ui';
import { api, ExploreColumn, ExplorePayload } from '../../services/api';
import { BoxCompare, CategoryBars, Heatmap, Histogram, Legend, MiniBars, MiniHist, MissingMatrix, ParallelCoords, type Group } from '../../components/viz/charts';
import { Scatter2D, Scatter3D } from '../../components/viz/scatter';
import { AskPanel } from '../../components/agents/AskPanel';
import {
  CLUSTER_KEY,
  PALETTE,
  categoryValues,
  encodeColor,
  finite,
  fmt,
  fmtDate,
  isAxisColumn,
  isGroupColumn,
  numericValues,
  sturges,
} from '../../components/viz/lib';

type Mode = 'overview' | 'distribution' | 'scatter' | 'space' | 'projection' | 'correlation' | 'missing' | 'compare' | 'parallel' | 'ask';

const MODES: { key: Mode; label: string; hint: string }[] = [
  { key: 'overview', label: 'overview', hint: 'every column at a glance' },
  { key: 'distribution', label: 'distribution', hint: 'histogram, density, box' },
  { key: 'scatter', label: 'scatter 2d', hint: 'any two columns' },
  { key: 'space', label: '3d space', hint: 'any three columns' },
  { key: 'projection', label: 'pca + clusters', hint: 'all features squeezed to 2d/3d' },
  { key: 'correlation', label: 'correlation', hint: 'who moves with whom' },
  { key: 'missing', label: 'missingness', hint: 'where the holes are' },
  { key: 'compare', label: 'compare groups', hint: 'numeric split by category' },
  { key: 'parallel', label: 'parallel axes', hint: 'many dimensions, one picture' },
  { key: 'ask', label: 'ask ai', hint: 'questions in plain language, answered with verified queries' },
];

const DATASET_PARAM = 'dataset';

function Select({ label, value, options, onChange, allowNone }: {
  label: string;
  value: string | null;
  options: { value: string; label: string }[];
  onChange: (v: string | null) => void;
  allowNone?: string;
}) {
  return (
    <label className="ctl">
      <span className="ctl-label">{label}</span>
      <select className="select sm" value={value ?? ''} onChange={(e) => onChange(e.target.value || null)}>
        {allowNone && <option value="">{allowNone}</option>}
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

function Seg<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string; disabled?: boolean }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.value} type="button" className={value === o.value ? 'on' : ''} disabled={o.disabled} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Toggle({ on, label, onChange }: { on: boolean; label: string; onChange: (v: boolean) => void }) {
  return (
    <button type="button" className={`toggle-chip ${on ? 'on' : ''}`} onClick={() => onChange(!on)} aria-pressed={on}>
      <i />{label}
    </button>
  );
}

const kindTone: Record<string, string> = { numeric: 'accent', categorical: 'good', datetime: 'warn', text: '', id: '' };

export default function ExplorePage() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [source, setSource] = useState<'raw' | 'cleaned'>('raw');
  const [payload, setPayload] = useState<ExplorePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>('overview');

  const [colorBy, setColorBy] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [bins, setBins] = useState(0);
  const [density, setDensity] = useState(true);
  const [logY, setLogY] = useState(false);
  const [splitDist, setSplitDist] = useState(false);
  const [normalizeBars, setNormalizeBars] = useState(false);
  const [sx, setSx] = useState<string | null>(null);
  const [sy, setSy] = useState<string | null>(null);
  const [sz, setSz] = useState<string | null>(null);
  const [sizeBy, setSizeBy] = useState<string | null>(null);
  const [trend, setTrend] = useState(true);
  const [spin, setSpin] = useState(true);
  const [projDims, setProjDims] = useState<'2' | '3'>('3');
  const [corrMethod, setCorrMethod] = useState<'pearson' | 'spearman'>('pearson');
  const [sortMissing, setSortMissing] = useState(false);
  const [cmpNum, setCmpNum] = useState<string | null>(null);
  const [cmpGroup, setCmpGroup] = useState<string | null>(null);
  const [axes, setAxes] = useState<string[]>([]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setDatasetId(params.get(DATASET_PARAM));
    if (params.get('source') === 'cleaned') setSource('cleaned');
    if (!params.get(DATASET_PARAM)) {
      setLoading(false);
      setError('No dataset selected. Open the visual lab from the dashboard after loading a dataset.');
    }
  }, []);

  const load = useCallback(async (id: string, src: 'raw' | 'cleaned') => {
    setLoading(true);
    setError(null);
    try {
      setPayload(await api.explore(id, src));
    } catch (err: any) {
      setError(err.message || 'Could not load the dataset');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (datasetId) load(datasetId, source);
  }, [datasetId, source, load]);

  // Sensible defaults every time a payload arrives.
  useEffect(() => {
    if (!payload) return;
    const cols = payload.columns;
    const axisCols = cols.filter(isAxisColumn).filter((c) => !c.is_target);
    const numeric = axisCols.filter((c) => c.kind === 'numeric');
    const pool = numeric.length >= 2 ? numeric : axisCols;
    const target = cols.find((c) => c.is_target);
    const groupCols = cols.filter(isGroupColumn);
    const keep = (v: string | null, ok: (c: ExploreColumn) => boolean) => (v && cols.some((c) => c.name === v && ok(c)) ? v : null);

    const defaultColor = target && (target.kind === 'categorical' || target.kind === 'numeric')
      ? target.name
      : payload.projection?.clusters ? CLUSTER_KEY : groupCols[0]?.name ?? null;
    setColorBy((v) => (v === CLUSTER_KEY && payload.projection?.clusters ? v : keep(v, (c) => c.included)) ?? defaultColor);
    setFocus((v) => keep(v, (c) => c.included) ?? (pool[0]?.name ?? cols.find((c) => c.included)?.name ?? null));
    setSx((v) => keep(v, isAxisColumn) ?? pool[0]?.name ?? null);
    setSy((v) => keep(v, isAxisColumn) ?? pool[1]?.name ?? pool[0]?.name ?? null);
    setSz((v) => keep(v, isAxisColumn) ?? pool[2]?.name ?? (target && isAxisColumn(target) ? target.name : pool[0]?.name) ?? null);
    setCmpNum((v) => keep(v, isAxisColumn) ?? pool[0]?.name ?? null);
    setCmpGroup((v) => keep(v, isGroupColumn) ?? (target && isGroupColumn(target) ? target.name : groupCols[0]?.name) ?? null);
    setAxes((v) => {
      const valid = v.filter((a) => cols.some((c) => c.name === a && isAxisColumn(c)));
      return valid.length >= 2 ? valid : [...axisCols.slice(0, 6).map((c) => c.name), ...(target && isAxisColumn(target) ? [target.name] : [])];
    });
    if ((payload.projection?.dims ?? 0) < 3) setProjDims('2');
  }, [payload]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey) return;
      if (e.target instanceof HTMLTextAreaElement) return;
      const n = e.key === '0' ? 10 : Number(e.key);
      if (n >= 1 && n <= MODES.length) setMode(MODES[n - 1].key);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const meta = useMemo(() => new Map((payload?.columns ?? []).map((c) => [c.name, c])), [payload]);
  const color = useMemo(() => (payload ? encodeColor(payload, colorBy) : null), [payload, colorBy]);
  const axisOptions = useMemo(() => (payload?.columns ?? []).filter(isAxisColumn).map((c) => ({ value: c.name, label: `${c.name}${c.is_target ? '  ★' : ''}` })), [payload]);
  const groupOptions = useMemo(() => (payload?.columns ?? []).filter(isGroupColumn).map((c) => ({ value: c.name, label: `${c.name}${c.is_target ? '  ★' : ''}` })), [payload]);
  const colorOptions = useMemo(() => [
    ...(payload?.projection?.clusters ? [{ value: CLUSTER_KEY, label: `k-means cluster (k=${payload.projection.k})` }] : []),
    ...(payload?.columns ?? []).filter((c) => c.included && (isGroupColumn(c) || c.kind === 'numeric')).map((c) => ({ value: c.name, label: `${c.name}${c.is_target ? '  ★' : ''}` })),
  ], [payload]);

  const numCache = useMemo(() => new Map<string, ReturnType<typeof numericValues>>(), [payload]);
  const nums = useCallback((col: string | null) => {
    if (!payload || !col) return [];
    if (!numCache.has(col)) numCache.set(col, numericValues(payload, col));
    return numCache.get(col)!;
  }, [payload, numCache]);
  const pcs = useMemo(() => [0, 1, 2].map((k) => payload?.projection?.coords.map((r) => r[k] ?? 0) ?? []), [payload]);
  const isDate = (col: string | null) => !!col && meta.get(col)?.kind === 'datetime';

  const rowTip = useCallback((i: number, cols: (string | null)[]) => {
    if (!payload) return null;
    const show = Array.from(new Set([...cols, payload.target, colorBy].filter(Boolean) as string[]));
    return (
      <>
        <b>row {i + 1}</b>
        {show.map((c) => {
          const raw = c === CLUSTER_KEY ? payload.projection?.clusters?.[i] : payload.data[c]?.[i];
          const val = raw === null || raw === undefined ? 'null' : typeof raw === 'number' ? (isDate(c) ? fmtDate(raw) : fmt(raw)) : String(raw);
          return <span key={c}><em>{c === CLUSTER_KEY ? 'cluster' : c}</em> {val}</span>;
        })}
      </>
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload, colorBy, meta]);

  const groupsFor = useCallback((numCol: string, groupCol: string | null, max = 6): Group[] => {
    if (!payload || !groupCol) return [];
    const v = numericValues(payload, numCol);
    const g = categoryValues(payload, groupCol);
    const map = new Map<string, number[]>();
    g.forEach((k, i) => {
      if (k === null || v[i] === null) return;
      if (!map.has(k)) map.set(k, []);
      map.get(k)!.push(v[i]!);
    });
    return Array.from(map.entries()).sort((a, b) => b[1].length - a[1].length).slice(0, max).map(([label, values], i) => ({ label, values, color: PALETTE[i % PALETTE.length] }));
  }, [payload]);

  const missingPct = useMemo(() => Object.fromEntries((payload?.columns ?? []).map((c) => [c.name, c.missing_pct])), [payload]);
  const isMissing = useCallback((col: string, row: number) => (payload?.data[col]?.[row] ?? null) === null, [payload]);

  const backHref = datasetId ? `/?${DATASET_PARAM}=${datasetId}` : '/';
  const colorLegend = color && <Legend items={color.legend} range={color.range} label={colorBy ? `color · ${colorBy === CLUSTER_KEY ? 'cluster' : colorBy}` : undefined} />;
  const colorSelect = <Select label="color by" value={colorBy} options={colorOptions} onChange={setColorBy} allowNone="— none —" />;

  const renderView = () => {
    if (!payload || !color) return null;
    const focusMeta = focus ? meta.get(focus) : undefined;

    switch (mode) {
      case 'overview':
        return (
          <div className="overview-grid">
            {payload.columns.map((c, i) => {
              const vals = c.included && (c.kind === 'numeric' || c.kind === 'datetime') ? finite(numericValues(payload, c.name)) : [];
              return (
                <Reveal key={c.name} delay={Math.min(i, 18) * 35}>
                  <Panel
                    as="button"
                    type="button"
                    lift
                    tilt
                    className={`col-card ${c.is_target ? 'target' : ''}`}
                    onClick={() => { setFocus(c.name); setMode('distribution'); }}
                  >
                    <span className="col-card-top">
                      <span className={`chip ${kindTone[c.kind]}`}>{c.kind}</span>
                      {c.is_target && <span className="chip accent">★ target</span>}
                      <span className="col-miss" style={{ color: c.missing_pct > 20 ? 'var(--bad)' : c.missing_pct > 0 ? 'var(--warn)' : 'var(--dim)' }}>
                        {c.missing_pct > 0 ? `${c.missing_pct}% null` : 'no nulls'}
                      </span>
                    </span>
                    <span className="col-name" title={c.name}>{c.name}</span>
                    <span className="col-viz">
                      {vals.length > 1 ? <MiniHist values={vals} color={c.is_target ? 'var(--accent)' : undefined} />
                        : c.top?.length ? <MiniBars items={c.top} />
                          : <span className="dim mono" style={{ fontSize: '.7rem' }}>{c.included ? 'no chartable values' : 'not sampled (very wide dataset)'}</span>}
                    </span>
                    <span className="col-foot">
                      {c.stats ? <>μ {fmt(c.stats.mean)} · σ {fmt(c.stats.std)} · [{fmt(c.stats.min)}, {fmt(c.stats.max)}]</>
                        : c.range ? <>{c.range[0].slice(0, 10)} → {c.range[1].slice(0, 10)}</>
                          : <>{c.unique.toLocaleString()} unique</>}
                    </span>
                  </Panel>
                </Reveal>
              );
            })}
          </div>
        );

      case 'distribution': {
        const chartable = focusMeta && (focusMeta.kind === 'numeric' || focusMeta.kind === 'datetime' || focusMeta.numeric_like) && focusMeta.kind !== 'categorical';
        const vals = focus && chartable ? finite(nums(focus)) : [];
        const groupable = colorBy && colorBy !== focus && meta.get(colorBy)?.kind !== 'numeric';
        const groups = splitDist && groupable && focus ? groupsFor(focus, colorBy) : undefined;
        const sampleCats = focus && !chartable ? categoryValues(payload, focus) : [];
        let barItems: { label: string; count: number; segments?: { label: string; color: string; count: number }[] }[] = [];
        if (focus && !chartable) {
          const counts = new Map<string, number>();
          sampleCats.forEach((v) => counts.set(v ?? 'null', (counts.get(v ?? 'null') ?? 0) + 1));
          const top = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 15);
          const legendIdx = new Map(color.legend.map((l, i) => [l.label, i]));
          barItems = top.map(([label, count]) => {
            if (color.kind !== 'category' || colorBy === focus) return { label, count };
            const seg = new Map<string, number>();
            const colorVals = categoryValues(payload, colorBy!);
            sampleCats.forEach((v, i) => {
              if ((v ?? 'null') !== label) return;
              const k = colorVals[i] ?? 'null';
              const key = legendIdx.has(k) ? k : 'other';
              seg.set(key, (seg.get(key) ?? 0) + 1);
            });
            return {
              label,
              count,
              segments: color.legend.filter((l) => seg.has(l.label)).map((l) => ({ label: l.label, color: l.color, count: seg.get(l.label)! })),
            };
          });
        }
        const s = focusMeta?.stats;
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Select label="column" value={focus} options={payload.columns.filter((c) => c.included).map((c) => ({ value: c.name, label: `${c.name} · ${c.kind}` }))} onChange={setFocus} />
              {chartable ? (
                <>
                  <label className="ctl">
                    <span className="ctl-label">bins <b>{bins || 'auto'}</b></span>
                    <input type="range" min={0} max={80} value={bins} onChange={(e) => setBins(Number(e.target.value))} className="range" />
                  </label>
                  <div className="ctl-row">
                    <Toggle on={density} label="density curve" onChange={setDensity} />
                    <Toggle on={logY} label="log y" onChange={setLogY} />
                    {groupable && <Toggle on={splitDist} label={`split by ${colorBy === CLUSTER_KEY ? 'cluster' : colorBy}`} onChange={setSplitDist} />}
                  </div>
                </>
              ) : (
                <div className="ctl-row"><Toggle on={normalizeBars} label="100% stacked" onChange={setNormalizeBars} /></div>
              )}
              {colorSelect}
              {s && (
                <div className="stat-grid">
                  {([['mean', s.mean], ['median', s.median], ['std', s.std], ['skew', s.skew], ['min', s.min], ['max', s.max], ['q1', s.q1], ['q3', s.q3]] as [string, number][]).map(([k, v]) => (
                    <div key={k}><span>{k}</span><b>{fmt(v)}</b></div>
                  ))}
                  <div><span>missing</span><b>{focusMeta?.missing_pct}%</b></div>
                  <div><span>unique</span><b>{focusMeta?.unique.toLocaleString()}</b></div>
                </div>
              )}
              {s && Math.abs(s.skew) > 1 && <Note tone="red" rot={-3} delay={0.4}>heavily skewed, try log y</Note>}
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>distribution · {focus}</span></div>
                {groups ? <Legend items={groups.map((g) => ({ label: g.label, color: g.color, count: g.values.length }))} /> : <span>{chartable ? `${vals.length} values` : `${focusMeta?.unique} categories`}</span>}
              </div>
              <div className="panel-body">
                {chartable && vals.length > 1 ? (
                  <Histogram values={vals} groups={groups} bins={bins || sturges(vals.length)} density={density} log={logY} isDate={isDate(focus)} label={focus!} />
                ) : !chartable && barItems.length ? (
                  <>
                    <CategoryBars items={barItems} total={sampleCats.length} normalize={normalizeBars} />
                    {barItems[0]?.segments && <div style={{ marginTop: '1rem' }}>{colorLegend}</div>}
                  </>
                ) : (
                  <p className="empty">Not enough values to chart this column.</p>
                )}
              </div>
            </Panel>
          </div>
        );
      }

      case 'scatter':
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Select label="x axis" value={sx} options={axisOptions} onChange={setSx} />
              <Select label="y axis" value={sy} options={axisOptions} onChange={setSy} />
              <button type="button" className="btn btn-light btn-sm" onClick={() => { setSx(sy); setSy(sx); }}>⇄ swap axes</button>
              {colorSelect}
              <Select label="size by" value={sizeBy} options={axisOptions} onChange={setSizeBy} allowNone="— fixed —" />
              <div className="ctl-row"><Toggle on={trend} label="trend line + r" onChange={setTrend} /></div>
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>{sy} vs {sx}</span></div>
                {colorLegend}
              </div>
              <div className="panel-body">
                {sx && sy ? (
                  <Scatter2D
                    x={nums(sx)}
                    y={nums(sy)}
                    colors={color.colors}
                    sizes={sizeBy ? nums(sizeBy) : undefined}
                    xLabel={sx}
                    yLabel={sy}
                    xIsDate={isDate(sx)}
                    yIsDate={isDate(sy)}
                    trend={trend}
                    tooltip={(i) => rowTip(i, [sx, sy, sizeBy])}
                  />
                ) : <p className="empty">This dataset needs at least one numeric or date column for a scatter plot.</p>}
              </div>
            </Panel>
          </div>
        );

      case 'space':
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Select label="x axis" value={sx} options={axisOptions} onChange={setSx} />
              <Select label="y axis" value={sy} options={axisOptions} onChange={setSy} />
              <Select label="z axis" value={sz} options={axisOptions} onChange={setSz} />
              {colorSelect}
              <div className="ctl-row"><Toggle on={spin} label="auto-rotate" onChange={setSpin} /></div>
              <Note tone="purple" rot={-3} delay={0.5}>grab it and spin it</Note>
            </div>
            <Panel className="lab-chart dark">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>3d · {sx} × {sy} × {sz}</span></div>
                {colorLegend}
              </div>
              <div className="panel-body">
                {sx && sy && sz ? (
                  <Scatter3D x={nums(sx)} y={nums(sy)} z={nums(sz)} colors={color.colors} labels={[sx, sy, sz]} autoRotate={spin} tooltip={(i) => rowTip(i, [sx, sy, sz])} />
                ) : <p className="empty">Need numeric columns for a 3D view.</p>}
              </div>
            </Panel>
          </div>
        );

      case 'projection': {
        const proj = payload.projection;
        if (!proj) return <Panel className="panel-pad"><p className="empty">PCA needs at least two numeric features with variation.</p></Panel>;
        const c = (k: number) => pcs[k];
        const pcLabel = (k: number) => `PC${k + 1} (${(proj.explained[k] * 100).toFixed(1)}%)`;
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Seg value={projDims} onChange={setProjDims} options={[{ value: '2', label: '2d' }, { value: '3', label: '3d', disabled: proj.dims < 3 }]} />
              {colorSelect}
              {projDims === '3' && <div className="ctl-row"><Toggle on={spin} label="auto-rotate" onChange={setSpin} /></div>}
              <div className="ctl-label" style={{ marginTop: '.5rem' }}>variance explained</div>
              <div className="var-bars">
                {proj.explained.map((v, k) => (
                  <div key={k}>
                    <span>PC{k + 1}</span>
                    <i><b style={{ width: `${v * 100}%` }} /></i>
                    <span>{(v * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="ctl-label" style={{ marginTop: '.5rem' }}>what drives each axis</div>
              {proj.loadings.map((l, k) => (
                <div key={k} className="loading-row">
                  <b>PC{k + 1}</b>
                  {l.slice(0, 3).map((f) => (
                    <span key={f.feature} className={`chip ${f.weight > 0 ? 'accent' : 'bad'}`}>{f.weight > 0 ? '+' : '−'}{f.feature}</span>
                  ))}
                </div>
              ))}
              {proj.k && <p className="dim mono" style={{ fontSize: '.68rem', marginTop: '.4rem' }}>k-means picked k={proj.k} (silhouette {proj.silhouette})</p>}
            </div>
            <Panel className={`lab-chart ${projDims === '3' ? 'dark' : ''}`}>
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>pca of {proj.features.length} numeric features</span></div>
                {colorLegend}
              </div>
              <div className="panel-body">
                {projDims === '3' && proj.dims >= 3 ? (
                  <Scatter3D x={c(0)} y={c(1)} z={c(2)} colors={color.colors} labels={['PC1', 'PC2', 'PC3']} autoRotate={spin} tooltip={(i) => rowTip(i, [])} />
                ) : (
                  <Scatter2D x={c(0)} y={c(1)} colors={color.colors} xLabel={pcLabel(0)} yLabel={pcLabel(1)} tooltip={(i) => rowTip(i, [])} />
                )}
              </div>
            </Panel>
          </div>
        );
      }

      case 'correlation': {
        const corr = payload.correlation;
        if (!corr) return <Panel className="panel-pad"><p className="empty">Correlation needs at least two numeric columns.</p></Panel>;
        const m = corr[corrMethod];
        const pairs: { a: string; b: string; r: number }[] = [];
        corr.columns.forEach((a, i) => corr.columns.forEach((b, j) => {
          const r = m[i][j];
          if (j > i && r !== null) pairs.push({ a, b, r });
        }));
        pairs.sort((p, q) => Math.abs(q.r) - Math.abs(p.r));
        const openPair = (a: string, b: string) => { setSx(a); setSy(b); setMode('scatter'); };
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Seg value={corrMethod} onChange={setCorrMethod} options={[{ value: 'pearson', label: 'pearson' }, { value: 'spearman', label: 'spearman' }]} />
              <p className="dim" style={{ fontSize: '.78rem', lineHeight: 1.5 }}>
                {corrMethod === 'pearson' ? 'Linear relationships. Sensitive to outliers.' : 'Rank-based. Catches any monotonic trend and shrugs off outliers.'}
              </p>
              <div className="ctl-label">strongest pairs</div>
              <div className="pair-list">
                {pairs.slice(0, 8).map((p) => (
                  <button key={`${p.a}-${p.b}`} type="button" onClick={() => openPair(p.a, p.b)}>
                    <span className="pair-r" style={{ color: p.r > 0 ? 'var(--accent)' : 'var(--bad)' }}>{p.r > 0 ? '+' : ''}{p.r.toFixed(2)}</span>
                    <span className="pair-names">{p.a} × {p.b}</span>
                  </button>
                ))}
              </div>
              <div className="legend"><span className="dim">−1</span><span className="ramp div" /><span className="dim">+1</span></div>
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>{corrMethod} · {corr.columns.length} columns (full data)</span></div>
                <span>click any cell</span>
              </div>
              <div className="panel-body"><Heatmap columns={corr.columns} matrix={m} onPick={openPair} /></div>
            </Panel>
          </div>
        );
      }

      case 'missing': {
        const cols = payload.columns.filter((c) => c.included).map((c) => c.name);
        const withNulls = payload.columns.filter((c) => c.missing_pct > 0);
        const rowsWithNull = Array.from({ length: payload.sample_rows }, (_, r) => cols.some((c) => isMissing(c, r))).filter(Boolean).length;
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <div className="ctl-row"><Toggle on={sortMissing} label="sort rows by holes" onChange={setSortMissing} /></div>
              <div className="stat-grid">
                <div><span>cols with nulls</span><b>{withNulls.length}/{payload.total_cols}</b></div>
                <div><span>rows with nulls</span><b>{((rowsWithNull / Math.max(1, payload.sample_rows)) * 100).toFixed(1)}%</b></div>
              </div>
              <div className="ctl-label">worst columns</div>
              <div className="pair-list">
                {[...withNulls].sort((a, b) => b.missing_pct - a.missing_pct).slice(0, 8).map((c) => (
                  <button key={c.name} type="button" onClick={() => { setFocus(c.name); setMode('distribution'); }}>
                    <span className="pair-r" style={{ color: c.missing_pct > 20 ? 'var(--bad)' : 'var(--warn)' }}>{c.missing_pct}%</span>
                    <span className="pair-names">{c.name}</span>
                  </button>
                ))}
                {!withNulls.length && <span className="dim" style={{ fontSize: '.8rem' }}>No missing values anywhere. Nice.</span>}
              </div>
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>nullity matrix · black = missing</span></div>
                <span>{payload.sample_rows} rows × {cols.length} cols</span>
              </div>
              <div className="panel-body">
                <MissingMatrix columns={cols} rows={payload.sample_rows} isMissing={isMissing} missingPct={missingPct} sortRows={sortMissing} />
              </div>
            </Panel>
          </div>
        );
      }

      case 'compare': {
        const groups = cmpNum && cmpGroup ? groupsFor(cmpNum, cmpGroup, 10) : [];
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <Select label="numeric value" value={cmpNum} options={axisOptions} onChange={setCmpNum} />
              <Select label="split by" value={cmpGroup} options={[...(payload.projection?.clusters ? [{ value: CLUSTER_KEY, label: 'k-means cluster' }] : []), ...groupOptions]} onChange={setCmpGroup} />
              {groups.length > 1 && (() => {
                const meds = groups.map((g) => {
                  const s = [...g.values].sort((a, b) => a - b);
                  return { label: g.label, med: s[Math.floor(s.length / 2)] };
                }).sort((a, b) => b.med - a.med);
                return <Note tone="green" rot={-3} delay={0.4}>{meds[0].label} runs highest</Note>;
              })()}
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>{cmpNum} by {cmpGroup === CLUSTER_KEY ? 'cluster' : cmpGroup}</span></div>
                <span>box = middle 50% · dots = rows</span>
              </div>
              <div className="panel-body">
                {groups.length ? <BoxCompare groups={groups} label={cmpNum!} isDate={isDate(cmpNum)} /> : <p className="empty">Pick a numeric column and a category to compare.</p>}
              </div>
            </Panel>
          </div>
        );
      }

      case 'parallel': {
        const values = Object.fromEntries(axes.map((a) => [a, nums(a)]));
        return (
          <div className="lab-grid">
            <div className="lab-ctl">
              <div className="ctl-label">axes ({axes.length}/10)</div>
              <div className="axis-chips">
                {axisOptions.map((o) => {
                  const on = axes.includes(o.value);
                  return (
                    <button
                      key={o.value}
                      type="button"
                      className={`toggle-chip ${on ? 'on' : ''}`}
                      disabled={!on && axes.length >= 10}
                      onClick={() => setAxes((a) => (on ? a.filter((x) => x !== o.value) : [...a, o.value]))}
                    >
                      <i />{o.label}
                    </button>
                  );
                })}
              </div>
              {colorSelect}
            </div>
            <Panel className="lab-chart">
              <div className="panel-head">
                <div className="left"><span className="dots"><i /><i /><i /></span><span>parallel coordinates</span></div>
                {colorLegend}
              </div>
              <div className="panel-body">
                {axes.length >= 2 ? <ParallelCoords axes={axes} values={values} colors={color.colors} rows={payload.sample_rows} /> : <p className="empty">Pick at least two axes.</p>}
              </div>
            </Panel>
          </div>
        );
      }

      case 'ask':
        return datasetId ? <AskPanel datasetId={datasetId} columns={payload.columns} target={payload.target} source={source} /> : null;
    }
  };

  const activeIdx = MODES.findIndex((m) => m.key === mode);

  return (
    <div className="app-shell">
      <div className="grain" aria-hidden />
      <div className="guides" aria-hidden />
      <SkyBand />
      <Header datasetName={payload?.filename} />

      <main className="app-main">
        <div className="frame">
          <div className="step-head">
            <a href={backHref} className="btn btn-ghost btn-sm" style={{ alignSelf: 'flex-start', marginLeft: '-.6rem' }}>
              <ArrowLeft size={14} className="arrow arrow-left" /> back to pipeline
            </a>
            <Eyebrow label="visual-lab" caret />
            <div className="step-head-row">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.9rem' }}>
                <h1 className="h-title"><Scramble text="See it" /> <Underlined>every way.</Underlined></h1>
                <p className="h-sub">
                  {payload
                    ? <>{payload.sample_rows.toLocaleString()} of {payload.total_rows.toLocaleString()} rows sampled · {payload.total_cols} columns{payload.target && <> · target <strong>{payload.target}</strong></>}. Stats, correlation and PCA use the full data.</>
                    : 'Nine ways to look at your data. Pick one, poke it, spin it.'}
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem' }}>
                {payload && !payload.has_cleaned && source === 'raw' && <Note rot={-4} delay={0.8}>apply fixes to unlock cleaned</Note>}
                <Seg
                  value={source}
                  onChange={setSource}
                  options={[{ value: 'raw', label: 'raw data' }, { value: 'cleaned', label: 'cleaned data', disabled: !payload?.has_cleaned && source !== 'cleaned' }]}
                />
              </div>
            </div>
          </div>

          {error && (
            <Banner tone="error" onClose={() => setError(null)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span>{error}</span>
                {datasetId && <button className="btn btn-light btn-sm" onClick={() => load(datasetId, source)}><RefreshCw size={13} /> Retry</button>}
              </div>
            </Banner>
          )}

          <nav className="lab-tabs" aria-label="Visualizations">
            {MODES.map((m, i) => (
              <button key={m.key} type="button" className={mode === m.key ? 'on' : ''} onClick={() => setMode(m.key)} title={`${m.hint} (key ${i + 1})`}>
                <span className="n">{(i + 1) % 10}</span>{m.label}
              </button>
            ))}
            <span className="lab-tabs-fill" style={{ left: `${(activeIdx / MODES.length) * 100}%`, width: `${100 / MODES.length}%` }} />
          </nav>
          <p className="lab-mode-hint mono"><span className="dim">$</span> {MODES[activeIdx].hint}<span className="dim"> · press 1–9 or 0 to switch</span></p>

          {loading ? (
            <Panel className="panel-pad lab-loading">
              <Loader2 size={18} className="animate-spin" />
              <span className="mono"><Scramble text={`sampling & profiling ${source} data`} /></span>
            </Panel>
          ) : (
            <div key={`${mode}-${source}`} className="step-enter">{renderView()}</div>
          )}
        </div>
      </main>
    </div>
  );
}
