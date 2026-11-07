'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowLeft, BarChart3, Loader2, RefreshCw, Sparkles } from 'lucide-react';
import { Header } from '../../components/Header';
import { Banner, CountUp, Eyebrow, Note, Panel, Reveal, Scramble, SkyBand, Underlined } from '../../components/ui';
import { api, SummaryColumn, SummaryPayload, TASK_LABEL, isUnsupervised } from '../../services/api';
import { BarList, BoxStrip, Donut, HistBars, kindColor, Ring, Timeline } from '../../components/summary/SummaryCharts';
import { Heatmap, Legend } from '../../components/viz/charts';
import { Scatter2D } from '../../components/viz/scatter';
import { fmt, PALETTE } from '../../components/viz/lib';
import { ModelLeaderboard } from '../../components/ModelLeaderboard';
import { HealthScoreGauge } from '../../components/HealthScoreGauge';
import { Markdown } from '../../components/agents/Markdown';
import { useAgentRun } from '../../components/agents/useAgentRun';

const DATASET_PARAM = 'dataset';
const TONE: Record<string, string> = { good: 'var(--good)', warn: 'var(--warn)', bad: 'var(--bad)', info: 'var(--accent)' };
const SEV_COLOR: Record<string, string> = { critical: '#b42318', high: 'var(--bad)', medium: 'var(--warn)', low: 'var(--accent)', info: 'var(--dim)' };

function Section({ id, eyebrow, title, sub, children }: { id: string; eyebrow: string; title: React.ReactNode; sub?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section id={id} className="sum-section">
      <Reveal>
        <div className="sum-head">
          <Eyebrow label={eyebrow} />
          <h2>{title}</h2>
          {sub && <p>{sub}</p>}
        </div>
      </Reveal>
      {children}
    </section>
  );
}

function Kpi({ label, value, decimals = 0, suffix, tone, delay = 0 }: { label: string; value: number; decimals?: number; suffix?: string; tone?: string; delay?: number }) {
  return (
    <Reveal delay={delay}>
      <div className="kpi">
        <span className="kpi-label mono">{label}</span>
        <b style={{ color: tone }}><CountUp value={value} decimals={decimals} duration={1400} />{suffix}</b>
      </div>
    </Reveal>
  );
}

function NumericCard({ c, i }: { c: SummaryColumn; i: number }) {
  const s = c.stats;
  return (
    <Reveal delay={Math.min(i, 12) * 50}>
      <Panel className="sum-card" lift>
        <div className="sum-card-top">
          <b title={c.name}>{c.name}</b>
          {c.is_target && <span className="chip accent">★ target</span>}
          {s && Math.abs(s.skew) > 1 && <span className="chip warn">skew {s.skew.toFixed(1)}</span>}
          {c.missing_pct > 0 && <span className="chip" style={{ color: c.missing_pct > 20 ? 'var(--bad)' : 'var(--warn)' }}>{c.missing_pct}% null</span>}
        </div>
        {c.hist ? <HistBars hist={c.hist} color={PALETTE[i % PALETTE.length]} highlight={c.box ? [c.box.q1, c.box.q3] : null} /> : <p className="empty">single value</p>}
        {c.box && <BoxStrip box={c.box} />}
        {s && (
          <div className="sum-stats mono">
            <span>mean <b>{fmt(s.mean)}</b></span>
            <span>median <b>{fmt(s.median)}</b></span>
            <span>σ <b>{fmt(s.std)}</b></span>
            <span>range <b>{fmt(s.min)} – {fmt(s.max)}</b></span>
          </div>
        )}
      </Panel>
    </Reveal>
  );
}

function CategoryCard({ c, i }: { c: SummaryColumn; i: number }) {
  const total = (c.top || []).reduce((s, t) => s + t.count, 0) + (c.other_count || 0);
  const items = (c.top || []).slice(0, 8).map((t) => ({ label: t.value, value: t.count, display: `${((t.count / Math.max(1, total)) * 100).toFixed(1)}%` }));
  if (c.other_count) items.push({ label: 'all other values', value: c.other_count, display: `${((c.other_count / Math.max(1, total)) * 100).toFixed(1)}%` });
  return (
    <Reveal delay={Math.min(i, 12) * 50}>
      <Panel className="sum-card" lift>
        <div className="sum-card-top">
          <b title={c.name}>{c.name}</b>
          {c.is_target && <span className="chip accent">★ target</span>}
          <span className="chip">{c.unique.toLocaleString()} distinct</span>
          {(c.top_share ?? 0) >= 0.9 && <span className="chip warn">near-constant</span>}
          {c.missing_pct > 0 && <span className="chip" style={{ color: c.missing_pct > 20 ? 'var(--bad)' : 'var(--warn)' }}>{c.missing_pct}% null</span>}
        </div>
        <BarList items={items} colorFor={() => PALETTE[(i + 1) % PALETTE.length]} compact />
      </Panel>
    </Reveal>
  );
}

export default function SummaryPage() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [source, setSource] = useState<'raw' | 'cleaned'>('raw');
  const [data, setData] = useState<SummaryPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState('overview');

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setDatasetId(params.get(DATASET_PARAM));
    if (params.get('source') === 'cleaned') setSource('cleaned');
    if (!params.get(DATASET_PARAM)) {
      setLoading(false);
      setError('No dataset selected. Open the summary from the dashboard after setting an objective.');
    }
  }, []);

  const load = useCallback(async (id: string, src: 'raw' | 'cleaned') => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.summary(id, src));
    } catch (e: any) {
      setError(e.message || 'Could not build the summary');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (datasetId) load(datasetId, source);
  }, [datasetId, source, load]);

  const reporter = useAgentRun((run) => {
    if (run.status === 'completed' && datasetId) load(datasetId, source);
  });
  const writing = !!reporter.run && ['queued', 'running'].includes(reporter.run.status);

  const numeric = useMemo(() => (data?.columns || []).filter((c) => c.kind === 'numeric' && c.hist), [data]);
  const categorical = useMemo(() => (data?.columns || []).filter((c) => (c.kind === 'categorical' || c.kind === 'text') && c.top?.length), [data]);
  const dated = useMemo(() => (data?.columns || []).filter((c) => c.kind === 'datetime' && c.timeline), [data]);

  const sections = useMemo(() => {
    if (!data) return [];
    return [
      { id: 'overview', label: 'overview' },
      { id: 'insights', label: 'key findings' },
      { id: 'composition', label: 'columns' },
      ...(data.target_info ? [{ id: 'target', label: 'target' }] : []),
      ...(numeric.length ? [{ id: 'numeric', label: 'distributions' }] : []),
      ...(categorical.length ? [{ id: 'categorical', label: 'categories' }] : []),
      ...(dated.length ? [{ id: 'time', label: 'time' }] : []),
      ...(data.correlation.heatmap ? [{ id: 'relations', label: 'relationships' }] : []),
      ...(data.projection ? [{ id: 'structure', label: 'structure' }] : []),
      ...(data.health ? [{ id: 'health', label: 'health' }] : []),
      ...(data.execution ? [{ id: 'cleaning', label: 'cleaning' }] : []),
      ...(data.leaderboard ? [{ id: 'models', label: isUnsupervised(data.problem_type) ? 'findings' : 'models' }] : []),
      { id: 'report', label: 'report' },
    ];
  }, [data, numeric.length, categorical.length, dated.length]);

  useEffect(() => {
    if (!sections.length) return;
    const io = new IntersectionObserver((entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin: '-30% 0px -60% 0px' });
    sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, [sections]);

  const backHref = datasetId ? `/?${DATASET_PARAM}=${datasetId}` : '/';
  const task = data?.problem_type ? TASK_LABEL[data.problem_type] ?? data.problem_type : null;

  const projection = useMemo(() => {
    const p = data?.projection;
    if (!p) return null;
    const labels = p.color || [];
    const order = Array.from(new Set(labels.filter((l): l is string => l !== null))).slice(0, 9);
    const colorOf = (l: string | null) => (l === null ? '#d6d2ca' : l === 'anomaly' ? 'var(--bad)' : l === 'noise' ? '#b9b5ad' : PALETTE[order.indexOf(l) % PALETTE.length] ?? '#b9b5ad');
    return {
      x: p.coords.map((r) => r[0]),
      y: p.coords.map((r) => r[1] ?? 0),
      colors: p.coords.map((_, i) => (labels.length ? colorOf(labels[i] ?? null) : 'var(--ink)')),
      legend: order.map((l) => ({ label: l, color: colorOf(l), count: labels.filter((x) => x === l).length })),
      labels,
    };
  }, [data]);

  return (
    <div className="app-shell">
      <div className="grain" aria-hidden />
      <div className="guides" aria-hidden />
      <SkyBand />
      <Header datasetName={data?.filename} exploreHref={datasetId ? `/explore?${DATASET_PARAM}=${datasetId}&source=${source}` : undefined} />

      <main className="app-main">
        <div className="frame">
          <div className="step-head">
            <a href={backHref} className="btn btn-ghost btn-sm" style={{ alignSelf: 'flex-start', marginLeft: '-.6rem' }}>
              <ArrowLeft size={14} className="arrow arrow-left" /> back to pipeline
            </a>
            <Eyebrow label="summary" caret />
            <div className="step-head-row">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.9rem' }}>
                <h1 className="h-title"><Scramble text="The data," /> <Underlined>told.</Underlined></h1>
                <p className="h-sub">
                  {data ? <>Every chart, driver and finding for <strong>{data.filename}</strong>
                    {task && <> · {task}{data.target && <> on <strong>{data.target}</strong></>}</>}. Computed on all {data.rows.toLocaleString()} rows.</>
                    : 'Building charts and insights from the full dataset…'}
                </p>
              </div>
              <div className="seg">
                <button type="button" className={source === 'raw' ? 'on' : ''} onClick={() => setSource('raw')}>raw data</button>
                <button type="button" className={source === 'cleaned' ? 'on' : ''} disabled={!data?.has_cleaned && source !== 'cleaned'} onClick={() => setSource('cleaned')}>cleaned data</button>
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

          {loading && (
            <Panel className="panel-pad lab-loading">
              <Loader2 size={18} className="animate-spin" />
              <span className="mono"><Scramble text={`profiling every column of the ${source} data`} /></span>
            </Panel>
          )}

          {data && !loading && (
            <div key={source} className="step-enter">
              <nav className="sum-nav" aria-label="Summary sections">
                {sections.map((s) => (
                  <a key={s.id} href={`#${s.id}`} className={active === s.id ? 'on' : ''}>{s.label}</a>
                ))}
              </nav>

              {/* ── Overview ── */}
              <section id="overview" className="sum-section">
                <div className="kpi-grid">
                  <Kpi label="rows" value={data.rows} />
                  <Kpi label="columns" value={data.cols} delay={60} />
                  <Kpi label="cells blank" value={data.missing.cells_pct} decimals={2} suffix="%" tone={data.missing.cells_pct > 5 ? 'var(--warn)' : undefined} delay={120} />
                  <Kpi label="complete rows" value={data.missing.complete_rows_pct} decimals={1} suffix="%" delay={180} />
                  <Kpi label="duplicate rows" value={data.duplicates} tone={data.duplicates ? 'var(--warn)' : undefined} delay={240} />
                  {data.health && (
                    <Reveal delay={300}>
                      <div className="kpi kpi-ring">
                        <span className="kpi-label mono">health score</span>
                        <Ring value={data.health.composite_score} size={84} />
                      </div>
                    </Reveal>
                  )}
                </div>
              </section>

              {/* ── Insights ── */}
              <Section id="insights" eyebrow="key-findings" title={<>What stands out</>} sub="Each finding is computed from the full data, ordered from shape to outcome.">
                <div className="insight-grid">
                  {data.insights.map((ins, i) => (
                    <Reveal key={`${ins.kind}-${i}`} delay={i * 70}>
                      <div className={`insight ${ins.tone}`}>
                        <span className="insight-kind mono" style={{ color: TONE[ins.tone] }}>{ins.kind}</span>
                        <b>{ins.title}</b>
                        <p>{ins.text}</p>
                      </div>
                    </Reveal>
                  ))}
                </div>
              </Section>

              {/* ── Composition ── */}
              <Section id="composition" eyebrow="columns" title="What the columns hold" sub="Column types, and how complete each one is.">
                <div className="sum-two">
                  <Reveal>
                    <Panel className="panel-pad" style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <Donut
                        segments={Object.entries(data.kinds).map(([k, v]) => ({ label: k, value: v, color: kindColor[k] ?? '#b9b5ad' }))}
                        label={data.cols}
                        sub="columns"
                      />
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                        {Object.entries(data.kinds).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                          <span key={k} className="legend-row"><i style={{ background: kindColor[k] ?? '#b9b5ad' }} />{k}<b className="mono">{v}</b></span>
                        ))}
                      </div>
                    </Panel>
                  </Reveal>
                  <Reveal delay={100}>
                    <Panel className="panel-pad">
                      <div className="ctl-label" style={{ marginBottom: '.6rem' }}>completeness by column</div>
                      {data.missing.by_column.length ? (
                        <BarList
                          items={data.missing.by_column.slice(0, 12).map((m) => ({ label: m.column, value: m.pct, display: `${m.pct}% blank`, color: m.pct > 30 ? 'var(--bad)' : m.pct > 5 ? 'var(--warn)' : 'var(--accent)' }))}
                          max={100}
                        />
                      ) : <p className="dim" style={{ fontSize: '.85rem' }}>Every column is fully populated.</p>}
                    </Panel>
                  </Reveal>
                </div>
              </Section>

              {/* ── Target ── */}
              {data.target_info && (
                <Section id="target" eyebrow="target" title={<>What drives <span className="hl-word">{data.target_info.name}</span></>}
                  sub="The target's distribution, the features most associated with it (chance-corrected), and how the target changes across groups of the top drivers.">
                  <div className="sum-two">
                    <Reveal>
                      <Panel className="panel-pad">
                        <div className="ctl-label" style={{ marginBottom: '.6rem' }}>{data.target_info.classes ? 'class balance' : 'distribution'}</div>
                        {data.target_info.classes ? (
                          <Donut
                            segments={data.target_info.classes.map((c, i) => ({ label: c.value, value: c.count, color: PALETTE[i % PALETTE.length] }))}
                            label={data.target_info.classes.length}
                            sub="classes"
                            size={200}
                          />
                        ) : data.target_info.hist ? (
                          <>
                            <HistBars hist={data.target_info.hist} color="var(--accent)" height={170} />
                            {data.target_info.box && <BoxStrip box={data.target_info.box} />}
                          </>
                        ) : null}
                        {data.target_info.classes && (
                          <div style={{ marginTop: '1rem' }}>
                            <BarList items={data.target_info.classes.map((c, i) => ({ label: c.value, value: c.count, display: c.count.toLocaleString(), color: PALETTE[i % PALETTE.length] }))} compact />
                          </div>
                        )}
                      </Panel>
                    </Reveal>
                    <Reveal delay={100}>
                      <Panel className="panel-pad">
                        <div className="ctl-label" style={{ marginBottom: '.6rem' }}>strongest drivers · association 0–1</div>
                        {data.target_info.drivers.length ? (
                          <BarList
                            items={data.target_info.drivers.map((d) => ({ label: d.feature, value: d.association, display: d.association.toFixed(3),
                              color: d.association >= 0.3 ? 'var(--good)' : d.association >= 0.1 ? 'var(--accent)' : 'var(--dim)' }))}
                            max={Math.max(0.3, ...data.target_info.drivers.map((d) => d.association))}
                          />
                        ) : <p className="dim">No feature could be related to the target.</p>}
                        <p className="dim mono" style={{ fontSize: '.66rem', marginTop: '.75rem' }}>≥ 0.3 strong · 0.1–0.3 moderate · under 0.1 weak</p>
                      </Panel>
                    </Reveal>
                  </div>
                  <div className="sum-three">
                    {data.target_info.drivers.filter((d) => d.breakdown).slice(0, 3).map((d, i) => (
                      <Reveal key={d.feature} delay={i * 90}>
                        <Panel className="panel-pad">
                          <div className="ctl-label">{d.breakdown!.measure} by <b style={{ color: 'var(--ink)' }}>{d.feature}</b></div>
                          <p className="dim mono" style={{ fontSize: '.66rem', margin: '.2rem 0 .7rem' }}>overall {fmt(d.breakdown!.overall)}</p>
                          <BarList
                            items={d.breakdown!.groups.map((g) => ({ label: g.group, value: g.value, display: fmt(g.value), note: `n=${g.n}`,
                              color: g.value >= d.breakdown!.overall ? 'var(--accent)' : 'var(--dim)' }))}
                            compact
                          />
                        </Panel>
                      </Reveal>
                    ))}
                  </div>
                </Section>
              )}

              {/* ── Numeric distributions ── */}
              {numeric.length > 0 && (
                <Section id="numeric" eyebrow="distributions" title="How the numbers are spread"
                  sub="Full-data histograms. The darker band is the middle 50% of values; the strip below is the box plot (whiskers = typical range).">
                  <div className="sum-grid">{numeric.map((c, i) => <NumericCard key={c.name} c={c} i={i} />)}</div>
                </Section>
              )}

              {/* ── Categorical ── */}
              {categorical.length > 0 && (
                <Section id="categorical" eyebrow="categories" title="The most common values" sub="Top values per column, as a share of all rows.">
                  <div className="sum-grid">{categorical.slice(0, 18).map((c, i) => <CategoryCard key={c.name} c={c} i={i} />)}</div>
                </Section>
              )}

              {/* ── Time ── */}
              {dated.length > 0 && (
                <Section id="time" eyebrow="time" title="Rows over time" sub="How many rows fall in each period of every date column.">
                  <div className="sum-two">
                    {dated.map((c, i) => (
                      <Reveal key={c.name} delay={i * 80}>
                        <Panel className="panel-pad">
                          <div className="sum-card-top">
                            <b>{c.name}</b>
                            <span className="chip">per {c.timeline!.granularity}</span>
                            <span className="chip">{c.timeline!.span_days.toLocaleString()} days</span>
                          </div>
                          <Timeline points={c.timeline!.points} color={PALETTE[(i + 3) % PALETTE.length]} />
                        </Panel>
                      </Reveal>
                    ))}
                  </div>
                </Section>
              )}

              {/* ── Relationships ── */}
              {data.correlation.heatmap && (
                <Section id="relations" eyebrow="relationships" title="Which columns move together" sub="Pearson correlation on the full data (−1 to +1).">
                  <div className="sum-two wide-left">
                    <Reveal>
                      <Panel>
                        <div className="panel-head"><div className="left"><span className="dots"><i /><i /><i /></span><span>correlation heatmap</span></div><span>hover a cell</span></div>
                        <div className="panel-body"><Heatmap columns={data.correlation.heatmap.columns} matrix={data.correlation.heatmap.matrix} /></div>
                      </Panel>
                    </Reveal>
                    <Reveal delay={100}>
                      <Panel className="panel-pad">
                        <div className="ctl-label" style={{ marginBottom: '.6rem' }}>strongest pairs</div>
                        <BarList
                          items={data.correlation.pairs.slice(0, 10).map((p) => ({ label: `${p.a} × ${p.b}`, value: Math.abs(p.r), display: `${p.r > 0 ? '+' : ''}${p.r.toFixed(2)}`,
                            color: p.r > 0 ? 'var(--accent)' : 'var(--bad)' }))}
                          max={1}
                        />
                      </Panel>
                    </Reveal>
                  </div>
                </Section>
              )}

              {/* ── Structure ── */}
              {data.projection && projection && (
                <Section id="structure" eyebrow="structure" title="The whole dataset in two dimensions"
                  sub={<>PCA squeezes {data.projection.features.length} numeric features into 2 axes ({((data.projection.explained[0] + (data.projection.explained[1] || 0)) * 100).toFixed(1)}% of the variation).
                    {data.projection.color_label && <> Colour: {data.projection.color_label}.</>}</>}>
                  <Reveal>
                    <Panel>
                      <div className="panel-head">
                        <div className="left"><span className="dots"><i /><i /><i /></span><span>pca projection</span></div>
                        {projection.legend.length > 0 && <Legend items={projection.legend} />}
                      </div>
                      <div className="panel-body">
                        <Scatter2D
                          x={projection.x}
                          y={projection.y}
                          colors={projection.colors}
                          xLabel={`PC1 (${(data.projection.explained[0] * 100).toFixed(1)}%)`}
                          yLabel={`PC2 (${((data.projection.explained[1] || 0) * 100).toFixed(1)}%)`}
                          height={440}
                          tooltip={(i) => <><b>row sample {i + 1}</b>{projection.labels[i] != null && <span>{projection.labels[i]}</span>}</>}
                        />
                      </div>
                      <div className="loading-strip">
                        {data.projection.loadings.slice(0, 2).map((l, k) => (
                          <span key={k} className="mono"><b>PC{k + 1}</b> {l.slice(0, 3).map((f) => `${f.weight > 0 ? '+' : '−'}${f.feature}`).join('  ')}</span>
                        ))}
                      </div>
                    </Panel>
                  </Reveal>
                </Section>
              )}

              {/* ── Health ── */}
              {data.health && (
                <Section id="health" eyebrow="health" title="How ready it is for machine learning" sub="Seven checks measured on the amount of data affected, combined with a weighted geometric mean.">
                  <Reveal><HealthScoreGauge score={data.health} stageName={source === 'cleaned' ? 'before cleaning' : 'raw dataset'} problemType={data.problem_type ?? undefined} /></Reveal>
                  {data.issues.length > 0 && (
                    <div className="sum-two">
                      <Reveal>
                        <Panel className="panel-pad">
                          <div className="ctl-label" style={{ marginBottom: '.6rem' }}>issues by severity</div>
                          <BarList
                            items={['critical', 'high', 'medium', 'low', 'info'].map((s) => ({ label: s, value: data.issues.filter((i) => i.severity === s).length, display: String(data.issues.filter((i) => i.severity === s).length), color: SEV_COLOR[s] })).filter((x) => x.value > 0)}
                          />
                        </Panel>
                      </Reveal>
                      <Reveal delay={100}>
                        <Panel className="panel-pad issue-scroll">
                          <div className="ctl-label" style={{ marginBottom: '.6rem' }}>every issue found</div>
                          {data.issues.map((iss, i) => (
                            <div key={i} className="issue-line"><i style={{ background: SEV_COLOR[iss.severity] }} /><span>{iss.title}</span></div>
                          ))}
                        </Panel>
                      </Reveal>
                    </div>
                  )}
                </Section>
              )}

              {/* ── Cleaning ── */}
              {data.execution && (
                <Section id="cleaning" eyebrow="cleaning" title="What cleaning changed" sub={`${data.execution.transformations_applied.length} steps, applied in order and reproduced by the exported script.`}>
                  <div className="sum-two">
                    <Reveal>
                      <Panel className="panel-pad" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around', gap: '1rem', flexWrap: 'wrap' }}>
                        <div className="ring-label"><Ring value={data.execution.before_health_score} size={120} /><span className="mono">before</span></div>
                        <span className="mono" style={{ fontSize: '1.4rem', color: 'var(--dim)' }}>→</span>
                        <div className="ring-label"><Ring value={data.execution.after_health_score} size={120} /><span className="mono">after</span></div>
                        <div style={{ width: '100%', textAlign: 'center' }} className="mono dim">
                          {data.execution.cleaned_rows.toLocaleString()} rows × {data.execution.cleaned_cols} model-ready columns
                        </div>
                      </Panel>
                    </Reveal>
                    <Reveal delay={100}>
                      <Panel className="panel-pad issue-scroll">
                        <ol className="steps-list" style={{ padding: 0 }}>
                          {data.execution.transformations_applied.map((s, i) => <li key={i}>{s}</li>)}
                        </ol>
                      </Panel>
                    </Reveal>
                  </div>
                </Section>
              )}

              {/* ── Models ── */}
              {data.leaderboard && (
                <Section id="models" eyebrow={isUnsupervised(data.problem_type) ? 'findings' : 'models'}
                  title={data.problem_type === 'clustering' ? 'The segments found' : data.problem_type === 'anomaly_detection' ? 'The anomalies found' : 'How well it can be predicted'}>
                  <ModelLeaderboard leaderboard={data.leaderboard} />
                </Section>
              )}

              {/* ── Report ── */}
              <Section id="report" eyebrow="report" title="The written report" sub="Written by the Report agent via OpenRouter. Every number is filled in from computed facts, never typed by the model.">
                <Reveal>
                  <Panel className="panel-pad report-panel">
                    {data.report?.markdown ? (
                      <>
                        <Markdown text={data.report.markdown} />
                        <p className="dim mono" style={{ fontSize: '.66rem', marginTop: '1rem' }}>
                          {data.report.source === 'agent' ? 'written by the agent, every number verified' : 'deterministic template (no model available)'} · {new Date(data.report.generated_at).toLocaleString()}
                        </p>
                      </>
                    ) : <p className="dim">No report yet.</p>}
                    <div style={{ display: 'flex', gap: '.75rem', alignItems: 'center', marginTop: '1rem', flexWrap: 'wrap' }}>
                      <button className="btn btn-dark btn-sm" disabled={writing || !datasetId} onClick={() => datasetId && reporter.start(datasetId, 'report')}>
                        {writing ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                        {writing ? 'writing…' : data.report ? 'Rewrite with AI' : 'Write with AI'}
                      </button>
                      {writing && <span className="mono dim" style={{ fontSize: '.7rem' }}>{reporter.run?.steps.slice(-1)[0]?.text}</span>}
                      {reporter.error && <span className="agent-error">{reporter.error}</span>}
                      <a href={datasetId ? `/explore?${DATASET_PARAM}=${datasetId}&source=${source}` : '#'} className="btn btn-light btn-sm"><BarChart3 size={13} /> Dig deeper in the visual lab</a>
                    </div>
                  </Panel>
                </Reveal>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}><Note rot={-3} delay={0.3}>that&apos;s the whole story</Note></div>
              </Section>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
