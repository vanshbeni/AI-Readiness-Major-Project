'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  INK,
  OTHER_COLOR,
  PALETTE,
  divColor,
  extent,
  fmt,
  fmtDate,
  histogram,
  kde,
  linear,
  niceTicks,
  pad,
  quantile,
  setupCanvas,
  useIntro,
  useSize,
  type Num,
} from './lib';

const M = { l: 52, r: 16, t: 14, b: 40 };

export interface Group {
  label: string;
  color: string;
  values: number[];
}

// ─── Histogram ──────────────────────────────────────────────────────────────

interface HistogramProps {
  values: number[];
  groups?: Group[];
  bins: number;
  density?: boolean;
  isDate?: boolean;
  log?: boolean;
  label: string;
  height?: number;
}

export function Histogram({ values, groups, bins, density = true, isDate, log, label, height = 380 }: HistogramProps) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const intro = useIntro(`${label}|${bins}|${groups?.length ?? 0}|${values.length}`);
  const [hover, setHover] = useState<number | null>(null);

  const domain = useMemo(() => extent(values), [values]);
  const layers = useMemo(() => {
    const src = groups && groups.length ? groups : [{ label, color: PALETTE[0], values }];
    return src.map((g) => ({ ...g, bins: histogram(g.values, bins, domain) }));
  }, [groups, values, bins, domain, label]);

  const maxCount = Math.max(1, ...layers.flatMap((l) => l.bins.map((b) => b.count)));
  const innerW = Math.max(10, width - M.l - M.r);
  const boxH = 34;
  const innerH = height - M.t - M.b - boxH;
  const sx = linear(domain[0], domain[1], M.l, M.l + innerW);
  const yMax = log ? Math.log10(maxCount + 1) : maxCount;
  const sy = linear(0, yMax * 1.08, M.t + innerH, M.t);
  const yv = (c: number) => (log ? Math.log10(c + 1) : c);
  const binW = (domain[1] - domain[0]) / bins;

  const curve = useMemo(() => {
    if (!density || values.length < 5) return null;
    const xs = Array.from({ length: 120 }, (_, i) => domain[0] + ((domain[1] - domain[0]) * i) / 119);
    return layers.map((l) => {
      const dens = kde(l.values, xs, binW);
      return { color: l.color, pts: xs.map((x, i) => [x, dens[i]] as [number, number]) };
    });
  }, [density, layers, domain, binW, values.length]);

  const sorted = useMemo(() => [...values].sort((a, b) => a - b), [values]);
  const q1 = quantile(sorted, 0.25);
  const q2 = quantile(sorted, 0.5);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;
  const wLo = sorted.find((v) => v >= q1 - 1.5 * iqr) ?? domain[0];
  const wHi = [...sorted].reverse().find((v) => v <= q3 + 1.5 * iqr) ?? domain[1];
  const xf = isDate ? fmtDate : fmt;
  const overlay = layers.length > 1;
  const barGap = innerW / bins > 6 ? 1 : 0;

  return (
    <div ref={ref} className="viz-wrap" style={{ height }} onPointerLeave={() => setHover(null)}>
      <svg className="viz-svg" width={width} height={height} style={{ pointerEvents: 'all' }}>
        {niceTicks(0, yMax * 1.08, 5).map((t) => (
          <g key={t}>
            <line x1={M.l} x2={M.l + innerW} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={M.l - 8} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {log ? fmt(10 ** t - 1) : fmt(t)}
            </text>
          </g>
        ))}
        {niceTicks(domain[0], domain[1], Math.max(3, Math.floor(innerW / 100))).map((t) => (
          <text key={t} x={sx(t)} y={M.t + innerH + boxH + 18} className="tick" textAnchor="middle">{xf(t)}</text>
        ))}
        {layers.map((l, li) =>
          l.bins.map((b, i) => {
            const h = (M.t + innerH - sy(yv(b.count))) * intro;
            return (
              <rect
                key={`${li}-${i}`}
                x={sx(b.x0) + barGap}
                width={Math.max(0.5, sx(b.x1) - sx(b.x0) - barGap * 2)}
                y={M.t + innerH - h}
                height={h}
                fill={l.color}
                opacity={overlay ? 0.32 : hover === null || hover === i ? 0.85 : 0.4}
                onPointerEnter={() => setHover(i)}
              />
            );
          }),
        )}
        {curve?.map((c, ci) => (
          <path
            key={ci}
            d={c.pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x)},${sy(yv(y))}`).join('')}
            fill="none"
            stroke={overlay ? c.color : INK}
            strokeWidth={1.8}
            pathLength={1}
            className="draw-path"
            style={{ ['--delay' as string]: '0.25s' }}
          />
        ))}
        <line x1={sx(q2)} x2={sx(q2)} y1={M.t} y2={M.t + innerH} className="marker" />
        <text x={sx(q2) + 4} y={M.t + 10} className="marker-label">median {xf(q2)}</text>
        <g transform={`translate(0 ${M.t + innerH + 10})`}>
          <line x1={sx(wLo)} x2={sx(wHi)} y1={10} y2={10} stroke={INK} strokeWidth={1} />
          <line x1={sx(wLo)} x2={sx(wLo)} y1={4} y2={16} stroke={INK} />
          <line x1={sx(wHi)} x2={sx(wHi)} y1={4} y2={16} stroke={INK} />
          <rect x={sx(q1)} width={Math.max(1, sx(q3) - sx(q1))} y={2} height={16} fill="#fff" stroke={INK} rx={3} />
          <line x1={sx(q2)} x2={sx(q2)} y1={2} y2={18} stroke="#6d4aff" strokeWidth={2} />
          {sorted.filter((v) => v < wLo || v > wHi).slice(0, 300).map((v, i) => (
            <circle key={i} cx={sx(v)} cy={10} r={2} fill="#e5484d" opacity={0.6} />
          ))}
        </g>
        <text x={M.l + innerW / 2} y={height - 4} className="axis-label" textAnchor="middle">{label} →</text>
      </svg>
      {hover !== null && !overlay && layers[0].bins[hover] && (
        <div className="viz-tip" style={{ left: Math.min(sx(layers[0].bins[hover].x1) + 8, width - 200), top: M.t + 4 }}>
          <b>{xf(layers[0].bins[hover].x0)} – {xf(layers[0].bins[hover].x1)}</b>
          <span>{layers[0].bins[hover].count} rows · {((layers[0].bins[hover].count / values.length) * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}

// ─── Category bars ──────────────────────────────────────────────────────────

export interface BarItem {
  label: string;
  count: number;
  segments?: { label: string; color: string; count: number }[];
}

export function CategoryBars({ items, total, normalize }: { items: BarItem[]; total: number; normalize?: boolean }) {
  const intro = useIntro(items.map((i) => i.label).join('|') + String(normalize));
  const max = Math.max(1, ...items.map((i) => i.count));
  const [hover, setHover] = useState<{ row: number; seg: number } | null>(null);
  return (
    <div className="cat-bars">
      {items.map((item, r) => {
        const width = normalize ? 100 : (item.count / max) * 100;
        return (
          <div key={item.label} className="cat-row" onPointerLeave={() => setHover(null)}>
            <span className="cat-label" title={item.label}>{item.label}</span>
            <span className="cat-track">
              <span className="cat-fill" style={{ width: `${width * intro}%` }}>
                {item.segments
                  ? item.segments.map((s, si) => (
                    <i
                      key={s.label}
                      style={{ width: `${(s.count / (item.count || 1)) * 100}%`, background: s.color, opacity: hover && (hover.row !== r || hover.seg !== si) ? 0.45 : 1 }}
                      onPointerEnter={() => setHover({ row: r, seg: si })}
                      title={`${s.label}: ${s.count} (${((s.count / (item.count || 1)) * 100).toFixed(1)}%)`}
                    />
                  ))
                  : <i style={{ width: '100%', background: PALETTE[0] }} />}
              </span>
            </span>
            <span className="cat-count">
              {hover?.row === r && item.segments
                ? `${item.segments[hover.seg].label} ${((item.segments[hover.seg].count / (item.count || 1)) * 100).toFixed(0)}%`
                : `${item.count} · ${((item.count / (total || 1)) * 100).toFixed(1)}%`}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Box + jitter comparison ────────────────────────────────────────────────

export function BoxCompare({ groups, label, height = 420, isDate }: { groups: Group[]; label: string; height?: number; isDate?: boolean }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const intro = useIntro(`${label}|${groups.map((g) => g.label).join()}`);
  const all = groups.flatMap((g) => g.values);
  const [lo, hi] = pad(extent(all), 0.06);
  const innerW = Math.max(10, width - M.l - M.r);
  const innerH = height - M.t - M.b;
  const sy = linear(lo, hi, M.t + innerH, M.t);
  const band = innerW / Math.max(1, groups.length);
  const yf = isDate ? fmtDate : fmt;

  return (
    <div ref={ref} className="viz-wrap" style={{ height }}>
      <svg className="viz-svg" width={width} height={height}>
        {niceTicks(lo, hi, 6).map((t) => (
          <g key={t}>
            <line x1={M.l} x2={M.l + innerW} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={M.l - 8} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">{yf(t)}</text>
          </g>
        ))}
        {groups.map((g, gi) => {
          const s = [...g.values].sort((a, b) => a - b);
          if (!s.length) return null;
          const q1 = quantile(s, 0.25);
          const q2 = quantile(s, 0.5);
          const q3 = quantile(s, 0.75);
          const iqr = q3 - q1;
          const wl = s.find((v) => v >= q1 - 1.5 * iqr) ?? s[0];
          const wh = [...s].reverse().find((v) => v <= q3 + 1.5 * iqr) ?? s[s.length - 1];
          const cx = M.l + band * gi + band / 2;
          const bw = Math.min(56, band * 0.42);
          const mid = sy(q2);
          const ty = (v: number) => mid + (sy(v) - mid) * intro;
          const step = Math.max(1, Math.ceil(s.length / 400));
          return (
            <g key={g.label}>
              {g.values.filter((_, i) => i % step === 0).map((v, i) => {
                const jitter = ((((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * band * 0.7;
                return <circle key={i} cx={cx + jitter} cy={ty(v)} r={1.8} fill={g.color} opacity={0.28} />;
              })}
              <line x1={cx} x2={cx} y1={ty(wl)} y2={ty(wh)} stroke={INK} />
              <line x1={cx - bw / 4} x2={cx + bw / 4} y1={ty(wl)} y2={ty(wl)} stroke={INK} />
              <line x1={cx - bw / 4} x2={cx + bw / 4} y1={ty(wh)} y2={ty(wh)} stroke={INK} />
              <rect x={cx - bw / 2} width={bw} y={ty(q3)} height={Math.max(1, ty(q1) - ty(q3))} fill="#fff" fillOpacity={0.85} stroke={g.color} strokeWidth={2} rx={4} />
              <line x1={cx - bw / 2} x2={cx + bw / 2} y1={mid} y2={mid} stroke={INK} strokeWidth={2} />
              <text x={cx} y={M.t + innerH + 16} className="tick" textAnchor="middle">
                {g.label.length > 14 ? `${g.label.slice(0, 13)}…` : g.label}
              </text>
              <text x={cx} y={M.t + innerH + 30} className="tick dim" textAnchor="middle">n={s.length} · med {yf(q2)}</text>
            </g>
          );
        })}
        <text transform={`translate(14 ${M.t + innerH / 2}) rotate(-90)`} className="axis-label" textAnchor="middle">{label} →</text>
      </svg>
    </div>
  );
}

// ─── Correlation heatmap ────────────────────────────────────────────────────

export function Heatmap({
  columns,
  matrix,
  onPick,
}: {
  columns: string[];
  matrix: (number | null)[][];
  onPick?: (a: string, b: string) => void;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const intro = useIntro(columns.join('|'), 900);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const n = columns.length;
  const labelW = Math.min(150, Math.max(...columns.map((c) => c.length)) * 6.6 + 12);
  const cell = Math.max(10, Math.min(72, (width - labelW - 8) / n));
  const size = cell * n;
  const showVals = cell >= 30;

  return (
    <div ref={ref} className="viz-wrap" style={{ height: size + labelW + 10 }}>
      <svg className="viz-svg" width={width} height={size + labelW + 10} style={{ pointerEvents: 'all' }} onPointerLeave={() => setHover(null)}>
        <g transform={`translate(${labelW} 6)`}>
          {matrix.map((row, i) =>
            row.map((v, j) => {
              const delay = (i + j) / (2 * n);
              const t = Math.max(0, Math.min(1, (intro - delay * 0.6) / 0.4));
              const active = hover && (hover[0] === i || hover[1] === j);
              return (
                <g key={`${i}-${j}`} onPointerEnter={() => setHover([i, j])} onClick={() => i !== j && onPick?.(columns[i], columns[j])} style={{ cursor: i !== j ? 'pointer' : 'default' }}>
                  <rect
                    x={j * cell + (cell * (1 - t)) / 2}
                    y={i * cell + (cell * (1 - t)) / 2}
                    width={Math.max(0, cell * t - 1)}
                    height={Math.max(0, cell * t - 1)}
                    rx={cell > 20 ? 3 : 1}
                    fill={v === null ? '#eee' : divColor(v)}
                    stroke={hover && hover[0] === i && hover[1] === j ? INK : 'none'}
                    strokeWidth={1.5}
                    opacity={hover && !active ? 0.55 : 1}
                  />
                  {showVals && v !== null && t > 0.9 && (
                    <text x={j * cell + cell / 2} y={i * cell + cell / 2} className="cell-val" textAnchor="middle" dominantBaseline="middle" fill={Math.abs(v) > 0.55 ? '#fff' : INK}>
                      {v.toFixed(2).replace(/^(-?)0\./, '$1.')}
                    </text>
                  )}
                </g>
              );
            }),
          )}
          {columns.map((c, i) => (
            <text key={`r${c}`} x={-8} y={i * cell + cell / 2} className={`tick ${hover?.[0] === i ? 'hot' : ''}`} textAnchor="end" dominantBaseline="middle">
              {c.length > 20 ? `${c.slice(0, 19)}…` : c}
            </text>
          ))}
          {columns.map((c, j) => (
            <text
              key={`c${c}`}
              transform={`translate(${j * cell + cell / 2} ${size + 8}) rotate(45)`}
              className={`tick ${hover?.[1] === j ? 'hot' : ''}`}
            >
              {c.length > 20 ? `${c.slice(0, 19)}…` : c}
            </text>
          ))}
        </g>
      </svg>
      {hover && (
        <div className="viz-tip" style={{ left: Math.min(labelW + hover[1] * cell + cell + 10, width - 230), top: hover[0] * cell }}>
          <b>{columns[hover[0]]} × {columns[hover[1]]}</b>
          <span>r = {matrix[hover[0]][hover[1]]?.toFixed(3) ?? '—'}</span>
          {hover[0] !== hover[1] && <span className="dim">click to open scatter</span>}
        </div>
      )}
    </div>
  );
}

// ─── Missingness matrix ─────────────────────────────────────────────────────

export function MissingMatrix({
  columns,
  rows,
  isMissing,
  missingPct,
  sortRows,
}: {
  columns: string[];
  rows: number;
  isMissing: (col: string, row: number) => boolean;
  missingPct: Record<string, number>;
  sortRows: boolean;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverCol, setHoverCol] = useState<number | null>(null);
  const intro = useIntro(`${columns.join('|')}|${sortRows}`, 1000);
  const height = 420;
  const top = 70;
  const colW = width / Math.max(1, columns.length);

  const order = useMemo(() => {
    const idx = Array.from({ length: rows }, (_, i) => i);
    if (!sortRows) return idx;
    const score = idx.map((r) => columns.reduce((s, c) => s + (isMissing(c, r) ? 1 : 0), 0));
    return idx.sort((a, b) => score[b] - score[a]);
  }, [rows, columns, isMissing, sortRows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const ctx = setupCanvas(canvas, width, height);
    if (!ctx) return;
    const rowH = (height - top) / Math.max(1, rows);
    columns.forEach((c, ci) => {
      const x = ci * colW;
      const pct = missingPct[c] ?? 0;
      const barH = (top - 18) * (pct / 100) * intro;
      ctx.fillStyle = pct > 20 ? '#e5484d' : pct > 0 ? '#d98b06' : '#d6d2ca';
      ctx.fillRect(x + colW * 0.18, top - 8 - barH, colW * 0.64, Math.max(1, barH));
      ctx.fillStyle = ci === hoverCol ? '#efeaff' : '#f3f1ec';
      ctx.fillRect(x + 0.5, top, colW - 1, height - top);
    });
    const visibleRows = Math.ceil(rows * intro);
    ctx.fillStyle = INK;
    for (let k = 0; k < visibleRows; k++) {
      const r = order[k];
      const y = top + k * rowH;
      columns.forEach((c, ci) => {
        if (isMissing(c, r)) ctx.fillRect(ci * colW + 0.5, y, colW - 1, Math.max(1, rowH));
      });
    }
  }, [width, columns, rows, order, isMissing, missingPct, intro, hoverCol, colW]);

  return (
    <div ref={ref} className="viz-wrap" style={{ height: height + 90 }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas"
        style={{ height }}
        onPointerMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setHoverCol(Math.floor((e.clientX - rect.left) / colW));
        }}
        onPointerLeave={() => setHoverCol(null)}
      />
      <svg className="viz-svg" width={width} height={90} style={{ top: height }}>
        {columns.map((c, ci) =>
          colW >= 11 || ci === hoverCol ? (
            <text key={c} transform={`translate(${ci * colW + colW / 2} 8) rotate(45)`} className={`tick ${ci === hoverCol ? 'hot' : ''}`}>
              {c.length > 16 ? `${c.slice(0, 15)}…` : c}
            </text>
          ) : null,
        )}
      </svg>
      {hoverCol !== null && columns[hoverCol] && (
        <div className="viz-tip" style={{ left: Math.min(hoverCol * colW + colW + 6, width - 200), top: 6 }}>
          <b>{columns[hoverCol]}</b>
          <span>{(missingPct[columns[hoverCol]] ?? 0).toFixed(1)}% missing (full data)</span>
        </div>
      )}
    </div>
  );
}

// ─── Parallel coordinates ───────────────────────────────────────────────────

export function ParallelCoords({
  axes,
  values,
  colors,
  rows,
  height = 460,
}: {
  axes: string[];
  values: Record<string, Num[]>;
  colors: string[];
  rows: number;
  height?: number;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brush, setBrush] = useState<Record<string, [number, number]>>({});
  const dragging = useRef<{ axis: string; start: number } | null>(null);
  const intro = useIntro(axes.join('|'), 1100);
  const top = 30;
  const bottom = 30;
  const innerH = height - top - bottom;
  const left = 40;
  const gap = (width - left * 2) / Math.max(1, axes.length - 1);
  const ext = useMemo(() => Object.fromEntries(axes.map((a) => [a, extent(values[a] ?? [])])), [axes, values]);
  const normY = (a: string, v: number) => {
    const [lo, hi] = ext[a];
    return top + innerH - ((v - lo) / (hi - lo || 1)) * innerH;
  };

  useEffect(() => setBrush({}), [axes]);

  const selected = useMemo(() => {
    const keys = Object.keys(brush);
    return Array.from({ length: rows }, (_, r) =>
      keys.every((a) => {
        const v = values[a]?.[r];
        if (v === null || v === undefined) return false;
        const y = normY(a, v);
        const [b0, b1] = brush[a];
        return y >= Math.min(b0, b1) && y <= Math.max(b0, b1);
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brush, values, rows, ext, height]);
  const selectedCount = selected.filter(Boolean).length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width || axes.length < 2) return;
    const ctx = setupCanvas(canvas, width, height);
    if (!ctx) return;
    const reveal = left + (width - left * 2) * intro;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, reveal + 2, height);
    ctx.clip();
    const pass = (sel: boolean) => {
      for (let r = 0; r < rows; r++) {
        if (selected[r] !== sel) continue;
        ctx.strokeStyle = sel ? colors[r] ?? PALETTE[0] : OTHER_COLOR;
        ctx.globalAlpha = sel ? (selectedCount < 200 ? 0.6 : 0.22) : 0.05;
        ctx.beginPath();
        let started = false;
        axes.forEach((a, ai) => {
          const v = values[a]?.[r];
          if (v === null || v === undefined) {
            started = false;
            return;
          }
          const x = left + ai * gap;
          const y = normY(a, v);
          if (!started) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          started = true;
        });
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    };
    pass(false);
    pass(true);
    ctx.restore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, axes, values, colors, rows, selected, intro, height]);

  const axisAt = (mx: number) => {
    const ai = Math.round((mx - left) / gap);
    return ai >= 0 && ai < axes.length && Math.abs(left + ai * gap - mx) < 18 ? axes[ai] : null;
  };

  return (
    <div ref={ref} className="viz-wrap" style={{ height }}>
      <canvas ref={canvasRef} className="viz-canvas" />
      <svg
        className="viz-svg"
        width={width}
        height={height}
        style={{ pointerEvents: 'all', cursor: 'ns-resize' }}
        onPointerDown={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const a = axisAt(e.clientX - rect.left);
          if (!a) return;
          const y = e.clientY - rect.top;
          dragging.current = { axis: a, start: y };
          setBrush((b) => ({ ...b, [a]: [y, y] }));
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = dragging.current;
          if (!d) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const y = Math.max(top, Math.min(top + innerH, e.clientY - rect.top));
          setBrush((b) => ({ ...b, [d.axis]: [d.start, y] }));
        }}
        onPointerUp={() => {
          const d = dragging.current;
          dragging.current = null;
          if (d) setBrush((b) => {
            const r = b[d.axis];
            if (r && Math.abs(r[1] - r[0]) < 4) {
              const { [d.axis]: _, ...rest } = b;
              return rest;
            }
            return b;
          });
        }}
      >
        {axes.map((a, ai) => {
          const x = left + ai * gap;
          const b = brush[a];
          return (
            <g key={a}>
              <line x1={x} x2={x} y1={top} y2={top + innerH} stroke={INK} strokeOpacity={0.55} />
              <text x={x} y={top - 12} className="tick hot" textAnchor="middle">{a.length > 16 ? `${a.slice(0, 15)}…` : a}</text>
              <text x={x + 5} y={top + 4} className="tick dim">{fmt(ext[a][1])}</text>
              <text x={x + 5} y={top + innerH} className="tick dim">{fmt(ext[a][0])}</text>
              {b && (
                <rect x={x - 7} width={14} y={Math.min(b[0], b[1])} height={Math.abs(b[1] - b[0])} rx={3} fill="rgba(109,74,255,.18)" stroke="#6d4aff" />
              )}
            </g>
          );
        })}
      </svg>
      <div className="viz-hint">
        {Object.keys(brush).length
          ? `${selectedCount} of ${rows} rows selected · click an axis to clear`
          : 'drag along any axis to filter'}
      </div>
    </div>
  );
}

// ─── Minis for the overview grid ────────────────────────────────────────────

export function MiniHist({ values, color = INK }: { values: number[]; color?: string }) {
  const bins = useMemo(() => histogram(values, 24, extent(values)), [values]);
  const max = Math.max(1, ...bins.map((b) => b.count));
  return (
    <svg viewBox="0 0 120 40" preserveAspectRatio="none" className="mini">
      {bins.map((b, i) => (
        <rect key={i} x={i * 5} width={4.2} y={40 - (b.count / max) * 38} height={(b.count / max) * 38} fill={color} className="mini-bar" style={{ ['--i' as string]: i }} />
      ))}
    </svg>
  );
}

export function MiniBars({ items }: { items: { value: string; count: number }[] }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div className="mini-cats">
      {items.slice(0, 4).map((it, i) => (
        <div key={it.value}>
          <span>{it.value}</span>
          <i style={{ width: `${(it.count / max) * 100}%`, background: PALETTE[i % PALETTE.length], ['--i' as string]: i } as React.CSSProperties} />
        </div>
      ))}
    </div>
  );
}

export function Legend({ items, range, label }: { items: { label: string; color: string; count: number }[]; range?: [number, number]; label?: string }) {
  if (range) {
    return (
      <div className="legend">
        <span className="dim">{label}</span>
        <span>{fmt(range[0])}</span>
        <span className="ramp" />
        <span>{fmt(range[1])}</span>
      </div>
    );
  }
  if (!items.length) return null;
  return (
    <div className="legend">
      {label && <span className="dim">{label}</span>}
      {items.map((it) => (
        <span key={it.label}><i style={{ background: it.color }} />{it.label} <span className="dim">{it.count}</span></span>
      ))}
    </div>
  );
}
