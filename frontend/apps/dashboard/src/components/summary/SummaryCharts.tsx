'use client';

import React, { useMemo, useState } from 'react';
import { SummaryBox, SummaryHist } from '../../services/api';
import { useInView } from '../ui';
import { fmt, PALETTE } from '../viz/lib';

/** Donut that draws its arcs in when scrolled into view. */
export function Donut({ segments, size = 180, label, sub }: {
  segments: { label: string; value: number; color: string }[];
  size?: number;
  label?: React.ReactNode;
  sub?: string;
}) {
  const [ref, inView] = useInView<HTMLDivElement>(0.3);
  const [hover, setHover] = useState<number | null>(null);
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  const r = size / 2 - 14;
  const c = 2 * Math.PI * r;
  let acc = 0;
  return (
    <div ref={ref} className={`donut ${inView ? 'in' : ''}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line-soft)" strokeWidth={18} />
        {segments.map((s, i) => {
          const len = (s.value / total) * c;
          const el = (
            <circle
              key={s.label}
              cx={size / 2}
              cy={size / 2}
              r={r}
              fill="none"
              stroke={s.color}
              strokeWidth={hover === i ? 24 : 18}
              strokeDasharray={`${inView ? Math.max(0, len - 2) : 0} ${c}`}
              strokeDashoffset={-acc}
              style={{ transition: `stroke-dasharray 1.1s var(--ease-out) ${i * 140}ms, stroke-width .25s ease` }}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          );
          acc += len;
          return el;
        })}
      </svg>
      <div className="donut-center">
        {hover !== null ? (
          <>
            <b>{Math.round((segments[hover].value / total) * 100)}%</b>
            <span>{segments[hover].label}</span>
          </>
        ) : (
          <>
            <b>{label}</b>
            {sub && <span>{sub}</span>}
          </>
        )}
      </div>
    </div>
  );
}

/** Horizontal bars that grow in sequence. */
export function BarList({ items, max, unit = '', colorFor, compact }: {
  items: { label: string; value: number; display?: string; note?: string; color?: string }[];
  max?: number;
  unit?: string;
  colorFor?: (i: number) => string;
  compact?: boolean;
}) {
  const [ref, inView] = useInView<HTMLDivElement>(0.2);
  const top = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div ref={ref} className={`barlist ${compact ? 'compact' : ''} ${inView ? 'in' : ''}`}>
      {items.map((it, i) => (
        <div key={`${it.label}-${i}`} className="barlist-row">
          <span className="barlist-label" title={it.label}>{it.label}</span>
          <span className="barlist-track">
            <i style={{ ['--w' as string]: `${Math.min(100, (Math.abs(it.value) / top) * 100)}%`, ['--d' as string]: `${i * 70}ms`, background: it.color ?? colorFor?.(i) ?? 'var(--ink)' }} />
          </span>
          <span className="barlist-val mono">{it.display ?? `${fmt(it.value)}${unit}`}{it.note && <em> {it.note}</em>}</span>
        </div>
      ))}
    </div>
  );
}

/** Histogram (binned or discrete) with bars rising from the baseline. */
export function HistBars({ hist, color = 'var(--ink)', height = 120, highlight }: {
  hist: SummaryHist;
  color?: string;
  height?: number;
  highlight?: [number, number] | null;
}) {
  const [ref, inView] = useInView<HTMLDivElement>(0.25);
  const max = Math.max(...hist.counts, 1);
  const n = hist.counts.length;
  const labels = hist.discrete
    ? (hist.values || []).map((v) => fmt(v))
    : hist.counts.map((_, i) => `${fmt(hist.edges![i])} – ${fmt(hist.edges![i + 1])}`);
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div ref={ref} className={`histbars ${inView ? 'in' : ''}`} style={{ height }}>
      <div className="histbars-plot">
        {hist.counts.map((c, i) => {
          const center = hist.discrete ? (hist.values || [])[i] : (hist.edges![i] + hist.edges![i + 1]) / 2;
          const inBox = highlight && center >= highlight[0] && center <= highlight[1];
          return (
            <span
              key={i}
              className="histbars-bar"
              style={{
                height: `${(c / max) * 100}%`,
                background: color,
                opacity: highlight ? (inBox ? 1 : 0.45) : 1,
                transitionDelay: `${Math.min(i, 40) * 18}ms`,
                width: `${100 / n}%`,
              }}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          );
        })}
      </div>
      {hover !== null && (
        <div className="histbars-tip mono">{labels[hover]}: <b>{hist.counts[hover].toLocaleString()}</b></div>
      )}
      <div className="histbars-axis mono">
        <span>{hist.discrete ? labels[0] : fmt(hist.edges![0])}</span>
        <span>{hist.discrete ? labels[labels.length - 1] : fmt(hist.edges![n])}</span>
      </div>
    </div>
  );
}

/** Compact box plot: whiskers, box (IQR) and median, scaled to the column's min..max. */
export function BoxStrip({ box }: { box: SummaryBox }) {
  const [ref, inView] = useInView<HTMLDivElement>(0.3);
  const span = box.max - box.min || 1;
  const x = (v: number) => `${((v - box.min) / span) * 100}%`;
  return (
    <div ref={ref} className={`boxstrip ${inView ? 'in' : ''}`} title={`q1 ${fmt(box.q1)} · median ${fmt(box.median)} · q3 ${fmt(box.q3)}`}>
      <span className="bs-whisker" style={{ left: x(box.lo), width: `calc(${x(box.hi)} - ${x(box.lo)})` }} />
      <span className="bs-box" style={{ left: x(box.q1), width: `calc(${x(box.q3)} - ${x(box.q1)})` }} />
      <span className="bs-median" style={{ left: x(box.median) }} />
      {box.outliers > 0 && <span className="bs-out mono">{box.outliers} outlier{box.outliers === 1 ? '' : 's'}</span>}
    </div>
  );
}

/** Area chart for counts over time; the line draws itself in. */
export function Timeline({ points, height = 120, color = 'var(--accent)' }: { points: { t: string; count: number }[]; height?: number; color?: string }) {
  const [ref, inView] = useInView<HTMLDivElement>(0.3);
  const w = 600;
  const max = Math.max(...points.map((p) => p.count), 1);
  const path = useMemo(() => {
    if (points.length < 2) return { line: '', area: '' };
    const xy = points.map((p, i) => [(i / (points.length - 1)) * w, height - 6 - (p.count / max) * (height - 16)]);
    const line = xy.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    return { line, area: `${line} L${w},${height} L0,${height} Z` };
  }, [points, height, max]);
  return (
    <div ref={ref} className={`timeline ${inView ? 'in' : ''}`}>
      <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height }}>
        <defs>
          <linearGradient id="tl-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.28} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <path d={path.area} fill="url(#tl-fill)" className="tl-area" />
        <path d={path.line} fill="none" stroke={color} strokeWidth={2} className="tl-line" pathLength={1} />
      </svg>
      <div className="histbars-axis mono">
        <span>{points[0]?.t.slice(0, 10)}</span>
        <span>{points[points.length - 1]?.t.slice(0, 10)}</span>
      </div>
    </div>
  );
}

/** Score ring 0..100 with a counting number. */
export function Ring({ value, size = 128, color }: { value: number; size?: number; color?: string }) {
  const [ref, inView] = useInView<HTMLDivElement>(0.3);
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  const tone = color ?? (value >= 85 ? 'var(--good)' : value >= 65 ? 'var(--accent)' : value >= 50 ? 'var(--warn)' : 'var(--bad)');
  return (
    <div ref={ref} className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--line-soft)" strokeWidth={8} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={tone}
          strokeWidth={8}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={inView ? c - (Math.max(0, Math.min(100, value)) / 100) * c : c}
          style={{ transition: 'stroke-dashoffset 1.6s var(--ease-out)' }}
        />
      </svg>
      <b style={{ color: tone }}>{Math.round(value * 10) / 10}</b>
    </div>
  );
}

export const kindColor: Record<string, string> = {
  numeric: PALETTE[0],
  categorical: PALETTE[1],
  datetime: PALETTE[3],
  text: PALETTE[4],
  id: '#b9b5ad',
};
