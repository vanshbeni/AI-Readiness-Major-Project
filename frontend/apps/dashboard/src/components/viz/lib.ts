import { useEffect, useRef, useState, type RefObject } from 'react';
import type { ExploreColumn, ExplorePayload } from '../../services/api';

export type Num = number | null;

export const PALETTE = ['#6d4aff', '#0f9f6e', '#e5484d', '#d98b06', '#1f8fd6', '#c2409a', '#00a3a3', '#8a5a44', '#5b6b00'];
export const OTHER_COLOR = '#b9b5ad';
export const INK = '#141414';

export function useSize<T extends HTMLElement>(): [RefObject<T>, { width: number; height: number }] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize((s) => (Math.abs(s.width - width) < 1 && Math.abs(s.height - height) < 1 ? s : { width, height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/** Animation progress 0 → 1 that restarts whenever `key` changes. */
export function useIntro(key: unknown, duration = 700): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setT(1);
      return;
    }
    let raf = 0;
    let start = 0;
    setT(0);
    const step = (now: number) => {
      if (!start) start = now;
      const p = Math.min(1, (now - start) / duration);
      setT(1 - Math.pow(1 - p, 3));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return t;
}

export function linear(d0: number, d1: number, r0: number, r1: number) {
  const span = d1 - d0 || 1;
  const f = (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
  f.invert = (px: number) => d0 + ((px - r0) / (r1 - r0 || 1)) * span;
  return f;
}

export function extent(values: Num[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

export function pad([lo, hi]: [number, number], frac = 0.05): [number, number] {
  const p = (hi - lo) * frac;
  return [lo - p, hi + p];
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

export function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return Number(v.toFixed(2)).toString();
  if (a === 0) return '0';
  return Number(v.toPrecision(2)).toString();
}

export function fmtDate(ms: number): string {
  const d = new Date(ms);
  return d.toISOString().slice(0, 10);
}

export function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / (values.length || 1);
}

export function quantile(sorted: number[], p: number): number {
  if (!sorted.length) return NaN;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function finite(values: Num[]): number[] {
  return values.filter((v): v is number => v !== null && Number.isFinite(v));
}

export interface Bin { x0: number; x1: number; count: number }

export function histogram(values: number[], bins: number, domain: [number, number]): Bin[] {
  const [lo, hi] = domain;
  const width = (hi - lo) / bins || 1;
  const out: Bin[] = Array.from({ length: bins }, (_, i) => ({ x0: lo + i * width, x1: lo + (i + 1) * width, count: 0 }));
  for (const v of values) {
    if (v < lo || v > hi) continue;
    const i = Math.min(bins - 1, Math.floor((v - lo) / width));
    out[i].count++;
  }
  return out;
}

/** Gaussian kernel density, scaled to match a histogram of `binWidth`. */
export function kde(values: number[], xs: number[], binWidth: number): number[] {
  const n = values.length;
  if (n < 2) return xs.map(() => 0);
  const sorted = [...values].sort((a, b) => a - b);
  const m = mean(values);
  const sd = Math.sqrt(mean(values.map((v) => (v - m) ** 2)));
  const iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25);
  const h = 0.9 * Math.min(sd, iqr / 1.34 || sd) * Math.pow(n, -0.2) || binWidth;
  const k = 1 / (Math.sqrt(2 * Math.PI) * h);
  return xs.map((x) => {
    let s = 0;
    for (const v of values) s += Math.exp(-0.5 * ((x - v) / h) ** 2);
    return k * s * binWidth;
  });
}

export function sturges(n: number): number {
  return Math.max(8, Math.min(60, Math.ceil(Math.log2(Math.max(n, 2)) + 1) * 2));
}

/** Least-squares line and Pearson r for paired values. */
export function regression(xs: Num[], ys: Num[]): { slope: number; intercept: number; r: number } | null {
  const px: number[] = [];
  const py: number[] = [];
  xs.forEach((x, i) => {
    const y = ys[i];
    if (x !== null && y !== null && Number.isFinite(x) && Number.isFinite(y)) {
      px.push(x);
      py.push(y);
    }
  });
  if (px.length < 3) return null;
  const mx = mean(px);
  const my = mean(py);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < px.length; i++) {
    sxy += (px[i] - mx) * (py[i] - my);
    sxx += (px[i] - mx) ** 2;
    syy += (py[i] - my) ** 2;
  }
  if (!sxx || !syy) return null;
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx, r: sxy / Math.sqrt(sxx * syy) };
}

function mix(a: [number, number, number], b: [number, number, number], t: number): string {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Sequential ramp: paper → violet → ink. */
export function seqColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  return x < 0.6 ? mix([233, 228, 255], [109, 74, 255], x / 0.6) : mix([109, 74, 255], [26, 16, 64], (x - 0.6) / 0.4);
}

/** Diverging ramp for -1..1: red → paper → violet. */
export function divColor(v: number): string {
  const x = Math.max(-1, Math.min(1, v));
  return x < 0 ? mix([248, 247, 244], [229, 72, 77], -x) : mix([248, 247, 244], [109, 74, 255], x);
}

// ─── Column access ──────────────────────────────────────────────────────────

export function numericValues(p: ExplorePayload, col: string): Num[] {
  const raw = p.data[col] ?? [];
  return raw.map((v) => {
    if (v === null) return null;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  });
}

export function categoryValues(p: ExplorePayload, col: string): (string | null)[] {
  if (col === CLUSTER_KEY) return p.projection?.clusters ?? [];
  return (p.data[col] ?? []).map((v) => (v === null ? null : String(v)));
}

export const CLUSTER_KEY = '⟨k-means cluster⟩';

export function isAxisColumn(c: ExploreColumn): boolean {
  return c.included && (c.kind === 'numeric' || c.kind === 'datetime' || c.numeric_like);
}

export function isGroupColumn(c: ExploreColumn): boolean {
  return c.included && (c.kind === 'categorical' || (c.kind === 'text' && c.unique <= 40));
}

export interface ColorEncoding {
  kind: 'none' | 'category' | 'numeric';
  /** Per-row CSS color. */
  colors: string[];
  legend: { label: string; color: string; count: number }[];
  range?: [number, number];
}

export function encodeColor(p: ExplorePayload, col: string | null, maxGroups = 8): ColorEncoding {
  const n = p.sample_rows;
  if (!col) return { kind: 'none', colors: Array(n).fill(PALETTE[0]), legend: [] };
  const meta = p.columns.find((c) => c.name === col);
  if (meta && meta.kind === 'numeric') {
    const vals = numericValues(p, col);
    const [lo, hi] = extent(vals);
    return {
      kind: 'numeric',
      colors: vals.map((v) => (v === null ? OTHER_COLOR : seqColor((v - lo) / (hi - lo || 1)))),
      legend: [],
      range: [lo, hi],
    };
  }
  const vals = categoryValues(p, col);
  const counts = new Map<string, number>();
  vals.forEach((v) => counts.set(v ?? 'null', (counts.get(v ?? 'null') ?? 0) + 1));
  const ranked = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const top = ranked.slice(0, maxGroups).map(([k]) => k);
  const colorOf = new Map(top.map((k, i) => [k, k === 'null' ? OTHER_COLOR : PALETTE[i % PALETTE.length]]));
  const otherCount = ranked.slice(maxGroups).reduce((s, [, c]) => s + c, 0);
  return {
    kind: 'category',
    colors: vals.map((v) => colorOf.get(v ?? 'null') ?? OTHER_COLOR),
    legend: [
      ...top.map((k) => ({ label: k, color: colorOf.get(k)!, count: counts.get(k)! })),
      ...(otherCount ? [{ label: 'other', color: OTHER_COLOR, count: otherCount }] : []),
    ],
  };
}

export function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}
