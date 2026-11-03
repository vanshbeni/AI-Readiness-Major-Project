'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { extent, fmt, fmtDate, linear, niceTicks, pad, regression, setupCanvas, useIntro, useSize, type Num } from './lib';

const M = { l: 58, r: 18, t: 18, b: 44 };

interface Scatter2DProps {
  x: Num[];
  y: Num[];
  colors: string[];
  sizes?: Num[];
  xLabel: string;
  yLabel: string;
  xIsDate?: boolean;
  yIsDate?: boolean;
  height?: number;
  trend?: boolean;
  tooltip: (i: number) => React.ReactNode;
}

type View = [number, number, number, number];

export function Scatter2D({ x, y, colors, sizes, xLabel, yLabel, xIsDate, yIsDate, height = 480, trend, tooltip }: Scatter2DProps) {
  const [wrapRef, { width }] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intro = useIntro(`${xLabel}|${yLabel}|${x.length}`);

  const base = useMemo<View>(() => {
    const [x0, x1] = pad(extent(x));
    const [y0, y1] = pad(extent(y));
    return [x0, x1, y0, y1];
  }, [x, y]);
  const [view, setView] = useState<View>(base);
  useEffect(() => setView(base), [base]);
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ px: number; py: number; view: View } | null>(null);

  const sizeScale = useMemo(() => {
    if (!sizes) return null;
    const [lo, hi] = extent(sizes);
    return (v: Num) => (v === null ? 2 : 2 + ((v - lo) / (hi - lo || 1)) * 8);
  }, [sizes]);

  const innerW = Math.max(10, width - M.l - M.r);
  const innerH = height - M.t - M.b;
  const sx = linear(view[0], view[1], M.l, M.l + innerW);
  const sy = linear(view[2], view[3], M.t + innerH, M.t);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const ctx = setupCanvas(canvas, width, height);
    if (!ctx) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(M.l, M.t, innerW, innerH);
    ctx.clip();
    const cx = M.l + innerW / 2;
    const cy = M.t + innerH / 2;
    for (let i = 0; i < x.length; i++) {
      const xv = x[i];
      const yv = y[i];
      if (xv === null || yv === null) continue;
      const px = cx + (sx(xv) - cx) * intro;
      const py = cy + (sy(yv) - cy) * intro;
      const r = (sizeScale ? sizeScale(sizes![i]) : 3.2) * (0.4 + 0.6 * intro);
      ctx.globalAlpha = hover === null ? 0.72 : i === hover ? 1 : 0.35;
      ctx.fillStyle = colors[i] ?? '#6d4aff';
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
    if (hover !== null && x[hover] !== null && y[hover] !== null) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#141414';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx(x[hover]!), sy(y[hover]!), 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [x, y, colors, sizes, width, height, view, intro, hover]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      setView((v) => {
        const fx = linear(v[0], v[1], M.l, M.l + innerW).invert(mx);
        const fy = linear(v[2], v[3], M.t + innerH, M.t).invert(my);
        return [fx + (v[0] - fx) * f, fx + (v[1] - fx) * f, fy + (v[2] - fy) * f, fy + (v[3] - fy) * f];
      });
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [innerW, innerH]);

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    if (drag.current) {
      const d = drag.current;
      const dx = ((mx - d.px) / innerW) * (d.view[1] - d.view[0]);
      const dy = ((my - d.py) / innerH) * (d.view[3] - d.view[2]);
      setView([d.view[0] - dx, d.view[1] - dx, d.view[2] + dy, d.view[3] + dy]);
      return;
    }
    let best = -1;
    let bestD = 144;
    for (let i = 0; i < x.length; i++) {
      if (x[i] === null || y[i] === null) continue;
      const d = (sx(x[i]!) - mx) ** 2 + (sy(y[i]!) - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    setHover(best >= 0 ? best : null);
  };

  const line = trend ? regression(x, y) : null;
  const xt = niceTicks(view[0], view[1], Math.max(3, Math.floor(innerW / 110)));
  const yt = niceTicks(view[2], view[3], 6);
  const xf = xIsDate ? fmtDate : fmt;
  const yf = yIsDate ? fmtDate : fmt;
  const zoomed = view.some((v, i) => Math.abs(v - base[i]) > 1e-9);
  const hx = hover !== null ? sx(x[hover]!) : 0;
  const hy = hover !== null ? sy(y[hover]!) : 0;

  return (
    <div ref={wrapRef} className="viz-wrap" style={{ height }}>
      <svg className="viz-svg" width={width} height={height}>
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={M.l} x2={M.l + innerW} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={M.l - 8} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">{yf(t)}</text>
          </g>
        ))}
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line y1={M.t} y2={M.t + innerH} x1={sx(t)} x2={sx(t)} className="grid" />
            <text y={M.t + innerH + 16} x={sx(t)} className="tick" textAnchor="middle">{xf(t)}</text>
          </g>
        ))}
        <text x={M.l + innerW / 2} y={height - 6} className="axis-label" textAnchor="middle">{xLabel} →</text>
        <text transform={`translate(14 ${M.t + innerH / 2}) rotate(-90)`} className="axis-label" textAnchor="middle">{yLabel} →</text>
        {line && (
          <g style={{ opacity: intro }}>
            <clipPath id="trend-clip"><rect x={M.l} y={M.t} width={innerW} height={innerH} /></clipPath>
            <line
              clipPath="url(#trend-clip)"
              x1={sx(view[0])}
              y1={sy(line.intercept + line.slope * view[0])}
              x2={sx(view[1])}
              y2={sy(line.intercept + line.slope * view[1])}
              className="trend"
            />
            <text x={M.l + innerW - 6} y={M.t + 14} textAnchor="end" className="trend-label">r = {line.r.toFixed(3)}</text>
          </g>
        )}
        {hover !== null && (
          <g className="crosshair">
            <line x1={M.l} x2={M.l + innerW} y1={hy} y2={hy} />
            <line y1={M.t} y2={M.t + innerH} x1={hx} x2={hx} />
          </g>
        )}
      </svg>
      <canvas
        ref={canvasRef}
        className="viz-canvas"
        style={{ cursor: drag.current ? 'grabbing' : 'crosshair' }}
        onPointerDown={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          drag.current = { px: e.clientX - rect.left, py: e.clientY - rect.top, view };
          e.currentTarget.setPointerCapture(e.pointerId);
          setHover(null);
        }}
        onPointerUp={() => { drag.current = null; }}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => setView(base)}
      />
      {hover !== null && (
        <div className="viz-tip" style={{ left: Math.min(hx + 14, width - 220), top: Math.max(hy - 10, 4) }}>
          {tooltip(hover)}
        </div>
      )}
      <div className="viz-hint">{zoomed ? 'double-click to reset' : 'scroll to zoom · drag to pan'}</div>
    </div>
  );
}

// ─── 3D ─────────────────────────────────────────────────────────────────────

interface Scatter3DProps {
  x: Num[];
  y: Num[];
  z: Num[];
  colors: string[];
  labels: [string, string, string];
  height?: number;
  autoRotate?: boolean;
  tooltip: (i: number) => React.ReactNode;
}

const CUBE_EDGES: [number, number][] = [
  [0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];
const CUBE = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => [i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);

export function Scatter3D({ x, y, z, colors, labels, height = 520, autoRotate = true, tooltip }: Scatter3DProps) {
  const [wrapRef, { width }] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cam = useRef({ yaw: -0.7, pitch: 0.42, zoom: 1, vy: 0, vp: 0, dragging: false, lx: 0, ly: 0, intro: 0 });
  const projected = useRef<{ i: number; px: number; py: number; depth: number }[]>([]);
  const [hover, setHover] = useState<{ i: number; px: number; py: number } | null>(null);
  const hoverRef = useRef<number | null>(null);
  const autoRef = useRef(autoRotate);
  autoRef.current = autoRotate;

  const norm = useMemo(() => {
    const ex = extent(x);
    const ey = extent(y);
    const ez = extent(z);
    const n = (v: Num, [lo, hi]: [number, number]) => (v === null ? null : ((v - lo) / (hi - lo || 1)) * 2 - 1);
    return { pts: x.map((_, i) => [n(x[i], ex), n(y[i], ey), n(z[i], ez)] as Num[]), ex, ey, ez };
  }, [x, y, z]);

  useEffect(() => {
    cam.current.intro = 0;
  }, [x, y, z]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    let raf = 0;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const frame = () => {
      const c = cam.current;
      if (!c.dragging) {
        c.yaw += c.vy;
        c.pitch = Math.max(-1.45, Math.min(1.45, c.pitch + c.vp));
        c.vy *= 0.94;
        c.vp *= 0.94;
        if (autoRef.current && !reduced && Math.abs(c.vy) < 0.002) c.yaw += 0.0035;
      }
      c.intro = reduced ? 1 : Math.min(1, c.intro + 0.025);
      const t = 1 - Math.pow(1 - c.intro, 3);

      const ctx = setupCanvas(canvas, width, height);
      if (!ctx) return;
      const cosY = Math.cos(c.yaw);
      const sinY = Math.sin(c.yaw);
      const cosP = Math.cos(c.pitch);
      const sinP = Math.sin(c.pitch);
      const scale = Math.min(width, height) * 0.27 * c.zoom;
      const dist = 6;
      const proj = (p: number[]) => {
        const x1 = p[0] * cosY - p[2] * sinY;
        const z1 = p[0] * sinY + p[2] * cosY;
        const y2 = p[1] * cosP - z1 * sinP;
        const z2 = p[1] * sinP + z1 * cosP;
        const f = scale * (dist / (dist + z2));
        return { px: width / 2 + x1 * f, py: height / 2 - y2 * f, depth: z2, f };
      };

      // floor grid on y = -1
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(20,20,20,0.06)';
      for (let g = -1; g <= 1.0001; g += 0.5) {
        const a = proj([g, -1, -1]);
        const b = proj([g, -1, 1]);
        const c2 = proj([-1, -1, g]);
        const d = proj([1, -1, g]);
        ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c2.px, c2.py); ctx.lineTo(d.px, d.py); ctx.stroke();
      }
      // cube
      ctx.strokeStyle = 'rgba(20,20,20,0.16)';
      for (const [a, b] of CUBE_EDGES) {
        const pa = proj(CUBE[a]);
        const pb = proj(CUBE[b]);
        ctx.beginPath(); ctx.moveTo(pa.px, pa.py); ctx.lineTo(pb.px, pb.py); ctx.stroke();
      }
      // axes from the (-1,-1,-1) corner
      const origin = proj([-1, -1, -1]);
      const axes: [number[], string, string][] = [
        [[1.25, -1, -1], labels[0], '#e5484d'],
        [[-1, 1.25, -1], labels[1], '#0f9f6e'],
        [[-1, -1, 1.25], labels[2], '#6d4aff'],
      ];
      ctx.font = '11px "Geist Mono", monospace';
      for (const [end, label, color] of axes) {
        const pe = proj(end);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(origin.px, origin.py); ctx.lineTo(pe.px, pe.py); ctx.stroke();
        ctx.fillStyle = color;
        const text = label.length > 22 ? `${label.slice(0, 21)}…` : label;
        const tx = Math.min(Math.max(4, pe.px + 4), width - ctx.measureText(text).width - 4);
        ctx.fillText(text, tx, Math.min(Math.max(14, pe.py - 4), height - 6));
      }

      // points, far to near
      const out: { i: number; px: number; py: number; depth: number; f: number }[] = [];
      norm.pts.forEach((p, i) => {
        if (p[0] === null || p[1] === null || p[2] === null) return;
        const q = proj([p[0] * t, p[1] * t, p[2] * t]);
        out.push({ i, ...q });
      });
      out.sort((a, b) => b.depth - a.depth);
      const hv = hoverRef.current;
      for (const o of out) {
        const fog = 0.35 + 0.65 * ((1 - o.depth) / 2);
        ctx.globalAlpha = hv === null ? Math.max(0.2, Math.min(1, fog)) : o.i === hv ? 1 : 0.18;
        ctx.fillStyle = colors[o.i] ?? '#6d4aff';
        const r = Math.max(1.4, (o.f / scale) * 3.4 * c.zoom ** 0.3);
        ctx.beginPath();
        ctx.arc(o.px, o.py, r, 0, Math.PI * 2);
        ctx.fill();
        if (o.i === hv) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = '#141414';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(o.px, o.py, r + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      projected.current = out;
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [width, height, norm, colors, labels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const c = cam.current;
      c.zoom = Math.max(0.4, Math.min(4, c.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  const pick = (mx: number, my: number) => {
    let best: { i: number; px: number; py: number } | null = null;
    let bestD = 100;
    for (const o of projected.current) {
      const d = (o.px - mx) ** 2 + (o.py - my) ** 2;
      if (d <= bestD) {
        bestD = d;
        best = o;
      }
    }
    hoverRef.current = best?.i ?? null;
    setHover(best);
  };

  return (
    <div ref={wrapRef} className="viz-wrap" style={{ height }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas"
        style={{ cursor: cam.current.dragging ? 'grabbing' : 'grab' }}
        onPointerDown={(e) => {
          const c = cam.current;
          c.dragging = true;
          c.lx = e.clientX;
          c.ly = e.clientY;
          c.vy = 0;
          c.vp = 0;
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const c = cam.current;
          if (c.dragging) {
            const dx = (e.clientX - c.lx) * 0.008;
            const dy = (e.clientY - c.ly) * 0.008;
            c.yaw += dx;
            c.pitch = Math.max(-1.45, Math.min(1.45, c.pitch + dy));
            c.vy = dx;
            c.vp = dy;
            c.lx = e.clientX;
            c.ly = e.clientY;
            return;
          }
          const rect = e.currentTarget.getBoundingClientRect();
          pick(e.clientX - rect.left, e.clientY - rect.top);
        }}
        onPointerUp={() => { cam.current.dragging = false; }}
        onPointerLeave={() => { hoverRef.current = null; setHover(null); }}
        onDoubleClick={() => Object.assign(cam.current, { yaw: -0.7, pitch: 0.42, zoom: 1, vy: 0, vp: 0 })}
      />
      {hover && (
        <div className="viz-tip" style={{ left: Math.min(hover.px + 14, width - 220), top: Math.max(hover.py - 10, 4) }}>
          {tooltip(hover.i)}
        </div>
      )}
      <div className="viz-hint">drag to orbit · scroll to zoom · double-click to reset</div>
      <div className="viz-axes-key">
        <span style={{ color: '#e5484d' }}>x {labels[0]} [{fmt(norm.ex[0])}, {fmt(norm.ex[1])}]</span>
        <span style={{ color: '#0f9f6e' }}>y {labels[1]} [{fmt(norm.ey[0])}, {fmt(norm.ey[1])}]</span>
        <span style={{ color: '#6d4aff' }}>z {labels[2]} [{fmt(norm.ez[0])}, {fmt(norm.ez[1])}]</span>
      </div>
    </div>
  );
}
