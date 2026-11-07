import React from 'react';
import { AgentChart as Spec } from '../../services/api';
import { PALETTE, extent, fmt, linear, niceTicks, pad, useSize } from '../viz/lib';

const M = { l: 52, r: 14, t: 12, b: 44 };

/** Renders a chart spec produced by the Analyst agent (bar / line / scatter over a result table). */
export function AgentChart({ spec, height = 260 }: { spec: Spec; height?: number }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const xi = spec.columns.indexOf(spec.x);
  const yis = spec.y.map((y) => spec.columns.indexOf(y)).filter((i) => i >= 0);
  const rows = spec.rows.slice(0, 60);
  const ys = rows.flatMap((r) => yis.map((i) => (typeof r[i] === 'number' ? (r[i] as number) : null)));
  const [y0, y1] = spec.type === 'bar' ? [Math.min(0, extent(ys)[0]), extent(ys)[1]] : pad(extent(ys));
  const innerW = Math.max(10, width - M.l - M.r);
  const innerH = height - M.t - M.b;
  const sy = linear(y0, y1 * (spec.type === 'bar' ? 1.08 : 1), M.t + innerH, M.t);
  const numericX = rows.every((r) => typeof r[xi] === 'number');
  const xs = rows.map((r) => r[xi]);
  const sxNum = numericX ? linear(...pad(extent(xs as number[])), M.l, M.l + innerW) : null;
  const band = innerW / Math.max(1, rows.length);
  const xPos = (i: number) => (sxNum ? sxNum(xs[i] as number) : M.l + band * i + band / 2);

  return (
    <div ref={ref} className="viz-wrap" style={{ height }}>
      <svg className="viz-svg" width={width} height={height}>
        {niceTicks(y0, y1, 5).map((t) => (
          <g key={t}>
            <line x1={M.l} x2={M.l + innerW} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={M.l - 8} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">{fmt(t)}</text>
          </g>
        ))}
        {spec.type === 'bar' && rows.map((r, i) =>
          yis.map((yi, k) => {
            const v = typeof r[yi] === 'number' ? (r[yi] as number) : 0;
            const w = (band * 0.7) / yis.length;
            const x = M.l + band * i + band * 0.15 + w * k;
            return <rect key={`${i}-${k}`} x={x} width={Math.max(1, w - 1)} y={Math.min(sy(v), sy(0))} height={Math.abs(sy(0) - sy(v))}
              fill={PALETTE[k % PALETTE.length]} rx={2} className="agent-bar" style={{ ['--i' as string]: i }} />;
          }),
        )}
        {spec.type === 'line' && yis.map((yi, k) => (
          <path key={k} fill="none" stroke={PALETTE[k % PALETTE.length]} strokeWidth={2} pathLength={1} className="draw-path"
            d={rows.map((r, i) => `${i ? 'L' : 'M'}${xPos(i)},${sy(Number(r[yi]) || 0)}`).join('')} />
        ))}
        {spec.type === 'scatter' && rows.map((r, i) => yis.map((yi, k) => (
          <circle key={`${i}-${k}`} cx={xPos(i)} cy={sy(Number(r[yi]) || 0)} r={3.5} fill={PALETTE[k % PALETTE.length]} opacity={0.75} />
        )))}
        {!sxNum && rows.length <= 16 && rows.map((r, i) => (
          <text key={i} x={xPos(i)} y={M.t + innerH + 14} className="tick" textAnchor="middle">
            {String(r[xi] ?? '—').slice(0, 12)}
          </text>
        ))}
        {sxNum && niceTicks(...pad(extent(xs as number[])), 5).map((t) => (
          <text key={t} x={sxNum(t)} y={M.t + innerH + 14} className="tick" textAnchor="middle">{fmt(t)}</text>
        ))}
        <text x={M.l + innerW / 2} y={height - 6} className="axis-label" textAnchor="middle">{spec.x} →</text>
      </svg>
      {yis.length > 1 && (
        <div className="legend" style={{ position: 'absolute', right: 6, top: 0 }}>
          {spec.y.map((y, k) => <span key={y}><i style={{ background: PALETTE[k % PALETTE.length] }} />{y}</span>)}
        </div>
      )}
    </div>
  );
}
