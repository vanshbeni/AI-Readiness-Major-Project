'use client';

import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';
import { Eyebrow, Reveal } from './primitives';
import { trackPointer, useInView, useReducedMotion } from '../lib/motion';

function Tile({
  className = '',
  index,
  title,
  desc,
  children,
  delay = 0,
}: {
  className?: string;
  index: string;
  title: string;
  desc: string;
  children?: React.ReactNode;
  delay?: number;
}) {
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>(0.2);

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    trackPointer(e);
    if (reduced || e.pointerType !== 'mouse') return;
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    el.style.setProperty('--ry', `${px * 7}deg`);
    el.style.setProperty('--rx', `${-py * 7}deg`);
  };
  const onLeave = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.style.setProperty('--rx', '0deg');
    e.currentTarget.style.setProperty('--ry', '0deg');
  };

  return (
    <div
      ref={ref}
      className={`tile reveal ${inView ? 'in' : ''} ${className}`}
      style={{ ['--d' as string]: `${delay}ms` }}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
    >
      <span className="tile-idx">{index}</span>
      <h3>{title}</h3>
      <p>{desc}</p>
      {children && <div className="tile-visual">{children}</div>}
    </div>
  );
}

const SUBSCORES = [
  { k: 'missingness', v: 72 },
  { k: 'duplicates', v: 88 },
  { k: 'outliers', v: 64 },
  { k: 'validity', v: 91 },
  { k: 'target balance', v: 58 },
  { k: 'feature quality', v: 80 },
];

function ApprovalDemo() {
  const [items, setItems] = useState([
    { label: 'Impute Age with median', on: true },
    { label: 'Drop Cabin (77% missing)', on: false },
    { label: 'Cap Fare outliers', on: true },
  ]);
  return (
    <div className="toggle-demo">
      {items.map((it, i) => (
        <div
          key={it.label}
          className="toggle-row"
          role="switch"
          aria-checked={it.on}
          tabIndex={0}
          onClick={() => setItems((prev) => prev.map((p, j) => (j === i ? { ...p, on: !p.on } : p)))}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              setItems((prev) => prev.map((p, j) => (j === i ? { ...p, on: !p.on } : p)));
            }
          }}
        >
          <span style={{ color: it.on ? 'var(--ink)' : 'var(--dim)', textDecoration: it.on ? 'none' : 'line-through' }}>{it.label}</span>
          <span className={`switch ${it.on ? 'on' : ''}`} />
        </div>
      ))}
    </div>
  );
}

export function Features() {
  return (
    <section className="section" id="features">
      <div className="guides" />
      <div className="frame">
        <div className="section-head">
          <Reveal><Eyebrow label="what you get" index="02/05" /></Reveal>
          <Reveal delay={80} as="h2" className="section-title">No black box. Every fix has a receipt.</Reveal>
          <Reveal delay={160} as="p" className="section-sub">
            Rules pick the fix from the column&apos;s statistics, an LLM only explains it, and nothing touches your data
            until you approve it.
          </Reveal>
        </div>

        <div className="bento">
          <Tile
            className="xwide"
            index="/01"
            title="A 0–100 health score you can argue with"
            desc="Six sub-scores, each traceable to a number in your data. The score is re-computed after cleaning, so a fix that hurts shows up as a drop."
          >
            <div className="bars">
              {SUBSCORES.map((s, i) => (
                <div className="bar-row" key={s.k}>
                  <span>{s.k}</span>
                  <span className="track">
                    <span className="fill" style={{ display: 'block', ['--w' as string]: `${s.v}%`, ['--d' as string]: `${200 + i * 90}ms` }} />
                  </span>
                  <span style={{ textAlign: 'right', color: 'var(--ink)' }}>{s.v}</span>
                </div>
              ))}
            </div>
          </Tile>

          <Tile
            index="/02"
            title="Explained like a colleague would"
            desc="Gemini writes the why, grounded only in your computed stats. If it's down, deterministic explanations take over."
            delay={80}
          >
            <div className="chips">
              <span className="chip"><Sparkles size={10} style={{ verticalAlign: '-1px' }} /> AI</span>
              <span className="chip">skew 1.3</span>
              <span className="chip">19.9% missing</span>
              <span className="chip">→ median</span>
            </div>
          </Tile>

          <Tile
            index="/03"
            title="You hold the switch"
            desc="Destructive steps are off by default. Toggle anything — go on, these work."
          >
            <ApprovalDemo />
          </Tile>

          <Tile
            className="xwide"
            index="/04"
            title="Benchmarks that don't cheat"
            desc="Scaling, imputation and SMOTE happen inside each cross-validation fold, so the leaderboard reflects what you'd see in production — not a suspicious 1.000."
            delay={80}
          >
            <div className="chips">
              {['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting', 'Logistic Regression', 'SVM'].map((m) => (
                <span className="chip" key={m}>{m}</span>
              ))}
            </div>
          </Tile>

          <Tile
            className="wide"
            index="/05"
            title="Handles the ugly files"
            desc="Semicolon CSVs, Windows encodings, Excel sheets with numeric headers, duplicate column names, infinities, IDs posing as features."
          >
            <div className="chips">
              {['utf-8-sig', 'cp1252', ';', '.xlsx', '.xls', 'inf → NaN'].map((m) => (
                <span className="chip" key={m}>{m}</span>
              ))}
            </div>
          </Tile>

          <Tile
            className="wide"
            index="/06"
            title="Leaves a paper trail"
            desc="Cleaned CSV, a standalone Python script that reproduces it exactly, a PDF audit report and a JSON manifest."
            delay={80}
          >
            <div className="chips">
              {['cleaned.csv', 'pipeline.py', 'report.pdf', 'manifest.json'].map((m) => (
                <span className="chip" key={m}>{m}</span>
              ))}
            </div>
          </Tile>
        </div>
      </div>
    </section>
  );
}
