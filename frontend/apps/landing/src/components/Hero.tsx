'use client';

import React, { useEffect, useRef } from 'react';
import { ArrowRight } from 'lucide-react';
import { Eyebrow, HandArrow, MagneticLink, Scramble, Squiggle } from './primitives';
import { useReducedMotion } from '../lib/motion';

type CloudSpec = { top: string; w: number; dur: number; phase: number; blur?: number; o?: number };

const LAYERS: { depth: number; clouds: CloudSpec[] }[] = [
  {
    depth: 0.35,
    clouds: [
      { top: '8%', w: 520, dur: 190, phase: 0.1, blur: 14, o: 0.7 },
      { top: '22%', w: 640, dur: 220, phase: 0.55, blur: 16, o: 0.65 },
      { top: '4%', w: 420, dur: 170, phase: 0.82, blur: 12, o: 0.6 },
    ],
  },
  {
    depth: 0.8,
    clouds: [
      { top: '48%', w: 760, dur: 140, phase: 0.2, blur: 8 },
      { top: '58%', w: 560, dur: 120, phase: 0.62, blur: 7 },
      { top: '36%', w: 380, dur: 110, phase: 0.9, blur: 6, o: 0.85 },
    ],
  },
  {
    depth: 1.5,
    clouds: [
      { top: '70%', w: 980, dur: 95, phase: 0.05, blur: 5 },
      { top: '76%', w: 820, dur: 85, phase: 0.48, blur: 4 },
      { top: '66%', w: 600, dur: 75, phase: 0.78, blur: 4 },
    ],
  },
];

const STACK = [
  { name: 'PANDAS', role: 'profiling' },
  { name: 'SCIKIT-LEARN', role: 'pipelines' },
  { name: 'FASTAPI', role: 'engine api' },
  { name: 'GEMINI', role: 'explanations' },
  { name: 'XGBOOST', role: 'benchmarks' },
  { name: 'NEXT.JS', role: 'dashboard' },
];

export function Nav({ dashboardUrl }: { dashboardUrl: string }) {
  return (
    <header className="nav">
      <div className="frame nav-inner">
        <a href="#top" className="logo" aria-label="Nebulus home">
          <svg className="logo-mark" viewBox="0 0 26 26" aria-hidden>
            <rect x="1" y="1" width="24" height="24" rx="7" fill="#6d4aff" />
            <path d="M8 18V8l10 10V8" stroke="#fff" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Nebulus
        </a>
        <nav className="nav-links">
          <a href="#scanner">Live Scan</a>
          <a href="#features">Features</a>
          <a href="#workflow">Workflow</a>
          <a href="#exports">Exports</a>
        </nav>
        <div className="nav-cta">
          <a href="#scanner" className="btn btn-light btn-sm">See it work</a>
          <MagneticLink href={dashboardUrl} className="btn btn-dark btn-sm" strength={0.18}>
            Open Dashboard
          </MagneticLink>
        </div>
      </div>
    </header>
  );
}

export function Hero({ dashboardUrl }: { dashboardUrl: string }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    let raf = 0;
    const onMove = (e: PointerEvent) => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const x = (e.clientX / window.innerWidth - 0.5) * 40;
        const y = (e.clientY / window.innerHeight - 0.5) * 24;
        el.style.setProperty('--cx', x.toFixed(1));
        el.style.setProperty('--cy', y.toFixed(1));
      });
    };
    const onScroll = () => el.style.setProperty('--sy', String(-window.scrollY));
    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('scroll', onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduced]);

  return (
    <section ref={ref} className="hero" id="top">
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden>
        <filter id="cloud-filter" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="4" seed="7" />
          <feDisplacementMap in="SourceGraphic" scale="70" />
        </filter>
      </svg>

      <div className="sky" />
      <div className="sun" />
      <div className="clouds" aria-hidden>
        {LAYERS.map((layer, li) => (
          <div key={li} className="cloud-layer" style={{ ['--depth' as string]: layer.depth }}>
            {layer.clouds.map((c, ci) => (
              <div
                key={ci}
                className="cloud-drift"
                style={{
                  top: c.top,
                  ['--dur' as string]: `${c.dur}s`,
                  ['--delay' as string]: `${-c.dur * c.phase}s`,
                }}
              >
                <div
                  className="cloud"
                  style={{
                    ['--w' as string]: `${c.w}px`,
                    ['--blur' as string]: `${c.blur ?? 6}px`,
                    ['--o' as string]: c.o ?? 0.95,
                  }}
                />
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="guides" />

      <div className="hero-content">
        <Eyebrow label="data readiness" index="00/05" />
        <h1 className="hero-title">
          <span className="line"><Scramble text="Messy data in." delay={250} /></span>
          <span className="line">
            <span className="underline-wrap">
              Model-ready
              <Squiggle delay={1.4} />
            </span>{' '}
            out.
          </span>
        </h1>
        <p className="hero-sub">
          Nebulus profiles your CSV, scores it <strong>0–100</strong>, explains every fix in plain English, and hands
          you a cleaned dataset, a reproducible pipeline script and an honest model leaderboard.
        </p>
        <div className="hero-ctas">
          <MagneticLink href={dashboardUrl} className="btn btn-dark btn-lg">
            Clean a dataset <ArrowRight size={15} className="arrow" />
          </MagneticLink>
          <MagneticLink href="#scanner" className="btn btn-light btn-lg" strength={0.2}>
            Watch the scan
          </MagneticLink>
          <span className="note hero-note-cta" style={{ ['--delay' as string]: '2.3s' }}>
            <HandArrow width={46} height={30} delay={2.4} d="M44 6 C 30 4, 14 10, 6 24 M6 24 l 1 -9 M6 24 l 8 -3" />
            no signup — 3 demo datasets inside
          </span>
        </div>
      </div>

      <div className="strip">
        <div className="strip-label">BUILT ON THE BORING, BATTLE-TESTED PYTHON DATA STACK</div>
        <div className="strip-grid">
          {STACK.map((s) => (
            <div key={s.name} className="strip-cell">
              <span className="name">{s.name}</span>
              <span className="role">{s.role}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
