'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Eyebrow } from './primitives';
import { useStickyProgress } from '../lib/motion';

const STEPS = [
  { tag: 'csv · xlsx', title: 'Drop a file', desc: 'Encoding, delimiter and messy headers are sorted out before you see a single row.' },
  { tag: 'objective', title: 'Pick what to predict', desc: 'Choose the target. Nebulus suggests classification or regression and refuses IDs as targets.' },
  { tag: 'diagnose', title: 'Get the score', desc: 'Profiling, issue detection and the 0–100 health score, with the drivers that pulled it down.' },
  { tag: 'approve', title: 'Approve the fixes', desc: 'Each fix comes with its statistics and a plain-English reason. Toggle what you trust.' },
  { tag: 'export', title: 'Ship it', desc: 'Re-scored data, an honest leaderboard and four downloadable artifacts.' },
];

export function Workflow() {
  const [sectionRef, progress] = useStickyProgress<HTMLElement>(60);
  const trackRef = useRef<HTMLDivElement>(null);
  const [distance, setDistance] = useState(0);
  const [height, setHeight] = useState<number | undefined>(undefined);

  useEffect(() => {
    const measure = () => {
      const track = trackRef.current;
      if (!track) return;
      const d = Math.max(0, track.scrollWidth - window.innerWidth);
      setDistance(d);
      setHeight(d + window.innerHeight);
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  const active = Math.min(STEPS.length - 1, Math.floor(progress * STEPS.length));

  return (
    <section ref={sectionRef} className="flow-section" id="workflow" style={{ height }}>
      <div className="flow-sticky">
        <div className="frame" style={{ width: '100%' }}>
          <Eyebrow label="workflow" index="03/05" />
          <h2 className="section-title" style={{ marginTop: '0.9rem', maxWidth: '20ch' }}>
            Five steps. One of them is just clicking “approve”.
          </h2>
        </div>

        <div ref={trackRef} className="flow-track" style={{ transform: `translate3d(${-progress * distance}px, 0, 0)` }}>
          {STEPS.map((s, i) => (
            <article key={s.title} className={`flow-card ${i <= active ? 'active' : ''}`}>
              <span className="tag">{s.tag}</span>
              <span className="big">0{i + 1}</span>
              <div>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </div>
            </article>
          ))}
        </div>

        <div className="frame" style={{ width: '100%' }}>
          <div className="flow-progress" style={{ ['--p' as string]: progress }}>
            <span>0{active + 1}</span>
            <span className="track"><i /></span>
            <span>05</span>
          </div>
        </div>
      </div>
    </section>
  );
}
