'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useInView, useReducedMotion, trackPointer } from '../lib/motion';

export function Reveal({
  children,
  delay = 0,
  as: Tag = 'div',
  className = '',
  style,
}: {
  children: React.ReactNode;
  delay?: number;
  as?: React.ElementType;
  className?: string;
  style?: React.CSSProperties;
}) {
  const [ref, inView] = useInView<HTMLElement>(0.15);
  return (
    <Tag
      ref={ref}
      className={`reveal ${inView ? 'in' : ''} ${className}`}
      style={{ ...style, ['--d' as string]: `${delay}ms` }}
    >
      {children}
    </Tag>
  );
}

export function Eyebrow({ label, index }: { label: string; index: string }) {
  return (
    <span className="eyebrow">
      <span className="dollar">$</span>
      <span>nebulus</span>
      <span className="sep">·</span>
      <b>{label}</b>
      <span className="count">[{index}]</span>
    </span>
  );
}

/** Link that drifts toward the cursor and carries a pointer-following highlight. */
export function MagneticLink({
  href,
  className,
  children,
  strength = 0.28,
}: {
  href: string;
  className: string;
  children: React.ReactNode;
  strength?: number;
}) {
  const ref = useRef<HTMLAnchorElement>(null);
  const reduced = useReducedMotion();

  const onMove = (e: React.PointerEvent<HTMLAnchorElement>) => {
    trackPointer(e);
    if (reduced || e.pointerType !== 'mouse') return;
    const el = ref.current!;
    const r = el.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    el.style.transform = `translate(${dx * strength}px, ${dy * strength}px)`;
  };
  const onLeave = () => {
    if (ref.current) ref.current.style.transform = '';
  };

  return (
    <a ref={ref} href={href} className={className} onPointerMove={onMove} onPointerLeave={onLeave}>
      {children}
    </a>
  );
}

const GLYPHS = '<>/\\[]{}#%&*+=01?!~^';

/** Text that starts as noise and resolves left-to-right; re-scrambles on hover. */
export function Scramble({ text, delay = 0, duration = 1100 }: { text: string; delay?: number; duration?: number }) {
  const reduced = useReducedMotion();
  const [resolved, setResolved] = useState(reduced ? text.length : 0);
  const [tick, force] = useState(0);
  const runId = useRef(0);

  const run = (startDelay: number) => {
    const id = ++runId.current;
    let start = 0;
    let lastTick = 0;
    const step = (t: number) => {
      if (id !== runId.current) return;
      if (!start) start = t + startDelay;
      const elapsed = Math.max(0, t - start);
      setResolved(Math.floor((elapsed / duration) * text.length));
      if (t - lastTick > 55) {
        lastTick = t;
        force((n) => n + 1);
      }
      if (elapsed < duration) requestAnimationFrame(step);
      else setResolved(text.length);
    };
    requestAnimationFrame(step);
  };

  useEffect(() => {
    if (reduced) {
      setResolved(text.length);
      return;
    }
    setResolved(0);
    run(delay);
    return () => {
      runId.current++;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, reduced]);

  return (
    <span
      aria-label={text}
      onPointerEnter={() => {
        if (!reduced && resolved >= text.length) {
          setResolved(0);
          run(0);
        }
      }}
    >
      {text.split('').map((ch, i) => {
        const done = i < resolved || ch === ' ';
        return (
          <span key={i} aria-hidden className={`scramble-char ${done ? '' : 'pending'}`}>
            {done ? (ch === ' ' ? '\u00a0' : ch) : GLYPHS[(i * 7 + tick * 13 + ((i * tick) % 5)) % GLYPHS.length]}
          </span>
        );
      })}
    </span>
  );
}

/** Hand-drawn squiggle underline that draws itself. */
export function Squiggle({ delay = 1.3, color = 'var(--accent)' }: { delay?: number; color?: string }) {
  return (
    <svg viewBox="0 0 300 20" preserveAspectRatio="none" aria-hidden>
      <path
        className="draw-path"
        pathLength={1}
        style={{ ['--delay' as string]: `${delay}s` }}
        d="M3 13 C 40 5, 70 17, 110 10 S 175 4, 210 11 S 270 15, 297 7"
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Small hand-drawn arrow, drawn on mount. */
export function HandArrow({ d, width = 60, height = 40, delay = 2.2 }: { d: string; width?: number; height?: number; delay?: number }) {
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
      <path
        className="draw-path"
        pathLength={1}
        style={{ ['--delay' as string]: `${delay}s` }}
        d={d}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
