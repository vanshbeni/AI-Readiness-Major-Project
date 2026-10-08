'use client';

import React, { useEffect, useRef, useState, type RefObject } from 'react';

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

export function useInView<T extends Element>(threshold = 0.15): [RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          io.disconnect();
        }
      },
      { threshold, rootMargin: '0px 0px -6% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return [ref, inView];
}

/** Sets --mx/--my on the element for cursor-following highlights. */
export function trackPointer(e: React.PointerEvent<HTMLElement>) {
  const el = e.currentTarget;
  const r = el.getBoundingClientRect();
  el.style.setProperty('--mx', `${e.clientX - r.left}px`);
  el.style.setProperty('--my', `${e.clientY - r.top}px`);
}

export function Reveal({
  children,
  delay = 0,
  className = '',
  style,
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
  style?: React.CSSProperties;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <div ref={ref} className={`reveal ${inView ? 'in' : ''} ${className}`} style={{ ...style, ['--d' as string]: `${delay}ms` }}>
      {children}
    </div>
  );
}

export function Eyebrow({ label, index, caret = false }: { label: string; index?: string; caret?: boolean }) {
  return (
    <span className="eyebrow">
      <span className="dollar">$</span>
      <span>nebulus</span>
      <span className="sep">·</span>
      <b>{label}</b>
      {index && <span className="count">[{index}]</span>}
      {caret && <span className="caret" />}
    </span>
  );
}

const GLYPHS = '<>/\\[]{}#%&*+=01?!~^';

/** Text that resolves from symbol noise, left to right. Re-scrambles on hover. */
export function Scramble({ text, delay = 0, duration = 900 }: { text: string; delay?: number; duration?: number }) {
  const reduced = useReducedMotion();
  const [resolved, setResolved] = useState(0);
  const [tick, setTick] = useState(0);
  const runId = useRef(0);

  const run = (startDelay: number) => {
    const id = ++runId.current;
    let start = 0;
    let last = 0;
    const step = (t: number) => {
      if (id !== runId.current) return;
      if (!start) start = t + startDelay;
      const elapsed = Math.max(0, t - start);
      setResolved(Math.floor((elapsed / duration) * text.length));
      if (t - last > 55) {
        last = t;
        setTick((n) => n + 1);
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
            {done ? (ch === ' ' ? '\u00a0' : ch) : GLYPHS[(i * 7 + tick * 13) % GLYPHS.length]}
          </span>
        );
      })}
    </span>
  );
}

export function Squiggle({ delay = 0.7, color = 'var(--accent)' }: { delay?: number; color?: string }) {
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

export function Underlined({ children, delay }: { children: React.ReactNode; delay?: number }) {
  return (
    <span className="underline-wrap">
      {children}
      <Squiggle delay={delay} />
    </span>
  );
}

/** Handwritten margin note with an optional drawn arrow. */
export function Note({
  children,
  tone,
  delay = 0.9,
  rot = -4,
  arrow,
  style,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
  tone?: 'red' | 'green' | 'purple';
  delay?: number;
  rot?: number;
  arrow?: 'left' | 'down' | 'up';
  style?: React.CSSProperties;
}) {
  const paths = {
    left: 'M40 10 C 28 6, 14 8, 4 14 M4 14 l 8 -6 M4 14 l 9 3',
    down: 'M6 4 C 4 14, 10 24, 22 28 M22 28 l -9 1 M22 28 l -3 -8',
    up: 'M6 28 C 6 18, 12 8, 24 4 M24 4 l -9 -1 M24 4 l -2 8',
  };
  return (
    <span
      className={`note ${tone ?? ''} ${className}`}
      style={{ ...style, ['--delay' as string]: `${delay}s`, ['--rot' as string]: `${rot}deg` }}
    >
      {arrow === 'left' && (
        <svg width="44" height="22" viewBox="0 0 44 22" aria-hidden>
          <path className="draw-path" pathLength={1} style={{ ['--delay' as string]: `${delay + 0.1}s` }} d={paths.left} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
      {children}
      {(arrow === 'down' || arrow === 'up') && (
        <svg width="30" height="32" viewBox="0 0 30 32" aria-hidden>
          <path className="draw-path" pathLength={1} style={{ ['--delay' as string]: `${delay + 0.1}s` }} d={paths[arrow]} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </span>
  );
}

/** Number that counts up from `from` to `value` when it scrolls into view. */
export function CountUp({ value, from = 0, decimals = 1, duration = 1300 }: { value: number; from?: number; decimals?: number; duration?: number }) {
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLSpanElement>(0.3);
  const [shown, setShown] = useState(from);

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setShown(value);
      return;
    }
    let raf = 0;
    let start = 0;
    const step = (t: number) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setShown(from + (value - from) * eased);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, from, duration, reduced]);

  return <span ref={ref}>{shown.toFixed(decimals)}</span>;
}

/** Card with a cursor spotlight and optional 3D tilt. */
export function Panel({
  children,
  className = '',
  tilt = false,
  lift = false,
  style,
  as: Tag = 'div',
  ...rest
}: {
  children: React.ReactNode;
  className?: string;
  tilt?: boolean;
  lift?: boolean;
  style?: React.CSSProperties;
  as?: React.ElementType;
} & Record<string, unknown>) {
  const reduced = useReducedMotion();
  const onMove = (e: React.PointerEvent<HTMLElement>) => {
    trackPointer(e);
    if (!tilt || reduced || e.pointerType !== 'mouse') return;
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--ry', `${((e.clientX - r.left) / r.width - 0.5) * 6}deg`);
    el.style.setProperty('--rx', `${-((e.clientY - r.top) / r.height - 0.5) * 6}deg`);
  };
  const onLeave = (e: React.PointerEvent<HTMLElement>) => {
    e.currentTarget.style.setProperty('--rx', '0deg');
    e.currentTarget.style.setProperty('--ry', '0deg');
  };
  return (
    <Tag
      className={`panel ${tilt ? 'tilt' : ''} ${lift ? 'lift' : ''} ${className}`}
      style={style}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      {...rest}
    >
      {children}
    </Tag>
  );
}

/** Button that drifts toward the cursor. */
export function MagneticButton({
  children,
  className = '',
  strength = 0.22,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { strength?: number }) {
  const ref = useRef<HTMLButtonElement>(null);
  const reduced = useReducedMotion();
  return (
    <button
      ref={ref}
      className={className}
      onPointerMove={(e) => {
        trackPointer(e);
        if (reduced || e.pointerType !== 'mouse' || rest.disabled) return;
        const r = e.currentTarget.getBoundingClientRect();
        e.currentTarget.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * strength}px, ${(e.clientY - r.top - r.height / 2) * strength}px)`;
      }}
      onPointerLeave={(e) => {
        e.currentTarget.style.transform = '';
      }}
      {...rest}
    >
      {children}
    </button>
  );
}

const CLOUDS = [
  { top: '-4%', w: 560, dur: 200, phase: 0.1, blur: 12, o: 0.75 },
  { top: '18%', w: 720, dur: 160, phase: 0.45, blur: 9, o: 0.85 },
  { top: '40%', w: 900, dur: 120, phase: 0.75, blur: 6, o: 0.9 },
  { top: '8%', w: 420, dur: 140, phase: 0.92, blur: 8, o: 0.7 },
  { top: '52%', w: 640, dur: 100, phase: 0.3, blur: 5, o: 0.9 },
];

/** Drifting cloud band behind the header, fading into the page. */
export function SkyBand() {
  return (
    <div className="sky-band" aria-hidden>
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <filter id="cloud-filter" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="4" seed="7" />
          <feDisplacementMap in="SourceGraphic" scale="70" />
        </filter>
      </svg>
      <div className="sky" />
      {CLOUDS.map((c, i) => (
        <div
          key={i}
          className="cloud-drift"
          style={{ top: c.top, ['--dur' as string]: `${c.dur}s`, ['--delay' as string]: `${-c.dur * c.phase}s` }}
        >
          <div className="cloud" style={{ ['--w' as string]: `${c.w}px`, ['--blur' as string]: `${c.blur}px`, ['--o' as string]: c.o }} />
        </div>
      ))}
    </div>
  );
}

export function Banner({
  tone,
  children,
  onClose,
}: {
  tone: 'error' | 'info';
  children: React.ReactNode;
  onClose?: () => void;
}) {
  return (
    <div className={`banner ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <span className="tag">{tone === 'error' ? '[error]' : '[note]'}</span>
      <div style={{ flex: 1 }}>{children}</div>
      {onClose && (
        <button className="x" onClick={onClose} aria-label="Dismiss">✕</button>
      )}
    </div>
  );
}
