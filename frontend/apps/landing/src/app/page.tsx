'use client';

import React, { useEffect, useRef } from 'react';
import { Hero, Nav } from '../components/Hero';
import { Scanner } from '../components/Scanner';
import { Features } from '../components/Features';
import { Workflow } from '../components/Workflow';
import { CTA, Exports, Footer } from '../components/Exports';

function ScrollProgress() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      ref.current?.style.setProperty('--p', String(max > 0 ? window.scrollY / max : 0));
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return <div ref={ref} className="scroll-progress" aria-hidden />;
}

export default function LandingPage() {
  const dashboardUrl = process.env.NEXT_PUBLIC_DASHBOARD_URL || 'http://localhost:3000';

  return (
    <>
      <ScrollProgress />
      <div className="grain" aria-hidden />
      <Nav dashboardUrl={dashboardUrl} />
      <main>
        <Hero dashboardUrl={dashboardUrl} />
        <Scanner />
        <Features />
        <Workflow />
        <Exports />
        <CTA dashboardUrl={dashboardUrl} />
      </main>
      <Footer />
    </>
  );
}
