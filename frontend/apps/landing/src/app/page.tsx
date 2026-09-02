'use client';

import React from 'react';
import {
  ShieldCheck,
  Sparkles,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cpu,
  Layers,
  FileCode,
  Download,
  Gauge,
  SlidersHorizontal,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';

export default function LandingPage() {
  const dashboardUrl = process.env.NEXT_PUBLIC_DASHBOARD_URL || 'http://localhost:3000';

  const features = [
    {
      icon: Gauge,
      title: '0–100 Data Health Score',
      desc: 'Transparent composite quality scoring across missingness, duplication, outliers, domain validity, class balance, and feature cardinality.',
      badge: 'Deterministic',
      color: '#38bdf8',
    },
    {
      icon: Sparkles,
      title: 'AI Explainable Remediations',
      desc: 'Every transformation recommendation is mathematically grounded and translated into natural-language reasoning with Gemini.',
      badge: 'Explainable AI',
      color: '#a855f7',
    },
    {
      icon: SlidersHorizontal,
      title: 'Human-in-the-Loop Control',
      desc: 'Interactive checklist ensuring zero destructive data modifications without explicit approval and preview confirmation.',
      badge: 'Safety First',
      color: '#10b981',
    },
    {
      icon: Cpu,
      title: 'AutoML Candidate Benchmarking',
      desc: 'Fast 3-fold cross-validation across XGBoost, LightGBM, Random Forest, and baseline models with ranked leaderboards.',
      badge: 'Performance',
      color: '#f59e0b',
    },
    {
      icon: Layers,
      title: 'Safe-Order Transformation Engine',
      desc: 'Strict mathematically sound sequence: deduplication → imputation → encoding → outlier handling → scaling → resampling.',
      badge: 'Orchestration',
      color: '#ec4899',
    },
    {
      icon: Download,
      title: 'Reproducible Production Artifacts',
      desc: 'Instant export of preprocessed CSVs, standalone scikit-learn Python scripts, PDF audit reports, and JSON manifests.',
      badge: 'Export Hub',
      color: '#38bdf8',
    },
  ];

  const steps = [
    { num: '01', title: 'Upload & Ingest', desc: 'Drag-and-drop CSV or Excel tabular datasets with automatic dtype inference.' },
    { num: '02', title: 'Define ML Objective', desc: 'Specify Supervised Classification or Regression task and target variable.' },
    { num: '03', title: 'AI Health Assessment', desc: 'Inspect composite quality scores, feature metrics, and anomaly distributions.' },
    { num: '04', title: 'Approve Explainable Fixes', desc: 'Review plain-English justifications and toggle remediation actions.' },
    { num: '05', title: 'Export Clean Data & Models', desc: 'Download cleaned CSV, scikit-learn pipeline scripts, and audit reports.' },
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation */}
      <nav style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1.25rem 2.5rem',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(7, 9, 14, 0.8)',
        backdropFilter: 'blur(16px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: 'linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(56, 189, 248, 0.4)',
          }}>
            <ShieldCheck size={20} color="#07090e" strokeWidth={2.5} />
          </div>
          <span style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
            AegisMind
          </span>
          <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>v1.0</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          <a
            href="#features"
            style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500 }}
          >
            Features
          </a>
          <a
            href="#how-it-works"
            style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500 }}
          >
            Workflow
          </a>
          <a
            href={dashboardUrl}
            className="btn-primary"
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.875rem' }}
          >
            <span>Launch Dashboard</span>
            <ArrowRight size={15} />
          </a>
        </div>
      </nav>

      <main style={{ flex: 1 }}>
        {/* HERO SECTION */}
        <section style={{
          padding: '6rem 2rem 4rem 2rem',
          maxWidth: 1100,
          margin: '0 auto',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem',
        }}>
          <div className="badge badge-cyan" style={{ padding: '0.35rem 0.85rem' }}>
            <Sparkles size={13} />
            <span>Explainable Pre-ML Diagnostic & Remediation Gateway</span>
          </div>

          <h1 style={{
            fontSize: 'clamp(2.5rem, 5vw, 4rem)',
            fontWeight: 800,
            lineHeight: 1.15,
            letterSpacing: '-0.03em',
            maxWidth: 900,
          }}>
            Close the Gap Between <br />
            <span className="gradient-text">Messy Data</span> & <span className="gradient-text">Robust ML Pipelines</span>
          </h1>

          <p style={{
            color: 'var(--text-muted)',
            fontSize: 'clamp(1rem, 2vw, 1.2rem)',
            maxWidth: 720,
            lineHeight: 1.6,
          }}>
            AegisMind is an intelligent pre-ML diagnosis system that calculates a transparent <strong>0–100 Data Health Score</strong>, explains remediation fixes with AI, and builds reproducible scikit-learn pipelines with human-in-the-loop control.
          </p>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            <a href={dashboardUrl} className="btn-primary" style={{ fontSize: '1.05rem', padding: '0.85rem 2rem' }}>
              <span>Open Diagnosis Dashboard</span>
              <ArrowRight size={18} />
            </a>
            <a href="#how-it-works" className="btn-secondary" style={{ fontSize: '1.05rem', padding: '0.85rem 1.75rem' }}>
              <span>See How It Works</span>
            </a>
          </div>

          {/* Interactive Hero Preview Card Mockup */}
          <div
            className="glass-panel glow-cyan"
            style={{
              marginTop: '3.5rem',
              width: '100%',
              padding: '2rem',
              textAlign: 'left',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              background: 'linear-gradient(180deg, rgba(14, 20, 31, 0.95) 0%, rgba(7, 9, 14, 0.9) 100%)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444' }} />
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#f59e0b' }} />
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#10b981' }} />
                <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginLeft: '0.5rem', fontFamily: 'monospace' }}>
                  aegismind-engine // diagnostic_pipeline.py
                </span>
              </div>
              <span className="badge badge-emerald">Engine Ready</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
              <div style={{ padding: '1.25rem', borderRadius: 12, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 600 }}>RAW DATA QUALITY</span>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f43f5e', marginTop: '0.25rem' }}>
                  54 <span style={{ fontSize: '0.9rem', color: 'var(--text-dim)' }}>/ 100</span>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#fb7185', marginTop: '0.25rem' }}>Grade: D • 4 Defect Drivers</p>
              </div>

              <div style={{ padding: '1.25rem', borderRadius: 12, background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <span style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 600 }}>CLEANED POST-PIPELINE</span>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#34d399', marginTop: '0.25rem' }}>
                  94 <span style={{ fontSize: '0.9rem', color: 'var(--text-dim)' }}>/ 100</span>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#34d399', marginTop: '0.25rem' }}>Grade: A+ • +40 pts Gain</p>
              </div>

              <div style={{ padding: '1.25rem', borderRadius: 12, background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: 600 }}>TOP RECOMMENDED MODEL</span>
                <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '0.35rem' }}>
                  XGBoost Classifier
                </div>
                <p style={{ fontSize: '0.75rem', color: '#38bdf8', marginTop: '0.25rem' }}>F1-Score: 0.884 (CV 3-Fold)</p>
              </div>
            </div>
          </div>
        </section>

        {/* FEATURES GRID */}
        <section id="features" style={{ padding: '5rem 2rem', maxWidth: 1100, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <span className="badge badge-purple" style={{ marginBottom: '0.75rem' }}>Core Capabilities</span>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 800 }}>
              Engineered for Transparent <span className="gradient-text">Data Engineering</span>
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginTop: '0.5rem' }}>
              Unlike black-box AutoML, AegisMind ensures every remediation step is deterministic, explainable, and user-approved.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
                <div
                  key={i}
                  className="glass-panel"
                  style={{
                    padding: '1.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.85rem',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{
                      width: 42,
                      height: 42,
                      borderRadius: 10,
                      background: 'rgba(255, 255, 255, 0.05)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: f.color,
                    }}>
                      <Icon size={22} />
                    </div>
                    <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>{f.badge}</span>
                  </div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>{f.title}</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    {f.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </section>

        {/* WORKFLOW STEPS */}
        <section id="how-it-works" style={{ padding: '5rem 2rem', maxWidth: 1000, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <span className="badge badge-emerald" style={{ marginBottom: '0.75rem' }}>End-to-End Workflow</span>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 800 }}>
              From Raw Ingestion to <span className="gradient-text">Clean Pipeline in 5 Steps</span>
            </h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {steps.map((s, idx) => (
              <div
                key={idx}
                className="glass-panel"
                style={{
                  padding: '1.5rem 2rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.75rem',
                }}
              >
                <span style={{ fontSize: '1.6rem', fontWeight: 800, color: '#38bdf8', fontFamily: 'monospace', width: 45 }}>
                  {s.num}
                </span>
                <div style={{ flex: 1 }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.2rem' }}>{s.title}</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{s.desc}</p>
                </div>
                <CheckCircle2 size={20} color="#10b981" />
              </div>
            ))}
          </div>
        </section>

        {/* CALL TO ACTION */}
        <section style={{
          padding: '6rem 2rem',
          maxWidth: 900,
          margin: '0 auto',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem',
        }}>
          <div
            className="glass-panel glow-cyan"
            style={{
              padding: '3.5rem 2rem',
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1.25rem',
              background: 'linear-gradient(135deg, rgba(14, 20, 31, 0.95) 0%, rgba(56, 189, 248, 0.1) 100%)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
            }}
          >
            <h2 style={{ fontSize: '2.25rem', fontWeight: 800 }}>
              Ready to Audit & Optimize Your Data?
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: 600 }}>
              Launch the AegisMind Dashboard and test your datasets with instant health diagnostics and explainable AI remediations.
            </p>
            <a href={dashboardUrl} className="btn-primary" style={{ padding: '0.85rem 2.25rem', fontSize: '1rem', marginTop: '0.5rem' }}>
              <span>Launch AegisMind Dashboard</span>
              <ArrowRight size={18} />
            </a>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer style={{
        padding: '2rem 2.5rem',
        borderTop: '1px solid var(--border)',
        background: 'rgba(7, 9, 14, 0.95)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        fontSize: '0.85rem',
        color: 'var(--text-dim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={16} color="#38bdf8" />
          <span>AegisMind — AI Data Readiness & Model Recommendation Platform</span>
        </div>
        <div>
          <span>© 2026 AegisMind. All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
}
