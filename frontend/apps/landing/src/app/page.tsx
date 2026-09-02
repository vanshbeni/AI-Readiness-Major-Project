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
      color: '#7c5cfc',
    },
    {
      icon: Sparkles,
      title: 'AI Explainable Remediations',
      desc: 'Every transformation recommendation is mathematically grounded and translated into natural-language reasoning with Gemini.',
      badge: 'Explainable AI',
      color: '#7c5cfc',
    },
    {
      icon: SlidersHorizontal,
      title: 'Human-in-the-Loop Control',
      desc: 'Interactive checklist ensuring zero destructive data modifications without explicit approval and preview confirmation.',
      badge: 'Safety First',
      color: '#059669',
    },
    {
      icon: Cpu,
      title: 'AutoML Candidate Benchmarking',
      desc: 'Fast 3-fold cross-validation across XGBoost, LightGBM, Random Forest, and baseline models with ranked leaderboards.',
      badge: 'Performance',
      color: '#d97706',
    },
    {
      icon: Layers,
      title: 'Safe-Order Transformation Engine',
      desc: 'Strict mathematically sound sequence: deduplication → imputation → encoding → outlier handling → scaling → resampling.',
      badge: 'Orchestration',
      color: '#db2777',
    },
    {
      icon: Download,
      title: 'Reproducible Production Artifacts',
      desc: 'Instant export of preprocessed CSVs, standalone scikit-learn Python scripts, PDF audit reports, and JSON manifests.',
      badge: 'Export Hub',
      color: '#7c5cfc',
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
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-main)' }}>
      {/* ═══════════════════════════════════════════════════════════════
          TOP NAVIGATION
          ═══════════════════════════════════════════════════════════════ */}
      <nav className="nav-container" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.875rem 2.5rem',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(248, 247, 244, 0.85)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'var(--text-main)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <ShieldCheck size={18} color="#ffffff" strokeWidth={2.5} />
          </div>
          <span style={{
            fontSize: '1.05rem',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: 'var(--text-main)',
          }}>
            AegisMind
          </span>
          <span className="badge badge-cyan" style={{ fontSize: '0.6rem' }}>v1.0</span>
        </div>

        <div className="nav-links" style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <a
            href="#features"
            style={{
              color: 'var(--text-muted)',
              textDecoration: 'none',
              fontSize: '0.875rem',
              fontWeight: 500,
              transition: 'color 0.2s ease',
            }}
          >
            Features
          </a>
          <a
            href="#how-it-works"
            style={{
              color: 'var(--text-muted)',
              textDecoration: 'none',
              fontSize: '0.875rem',
              fontWeight: 500,
              transition: 'color 0.2s ease',
            }}
          >
            Workflow
          </a>
          <a
            href={dashboardUrl}
            className="btn-primary"
            style={{ padding: '0.5rem 1.15rem', fontSize: '0.825rem' }}
          >
            <span>Launch Dashboard</span>
            <ArrowRight size={14} />
          </a>
        </div>
      </nav>

      <main style={{ flex: 1 }}>
        {/* ═══════════════════════════════════════════════════════════════
            HERO SECTION
            ═══════════════════════════════════════════════════════════════ */}
        <section style={{
          padding: '5rem var(--container-padding) 3.5rem',
          maxWidth: 'var(--container-max)',
          margin: '0 auto',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem',
        }}>
          <div className="badge badge-purple" style={{ padding: '0.3rem 0.75rem' }}>
            <Sparkles size={12} />
            <span>Explainable Pre-ML Diagnostic & Remediation Gateway</span>
          </div>

          <h1 style={{
            fontSize: 'clamp(2.25rem, 4.5vw, 3.5rem)',
            fontWeight: 800,
            lineHeight: 1.1,
            letterSpacing: '-0.035em',
            maxWidth: 800,
            color: 'var(--text-main)',
          }}>
            Close the Gap Between <br />
            <span className="gradient-text">Messy Data</span> & <span className="gradient-text">Robust ML Pipelines</span>
          </h1>

          <p style={{
            color: 'var(--text-muted)',
            fontSize: 'clamp(0.95rem, 1.5vw, 1.1rem)',
            maxWidth: 640,
            lineHeight: 1.65,
          }}>
            AegisMind is an intelligent pre-ML diagnosis system that calculates a transparent <strong style={{ color: 'var(--text-secondary)' }}>0–100 Data Health Score</strong>, explains remediation fixes with AI, and builds reproducible scikit-learn pipelines with human-in-the-loop control.
          </p>

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.75rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            <a href={dashboardUrl} className="btn-primary" style={{ fontSize: '0.95rem', padding: '0.75rem 1.75rem' }}>
              <span>Open Diagnosis Dashboard</span>
              <ArrowRight size={16} />
            </a>
            <a href="#how-it-works" className="btn-secondary" style={{ fontSize: '0.95rem', padding: '0.75rem 1.5rem' }}>
              <span>See How It Works</span>
            </a>
          </div>

          {/* Interactive Hero Preview Card Mockup */}
          <div
            className="card"
            style={{
              marginTop: '3rem',
              width: '100%',
              maxWidth: 960,
              padding: '1.75rem',
              textAlign: 'left',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.25rem',
              border: '1px solid var(--border)',
              background: 'var(--bg-card)',
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderBottom: '1px solid var(--border)',
              paddingBottom: '0.875rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444' }} />
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#f59e0b' }} />
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#10b981' }} />
                <span style={{
                  fontSize: '0.75rem',
                  color: 'var(--text-dim)',
                  marginLeft: '0.4rem',
                  fontFamily: "'JetBrains Mono', monospace",
                }}>
                  aegismind-engine // diagnostic_pipeline.py
                </span>
              </div>
              <span className="badge badge-emerald">Engine Ready</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              {/* Raw Data Quality */}
              <div style={{
                padding: '1.25rem',
                borderRadius: 'var(--radius-sm)',
                background: '#fef2f2',
                border: '1px solid #fecaca',
              }}>
                <span style={{ fontSize: '0.7rem', color: '#991b1b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>RAW DATA QUALITY</span>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#dc2626', marginTop: '0.25rem' }}>
                  54 <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>/ 100</span>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#b91c1c', marginTop: '0.2rem' }}>Grade: D • 4 Defect Drivers</p>
              </div>

              {/* Cleaned Post-Pipeline */}
              <div style={{
                padding: '1.25rem',
                borderRadius: 'var(--radius-sm)',
                background: '#ecfdf5',
                border: '1px solid #a7f3d0',
              }}>
                <span style={{ fontSize: '0.7rem', color: '#065f46', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>CLEANED POST-PIPELINE</span>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#059669', marginTop: '0.25rem' }}>
                  94 <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>/ 100</span>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#047857', marginTop: '0.2rem' }}>Grade: A+ • +40 pts Gain</p>
              </div>

              {/* Top Recommended Model */}
              <div style={{
                padding: '1.25rem',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-light)',
                border: '1px solid var(--border-accent)',
              }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--accent-text)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>TOP RECOMMENDED MODEL</span>
                <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '0.3rem' }}>
                  XGBoost Classifier
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--accent-text)', marginTop: '0.2rem' }}>F1-Score: 0.884 (CV 3-Fold)</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section Divider */}
        <hr className="section-divider" style={{ maxWidth: 'var(--container-max)', margin: '0 auto' }} />

        {/* ═══════════════════════════════════════════════════════════════
            FEATURES GRID
            ═══════════════════════════════════════════════════════════════ */}
        <section id="features" style={{
          padding: 'var(--section-gap) var(--container-padding)',
          maxWidth: 'var(--container-max)',
          margin: '0 auto',
        }}>
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <span className="eyebrow" style={{ display: 'block', marginBottom: '0.75rem' }}>Core Capabilities</span>
            <h2 style={{
              fontSize: 'clamp(1.75rem, 3vw, 2.25rem)',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.15,
              color: 'var(--text-main)',
            }}>
              Engineered for Transparent <span className="gradient-text">Data Engineering</span>
            </h2>
            <p style={{
              color: 'var(--text-muted)',
              fontSize: '0.95rem',
              marginTop: '0.75rem',
              maxWidth: 600,
              marginLeft: 'auto',
              marginRight: 'auto',
              lineHeight: 1.6,
            }}>
              Unlike black-box AutoML, AegisMind ensures every remediation step is deterministic, explainable, and user-approved.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '1.25rem',
          }}>
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
                <div
                  key={i}
                  className="card"
                  style={{
                    padding: '1.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{
                      width: 40,
                      height: 40,
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--accent-light)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: f.color,
                    }}>
                      <Icon size={20} />
                    </div>
                    <span className="badge badge-cyan" style={{ fontSize: '0.6rem' }}>{f.badge}</span>
                  </div>
                  <h3 style={{
                    fontSize: '1.05rem',
                    fontWeight: 700,
                    color: 'var(--text-main)',
                    letterSpacing: '-0.01em',
                  }}>
                    {f.title}
                  </h3>
                  <p style={{
                    fontSize: '0.85rem',
                    color: 'var(--text-muted)',
                    lineHeight: 1.55,
                  }}>
                    {f.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Section Divider */}
        <hr className="section-divider" style={{ maxWidth: 'var(--container-max)', margin: '0 auto' }} />

        {/* ═══════════════════════════════════════════════════════════════
            WORKFLOW STEPS
            ═══════════════════════════════════════════════════════════════ */}
        <section id="how-it-works" style={{
          padding: 'var(--section-gap) var(--container-padding)',
          maxWidth: 'var(--container-max)',
          margin: '0 auto',
        }}>
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <span className="eyebrow" style={{ display: 'block', marginBottom: '0.75rem' }}>End-to-End Workflow</span>
            <h2 style={{
              fontSize: 'clamp(1.75rem, 3vw, 2.25rem)',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.15,
              color: 'var(--text-main)',
            }}>
              From Raw Ingestion to <span className="gradient-text">Clean Pipeline in 5 Steps</span>
            </h2>
          </div>

          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            maxWidth: 800,
            margin: '0 auto',
          }}>
            {steps.map((s, idx) => (
              <div
                key={idx}
                className="card"
                style={{
                  padding: '1.25rem 1.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.5rem',
                }}
              >
                <span style={{
                  fontSize: '1.25rem',
                  fontWeight: 800,
                  color: 'var(--accent)',
                  fontFamily: "'JetBrains Mono', monospace",
                  width: 40,
                  flexShrink: 0,
                }}>
                  {s.num}
                </span>
                <div style={{ flex: 1 }}>
                  <h4 style={{
                    fontSize: '1rem',
                    fontWeight: 700,
                    marginBottom: '0.15rem',
                    color: 'var(--text-main)',
                    letterSpacing: '-0.01em',
                  }}>
                    {s.title}
                  </h4>
                  <p style={{
                    fontSize: '0.85rem',
                    color: 'var(--text-muted)',
                    lineHeight: 1.5,
                  }}>
                    {s.desc}
                  </p>
                </div>
                <CheckCircle2 size={18} color="var(--emerald)" style={{ flexShrink: 0 }} />
              </div>
            ))}
          </div>
        </section>

        {/* Section Divider */}
        <hr className="section-divider" style={{ maxWidth: 'var(--container-max)', margin: '0 auto' }} />

        {/* ═══════════════════════════════════════════════════════════════
            CALL TO ACTION
            ═══════════════════════════════════════════════════════════════ */}
        <section style={{
          padding: 'var(--section-gap) var(--container-padding)',
          maxWidth: 900,
          margin: '0 auto',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem',
        }}>
          <div
            className="card"
            style={{
              padding: '3.5rem 2rem',
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1rem',
              background: 'linear-gradient(135deg, #ffffff 0%, rgba(124, 92, 252, 0.04) 100%)',
              border: '1px solid var(--border)',
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <h2 style={{
              fontSize: 'clamp(1.75rem, 3vw, 2.25rem)',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.15,
              color: 'var(--text-main)',
            }}>
              Ready to Audit & Optimize Your Data?
            </h2>
            <p style={{
              color: 'var(--text-muted)',
              fontSize: '0.95rem',
              maxWidth: 560,
              lineHeight: 1.6,
            }}>
              Launch the AegisMind Dashboard and test your datasets with instant health diagnostics and explainable AI remediations.
            </p>
            <a href={dashboardUrl} className="btn-primary" style={{
              padding: '0.8rem 2rem',
              fontSize: '0.95rem',
              marginTop: '0.5rem',
            }}>
              <span>Launch AegisMind Dashboard</span>
              <ArrowRight size={16} />
            </a>
          </div>
        </section>
      </main>

      {/* ═══════════════════════════════════════════════════════════════
          FOOTER
          ═══════════════════════════════════════════════════════════════ */}
      <footer style={{
        padding: '1.75rem 2.5rem',
        borderTop: '1px solid var(--border)',
        background: 'var(--bg-card)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        fontSize: '0.8rem',
        color: 'var(--text-dim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={15} color="var(--accent)" />
          <span>AegisMind — AI Data Readiness & Model Recommendation Platform</span>
        </div>
        <div>
          <span>© 2026 AegisMind. All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
}
