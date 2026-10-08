'use client';

import React from 'react';
import { Check } from 'lucide-react';
import { Eyebrow } from './primitives';
import { clamp01, lerp, useStickyProgress } from '../lib/motion';

type Cell = string | { dirty: string; clean: string };
type Row = { cells: Cell[]; note?: string; fixedNote?: string; strike?: boolean; log?: string };

const COLUMNS = ['PassengerId', 'Pclass', 'Age', 'Fare', 'Embarked', 'Survived'];

const ROWS: Row[] = [
  { cells: ['1', '3', '22.0', '7.25', 'S', '0'] },
  { cells: ['2', '1', { dirty: 'NaN', clean: '28.0' }, '71.28', 'C', '1'], note: 'missing?', fixedNote: 'median', log: 'Impute Age with median (28.0)' },
  { cells: ['3', '3', '26.0', '7.92', 'S', '1'] },
  { cells: ['904', '3', '26.0', '7.92', 'S', '1'], note: 'same person!', fixedNote: 'dropped', strike: true, log: 'Drop 4 duplicate passengers' },
  { cells: ['5', '1', '35.0', { dirty: '512.33', clean: '65.63' }, 'S', '1'], note: 'outlier', fixedNote: 'capped', log: 'Cap Fare at IQR bound 65.63' },
  { cells: ['6', '3', { dirty: '-3.0', clean: '28.0' }, '8.46', 'Q', '0'], note: 'age -3 ??', fixedNote: 'fixed', log: 'Replace negative Age with median' },
  { cells: ['7', '2', '54.0', '51.86', { dirty: 'NaN', clean: 'S' }, '0'], note: 'empty', fixedNote: 'mode', log: "Fill Embarked with mode 'S'" },
  { cells: ['8', '3', '2.0', '21.08', 'S', { dirty: 'NaN', clean: '—' }], note: 'no label', fixedNote: 'dropped', strike: true, log: 'Drop row with missing target' },
  { cells: ['9', '3', '27.0', '11.13', 'S', '1'] },
];

const START = 0.06;
const SPAN = 0.8;

export function Scanner() {
  const [ref, progress] = useStickyProgress<HTMLElement>(60);
  const scan = clamp01((progress - START) / SPAN);
  const scanRow = scan * ROWS.length;
  const isFixed = (i: number) => scanRow > i + 0.55;

  const logRows = ROWS.map((r, i) => ({ ...r, i })).filter((r) => r.log);
  const fixedCount = logRows.filter((r) => isFixed(r.i)).length;
  const score = Math.round(lerp(54, 94, fixedCount / logRows.length));
  const scoreColor = score >= 85 ? 'var(--good)' : score >= 70 ? 'var(--warn)' : 'var(--bad)';

  return (
    <section ref={ref} className="scan-section" id="scanner">
      <div className="guides" />
      <div className="scan-sticky">
        <div className="frame" style={{ width: '100%' }}>
          <div className="scan-grid">
            <div className="table-card">
              <div className="table-top">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className="dots"><i /><i /><i /></span>
                  <span>titanic.csv — 304 rows</span>
                </div>
                <span className="live-score">
                  health <span style={{ color: scoreColor }}>{score}</span>
                  <span style={{ color: 'var(--dim)', fontWeight: 400 }}>/100</span>
                </span>
              </div>
              <div className="table-body" style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      {COLUMNS.map((c) => <th key={c}>{c}</th>)}
                      <th aria-label="notes" />
                    </tr>
                  </thead>
                  <tbody>
                    {ROWS.map((row, i) => {
                      const fixed = isFixed(i);
                      return (
                        <tr key={i} className={row.strike && fixed ? 'struck' : ''}>
                          {row.cells.map((cell, ci) => {
                            if (typeof cell === 'string') {
                              return (
                                <td key={ci}>
                                  <div className={`cell ${row.strike && !fixed ? 'dirty' : ''}`}>{cell}</div>
                                </td>
                              );
                            }
                            return (
                              <td key={ci}>
                                <div className={`cell ${fixed ? 'fixed' : 'dirty'}`}>
                                  <span className="v v-dirty">{cell.dirty}</span>
                                  <span className="v v-clean">{cell.clean}</span>
                                </div>
                              </td>
                            );
                          })}
                          <td>
                            <div className="cell">
                              {row.note && (
                                <span className={`row-note ${fixed ? 'resolved' : ''}`}>
                                  {fixed ? `✓ ${row.fixedNote}` : `← ${row.note}`}
                                </span>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {scan > 0 && scan < 1 && (
                  <div className="scan-area">
                    <div className="scanner" style={{ top: `${scan * 100}%` }}>
                      <span className="scan-tag">scanning row {Math.min(ROWS.length, Math.floor(scanRow) + 1)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="scan-copy">
              <Eyebrow label="live scan" index="01/05" />
              <h2>Scroll. Watch the data get honest.</h2>
              <p>
                Every red cell is something a model would silently learn from: gaps, a passenger counted twice under a
                new ID, a negative age, a 512-pound fare. Nebulus finds them, picks a fix that suits each column, and
                explains why.
              </p>

              <div className="score-row">
                <div className="score-box">
                  <div className="k">BEFORE</div>
                  <div className="num" style={{ color: 'var(--bad)' }}>54<small> /100</small></div>
                  <div className="meter"><i style={{ width: '54%', background: 'var(--bad)' }} /></div>
                </div>
                <div className="score-box">
                  <div className="k">NOW</div>
                  <div className="num" style={{ color: scoreColor }}>{score}<small> /100</small></div>
                  <div className="meter"><i style={{ width: `${score}%`, background: scoreColor }} /></div>
                </div>
              </div>

              <ul className="fix-log">
                {logRows.map((r) => (
                  <li key={r.i} className={isFixed(r.i) ? 'done' : ''}>
                    <span className="tick">{isFixed(r.i) && <Check size={10} color="#fff" strokeWidth={3} />}</span>
                    {r.log}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
