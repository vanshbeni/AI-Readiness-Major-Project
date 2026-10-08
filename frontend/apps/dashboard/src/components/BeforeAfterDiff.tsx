import React from 'react';
import { ExecutionResult } from '../services/api';
import { CountUp, Panel, Reveal } from './ui';

interface BeforeAfterDiffProps {
  result: ExecutionResult;
}

const decimalsOf = (v: number) => (Number.isInteger(v) ? 0 : 1);

export const BeforeAfterDiff: React.FC<BeforeAfterDiffProps> = ({ result }) => {
  const delta = result.health_score_delta;
  const stamp = delta > 0
    ? { cls: 'good', word: 'improved', color: 'var(--good)' }
    : delta < 0
      ? { cls: 'bad', word: 'dropped · review fixes', color: 'var(--bad)' }
      : { cls: 'flat', word: 'unchanged', color: 'var(--muted)' };

  return (
    <div className="section-gap">
      <Reveal>
        <Panel>
          <div className="panel-head">
            <div className="left">
              <span className="dots"><i /><i /><i /></span>
              <span>health_score.diff</span>
            </div>
            <span>{result.cleaned_rows.toLocaleString()} rows × {result.cleaned_cols} cols after cleaning</span>
          </div>
          <div className="result-hero">
            <div className="result-cell">
              <div className="k">BEFORE</div>
              <div className="big" style={{ color: 'var(--dim)' }}>
                {result.before_health_score}<small> /100</small>
              </div>
              <div className="meter in"><i style={{ ['--w' as string]: `${result.before_health_score}%`, background: 'var(--dim)' }} /></div>
            </div>
            <div className="result-cell">
              <div className="k">AFTER</div>
              <div className="big" style={{ color: stamp.color }}>
                <CountUp value={result.after_health_score} from={result.before_health_score} decimals={decimalsOf(result.after_health_score)} duration={1500} />
                <small> /100</small>
              </div>
              <div className="meter in"><i style={{ ['--w' as string]: `${result.after_health_score}%`, ['--d' as string]: '300ms', background: stamp.color }} /></div>
            </div>
            <div className="stamp-cell">
              <div className={`stamp ${stamp.cls}`}>
                <b>{delta > 0 ? '+' : ''}{delta} pts</b>
                <span>{stamp.word}</span>
              </div>
            </div>
          </div>
          <div className="summary-line">
            <span className="tag">[log]</span>
            <span>{result.message}</span>
          </div>
        </Panel>
      </Reveal>

      <div className="two-col">
        <Reveal>
          <Panel>
            <div className="panel-head">
              <div className="left">
                <span className="dots"><i /><i /><i /></span>
                <span>quality.metrics</span>
              </div>
              <span>raw → cleaned</span>
            </div>
            <div className="table-wrap">
              <table className="mono-table">
                <thead>
                  <tr>
                    <th>dimension</th>
                    <th>before</th>
                    <th>after</th>
                    <th>impact</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metric_deltas.map((d, idx) => (
                    <tr key={idx}>
                      <td style={{ color: 'var(--ink)', fontWeight: 500 }}>{d.metric_name}</td>
                      <td style={{ color: 'var(--bad)' }}>{String(d.before_value)}</td>
                      <td style={{ color: 'var(--good)', fontWeight: 600 }}>{String(d.after_value)}</td>
                      <td>{d.improvement ? <span className="chip">{d.improvement}</span> : <span style={{ color: 'var(--dim)' }}>—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </Reveal>

        <Reveal delay={100}>
          <Panel>
            <div className="panel-head">
              <div className="left">
                <span className="dots"><i /><i /><i /></span>
                <span>pipeline.steps</span>
              </div>
              <span>{result.transformations_applied.length} applied</span>
            </div>
            {result.transformations_applied.length === 0 ? (
              <p style={{ padding: '1rem 1.25rem', fontSize: '0.86rem', color: 'var(--muted)' }}>
                No transformations were applied; the dataset was exported as-is.
              </p>
            ) : (
              <ol className="steps-list">
                {result.transformations_applied.map((step, i) => <li key={i}>{step}</li>)}
              </ol>
            )}
          </Panel>
        </Reveal>
      </div>
    </div>
  );
};
