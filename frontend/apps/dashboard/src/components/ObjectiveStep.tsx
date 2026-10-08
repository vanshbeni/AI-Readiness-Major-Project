import React, { useEffect, useState } from 'react';
import { ArrowRight, Check, Loader2, Layers, TrendingUp } from 'lucide-react';
import { api, DatasetSummary, DatasetSample, Objective, TargetSuggestion } from '../services/api';
import { Eyebrow, MagneticButton, Note, Panel, Reveal, Scramble, Underlined } from './ui';

interface ObjectiveStepProps {
  dataset: DatasetSummary;
  sample: DatasetSample;
  initialObjective?: Objective | null;
  onSubmit: (problemType: string, targetColumn: string) => Promise<void>;
}

const TARGET_NAME_HINTS = ['target', 'label', 'class', 'survived', 'churn', 'outcome', 'y', 'price', 'medianhousevalue'];

const PROBLEM_TYPES = [
  {
    key: 'classification',
    icon: Layers,
    title: 'Classification',
    body: 'Predict a category or yes/no outcome: churn vs retain, survived vs perished.',
  },
  {
    key: 'regression',
    icon: TrendingUp,
    title: 'Regression',
    body: 'Predict a continuous number: house price, temperature, revenue.',
  },
];

export const ObjectiveStep: React.FC<ObjectiveStepProps> = ({ dataset, sample, initialObjective, onSubmit }) => {
  const cols = sample.columns;
  const nameGuess = cols.find((c) => TARGET_NAME_HINTS.includes(c.toLowerCase())) || cols[cols.length - 1] || '';

  const [suggestions, setSuggestions] = useState<Record<string, TargetSuggestion> | null>(null);
  const [problemType, setProblemType] = useState<string>(initialObjective?.problem_type || 'classification');
  const [targetColumn, setTargetColumn] = useState<string>(initialObjective?.target_column || nameGuess);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    api.getTargetSuggestions(dataset.id)
      .then((s) => {
        if (!active) return;
        setSuggestions(s);
        if (initialObjective) return;
        const preferred = s[nameGuess]?.suitable ? nameGuess : [...cols].reverse().find((c) => s[c]?.suitable) || nameGuess;
        setTargetColumn(preferred);
        const pt = s[preferred]?.problem_type;
        if (pt) setProblemType(pt);
      })
      .catch(() => undefined);
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.id]);

  const handleTargetChange = (col: string) => {
    setTargetColumn(col);
    const pt = suggestions?.[col]?.problem_type;
    if (pt) setProblemType(pt);
  };

  const suggestion = suggestions?.[targetColumn];
  const warning = !suggestion
    ? null
    : !suggestion.suitable
      ? `'${targetColumn}' looks unsuitable as a target (${suggestion.reason}).`
      : suggestion.problem_type && suggestion.problem_type !== problemType
        ? `'${targetColumn}' looks like a ${suggestion.problem_type} target (${suggestion.reason}).`
        : null;

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await onSubmit(problemType, targetColumn);
    } catch {
      // The page shows the error banner; just stop the spinner.
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="step-enter" style={{ maxWidth: 980, margin: '0 auto' }}>
      <div className="step-head">
        <Eyebrow label="objective" index="02/05" caret />
        <h1 className="h-title">
          <Scramble text="What should it" /> <Underlined>predict?</Underlined>
        </h1>
        <p className="h-sub">
          Pick the task and the column your models will learn. Every score and fix that follows is tuned to this choice.
        </p>
      </div>

      <div className="section-gap">
        <Reveal>
          <div className="field-label"><b>01</b> problem type</div>
          <div className="choice-grid">
            {PROBLEM_TYPES.map(({ key, icon: Icon, title, body }) => {
              const on = problemType === key;
              return (
                <button key={key} type="button" className={`choice ${on ? 'on' : ''}`} onClick={() => setProblemType(key)} aria-pressed={on}>
                  <span className="tick">{on && <Check size={11} color="#fff" strokeWidth={3} />}</span>
                  <Icon size={20} color={on ? 'var(--accent)' : 'var(--dim)'} />
                  <h4>{title}</h4>
                  <p>{body}</p>
                </button>
              );
            })}
          </div>
        </Reveal>

        <Reveal delay={80}>
          <div className="field-label" style={{ justifyContent: 'space-between' }}>
            <span><b>02</b> target column</span>
            {suggestion?.suitable && !warning && <span className="chip good">looks like a good target</span>}
          </div>
          <select className="select" value={targetColumn} onChange={(e) => handleTargetChange(e.target.value)}>
            {cols.map((c) => (
              <option key={c} value={c}>
                {c}{suggestions?.[c] && !suggestions[c].suitable ? '  (not recommended)' : ''}
              </option>
            ))}
          </select>
          {warning && (
            <div className="banner info" style={{ marginTop: '0.75rem', marginBottom: 0, borderColor: 'rgba(217,139,6,.35)', background: '#fffbf2' }}>
              <span className="tag" style={{ color: 'var(--warn)' }}>[heads-up]</span>
              <span>{warning}</span>
            </div>
          )}
        </Reveal>

        <Reveal delay={140}>
          <Panel>
            <div className="panel-head">
              <div className="left">
                <span className="dots"><i /><i /><i /></span>
                <span>~/{dataset.filename}</span>
              </div>
              <span>head(5) of {dataset.row_count.toLocaleString()} rows</span>
            </div>
            <div className="table-wrap">
              <table className="mono-table">
                <thead>
                  <tr>
                    {sample.columns.map((c) => (
                      <th key={c} className={c === targetColumn ? 'hl' : ''}>
                        {c}{c === targetColumn && ' ★'}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sample.sample_data.slice(0, 5).map((row, rIdx) => (
                    <tr key={rIdx}>
                      {sample.columns.map((c) => {
                        const isNull = row[c] === null;
                        return (
                          <td key={c} className={isNull ? 'null' : c === targetColumn ? 'hl' : ''}>
                            {isNull ? 'null' : String(row[c])}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </Reveal>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '1.25rem', flexWrap: 'wrap' }}>
          <Note tone="purple" rot={-4} delay={1.1}>takes a few seconds</Note>
          <MagneticButton className="btn btn-dark btn-lg" onClick={handleSubmit} disabled={loading || !targetColumn}>
            {loading ? (
              <>
                <Loader2 size={17} className="animate-spin" />
                Profiling & scoring…
              </>
            ) : (
              <>
                Run diagnostics on <span className="mono" style={{ fontWeight: 500 }}>{targetColumn}</span>
                <ArrowRight size={16} className="arrow" />
              </>
            )}
          </MagneticButton>
        </div>
      </div>
    </div>
  );
};
