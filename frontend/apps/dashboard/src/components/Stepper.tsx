import React from 'react';
import { Check } from 'lucide-react';

export type StepKey = 'upload' | 'objective' | 'diagnostic' | 'recommendations' | 'results';

interface StepperProps {
  currentStep: StepKey;
  onStepClick?: (step: StepKey) => void;
  completedSteps: StepKey[];
}

export const STEPS: { key: StepKey; label: string }[] = [
  { key: 'upload', label: 'ingest' },
  { key: 'objective', label: 'objective' },
  { key: 'diagnostic', label: 'health score' },
  { key: 'recommendations', label: 'approve fixes' },
  { key: 'results', label: 'clean & benchmark' },
];

export const stepIndex = (key: StepKey) => STEPS.findIndex((s) => s.key === key);

export const Stepper: React.FC<StepperProps> = ({ currentStep, onStepClick, completedSteps }) => {
  const activeIdx = stepIndex(currentStep);

  return (
    <nav className="stepper frame" aria-label="Progress">
      <div className="stepper-track">
        {STEPS.map((step, idx) => {
          const isActive = currentStep === step.key;
          const isDone = completedSteps.includes(step.key) && !isActive;
          const clickable = isDone;
          return (
            <button
              key={step.key}
              type="button"
              className={`step ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}
              onClick={() => clickable && onStepClick?.(step.key)}
              disabled={!clickable && !isActive}
              aria-current={isActive ? 'step' : undefined}
            >
              <span className="n">{isDone ? <Check size={12} strokeWidth={3} /> : `0${idx + 1}`}</span>
              <span className="label">{step.label}</span>
              {isActive && <span className="caret" style={{ marginLeft: 'auto', height: 11, width: 6 }} />}
            </button>
          );
        })}
        <span className="stepper-fill" style={{ width: `${((activeIdx + 1) / STEPS.length) * 100}%` }} />
      </div>
    </nav>
  );
};
