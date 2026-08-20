import React from 'react';
import { UploadCloud, Target, Activity, CheckSquare, Zap } from 'lucide-react';

export type StepKey = 'upload' | 'objective' | 'diagnostic' | 'recommendations' | 'results';

interface StepperProps {
  currentStep: StepKey;
  onStepClick?: (step: StepKey) => void;
  completedSteps: StepKey[];
}

const STEPS: { key: StepKey; label: string; icon: React.ComponentType<{ size: number }> }[] = [
  { key: 'upload', label: '1. Ingestion', icon: UploadCloud },
  { key: 'objective', label: '2. ML Objective', icon: Target },
  { key: 'diagnostic', label: '3. Data Health', icon: Activity },
  { key: 'recommendations', label: '4. Explainable Fixes', icon: CheckSquare },
  { key: 'results', label: '5. Cleaned & Models', icon: Zap },
];

export const Stepper: React.FC<StepperProps> = ({ currentStep, onStepClick, completedSteps }) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.75rem',
      padding: '1.25rem 2rem',
      maxWidth: 1000,
      margin: '0 auto',
    }}>
      {STEPS.map((step, idx) => {
        const Icon = step.icon;
        const isActive = currentStep === step.key;
        const isCompleted = completedSteps.includes(step.key);
        const isClickable = isCompleted || isActive;

        return (
          <React.Fragment key={step.key}>
            <button
              onClick={() => isClickable && onStepClick?.(step.key)}
              disabled={!isClickable}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 0.9rem',
                borderRadius: 10,
                background: isActive
                  ? 'rgba(56, 189, 248, 0.12)'
                  : isCompleted
                  ? 'rgba(16, 185, 129, 0.08)'
                  : 'transparent',
                border: isActive
                  ? '1px solid #38bdf8'
                  : isCompleted
                  ? '1px solid rgba(16, 185, 129, 0.3)'
                  : '1px solid transparent',
                color: isActive
                  ? '#38bdf8'
                  : isCompleted
                  ? '#34d399'
                  : 'var(--text-dim)',
                cursor: isClickable ? 'pointer' : 'not-allowed',
                fontWeight: isActive ? 700 : 500,
                fontSize: '0.85rem',
                transition: 'all 0.2s ease',
              }}
            >
              <Icon size={16} />
              <span>{step.label}</span>
            </button>
            {idx < STEPS.length - 1 && (
              <div style={{
                width: 24,
                height: 1,
                background: isCompleted ? 'rgba(16, 185, 129, 0.4)' : 'rgba(255, 255, 255, 0.1)',
              }} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
