import React, { useEffect, useState } from 'react';
import { FileSpreadsheet, RotateCcw } from 'lucide-react';
import { api } from '../services/api';

interface HeaderProps {
  datasetName?: string;
  onReset?: () => void;
}

const HEALTH_POLL_MS = 15000;
const LANDING_URL = process.env.NEXT_PUBLIC_LANDING_URL || 'http://localhost:3001';

export const Header: React.FC<HeaderProps> = ({ datasetName, onReset }) => {
  const [engineOnline, setEngineOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    const check = () => api.checkHealth().then((ok) => active && setEngineOnline(ok));
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const statusClass = engineOnline === null ? '' : engineOnline ? 'on' : 'off';
  const statusText = engineOnline === null ? 'checking engine' : engineOnline ? 'engine online' : 'engine offline';

  return (
    <header className="nav">
      <div className="frame nav-inner">
        <a href={LANDING_URL} className="logo" aria-label="Nebulus home">
          <svg className="logo-mark" viewBox="0 0 26 26" aria-hidden>
            <rect x="1" y="1" width="24" height="24" rx="7" fill="#6d4aff" />
            <path d="M8 18V8l10 10V8" stroke="#fff" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Nebulus <small>/ dashboard</small>
        </a>

        <div className="nav-right">
          {datasetName && (
            <span className="file-pill" title={datasetName}>
              <FileSpreadsheet size={13} color="var(--accent)" />
              <span>{datasetName}</span>
            </span>
          )}

          <span
            className={`status ${statusClass}`}
            title={engineOnline === false ? 'The backend API is not reachable' : undefined}
          >
            <i />
            <span>{statusText}</span>
          </span>

          {onReset && datasetName && (
            <button onClick={onReset} className="btn btn-light btn-sm" title="Upload a new dataset">
              <RotateCcw size={13} />
              New dataset
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
