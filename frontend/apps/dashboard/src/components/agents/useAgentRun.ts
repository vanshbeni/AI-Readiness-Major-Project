import { useCallback, useEffect, useRef, useState } from 'react';
import { api, AgentKind, AgentRun } from '../../services/api';

const POLL_MS = 700;
const LIVE = new Set(['queued', 'running']);

/** Starts / follows an agent run: polls the live trace and exposes resume + cancel. */
export function useAgentRun(onFinished?: (run: AgentRun) => void) {
  const [run, setRun] = useState<AgentRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const finishedRef = useRef(onFinished);
  finishedRef.current = onFinished;

  const stop = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };

  const poll = useCallback((current: AgentRun) => {
    stop();
    timer.current = setTimeout(async () => {
      try {
        const fresh = await api.getAgentRun(current.id, current.steps.length);
        const merged: AgentRun = { ...fresh, steps: [...current.steps, ...fresh.steps] };
        setRun(merged);
        if (LIVE.has(merged.status)) poll(merged);
        else if (merged.status === 'completed' || merged.status === 'failed') finishedRef.current?.(merged);
      } catch (e: any) {
        setError(e.message || 'Lost contact with the agent');
      }
    }, POLL_MS);
  }, []);

  useEffect(() => stop, []);

  const start = useCallback(async (datasetId: string, kind: AgentKind, body: Parameters<typeof api.startAgent>[2] = {}) => {
    setError(null);
    try {
      const created = await api.startAgent(datasetId, kind, body);
      setRun(created);
      poll(created);
      return created;
    } catch (e: any) {
      setError(e.message || 'Could not start the agent');
      return null;
    }
  }, [poll]);

  const resume = useCallback(async (approvals: Record<string, boolean>, answers: Record<string, number>) => {
    if (!run) return;
    setError(null);
    try {
      const updated = await api.resumeAgentRun(run.id, approvals, answers);
      const merged = { ...run, status: updated.status, gate: null };
      setRun(merged);
      poll(merged);
    } catch (e: any) {
      setError(e.message || 'Could not send your decision');
    }
  }, [run, poll]);

  const cancel = useCallback(async () => {
    if (!run) return;
    stop();
    try {
      const updated = await api.cancelAgentRun(run.id);
      setRun({ ...run, status: updated.status, gate: null });
    } catch (e: any) {
      setError(e.message || 'Could not cancel');
    }
  }, [run]);

  const clear = useCallback(() => {
    stop();
    setRun(null);
    setError(null);
  }, []);

  return { run, error, start, resume, cancel, clear };
}
