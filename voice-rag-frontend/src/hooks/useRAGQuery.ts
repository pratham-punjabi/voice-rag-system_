import { useState, useCallback, useRef } from 'react';
import type { QueryResponse } from '../types/api.types';
import { submitTextQuery, submitVoiceQuery } from '../services/api';

export type QueryStatus = 'idle' | 'loading' | 'success' | 'error';

export interface UseRAGQueryResult {
  status: QueryStatus;
  result: QueryResponse | null;
  error: string | null;
  submitText: (query: string) => Promise<void>;
  submitAudio: (blob: Blob, sampleRate?: number) => Promise<void>;
  reset: () => void;
  abortCurrent: () => void;
}

export function useRAGQuery(): UseRAGQueryResult {
  const [status, setStatus] = useState<QueryStatus>('idle');
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setError(null);
  }, []);

  const abortCurrent = useCallback(() => {
    abortRef.current?.abort();
    setStatus('idle');
  }, []);

  const submitText = useCallback(async (query: string) => {
    if (!query.trim()) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await submitTextQuery(query);
      setResult(response);
      setStatus('success');
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : 'Request failed';
      setError(msg);
      setStatus('error');
    }
  }, []);

  const submitAudio = useCallback(async (blob: Blob, sampleRate = 16000) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await submitVoiceQuery(blob, sampleRate);
      setResult(response);
      setStatus('success');
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : 'Audio processing failed';
      setError(msg);
      setStatus('error');
    }
  }, []);

  return { status, result, error, submitText, submitAudio, reset, abortCurrent };
}
