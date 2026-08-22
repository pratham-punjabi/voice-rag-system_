import type { QueryResponse, HealthStatus } from '../types/api.types';

const BASE = '/api';

export async function submitTextQuery(query: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function submitVoiceQuery(
  audioBlob: Blob,
  sampleRate = 16000,
): Promise<QueryResponse> {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.webm');
  form.append('sample_rate', String(sampleRate));

  const res = await fetch(`${BASE}/voice/query`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function submitVoiceTranscript(transcript: string): Promise<QueryResponse> {
  /**
   * Submit text transcribed by the browser's Web Speech API.
   * Used when no Sarvam STT is configured.
   */
  const form = new FormData();
  form.append('transcript', transcript);

  const res = await fetch(`${BASE}/voice/transcript`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchMetrics(): Promise<unknown> {
  const res = await fetch(`${BASE}/metrics`);
  if (!res.ok) throw new Error('Metrics unavailable');
  return res.json();
}

export async function fetchIngestStatus(): Promise<{
  total_chunks: number;
  has_data: boolean;
  data_source_mode: string;
}> {
  const res = await fetch(`${BASE}/ingest/status`);
  if (!res.ok) throw new Error('Ingest status unavailable');
  return res.json();
}
