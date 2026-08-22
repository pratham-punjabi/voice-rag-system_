/**
 * RAG domain types — higher-level than the raw API response types.
 * Used for UI state management and component props.
 */

import type { QueryResponse, LatencyBreakdown } from './api.types';

export interface PipelineStage {
  name: string;
  latencyMs: number;
  pct: number;       // share of total latency (0-100)
}

/**
 * Derive ordered pipeline stages from a latency breakdown for visualisation.
 */
export function toPipelineStages(lat: LatencyBreakdown): PipelineStage[] {
  const entries: [string, number][] = ([
    ['STT', lat.stt],
    ['Query Processing', lat.query_processing],
    ['Guardrail', lat.guardrail],
    ['Embedding', lat.embedding],
    ['Dense Retrieval', lat.dense_retrieval],
    ['BM25 Retrieval', lat.bm25_retrieval],
    ['Fusion', lat.fusion],
    ['Reranking', lat.reranking],
    ['Validation', lat.validation],
    ['Generation', lat.generation],
    ['Grounding', lat.grounding],
  ] as [string, number][]).filter(([, v]) => v > 0);

  const total = lat.total || entries.reduce((s, [, v]) => s + v, 0) || 1;

  return entries.map(([name, latencyMs]) => ({
    name,
    latencyMs,
    pct: Math.round((latencyMs / total) * 100),
  }));
}

export interface RAGSession {
  id: string;
  query: string;
  response: QueryResponse;
  timestamp: number;
}

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export function confidenceLevel(score: number): ConfidenceLevel {
  if (score >= 0.7) return 'high';
  if (score >= 0.4) return 'medium';
  return 'low';
}

export function confidenceColor(level: ConfidenceLevel): string {
  return { high: '#16a34a', medium: '#d97706', low: '#dc2626' }[level];
}

export interface RetrievalSummary {
  totalCandidates: number;
  finalPassages: number;
  topScore: number;
  passedValidation: boolean;
}

export function toRetrievalSummary(
  retrieval: QueryResponse['retrieval']
): RetrievalSummary {
  return {
    totalCandidates: retrieval.n_candidates,
    finalPassages: retrieval.n_final,
    topScore: retrieval.top_score,
    passedValidation: retrieval.passed_validation,
  };
}
