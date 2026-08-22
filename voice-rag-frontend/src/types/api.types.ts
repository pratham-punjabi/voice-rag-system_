export type RecordingState = 'idle' | 'recording' | 'processing' | 'error';

export interface QueryResponse {
  request_id: string;
  query: string;
  answer: string;
  transcript?: string;
  confidence: number;
  grounded: boolean;
  citations: Array<{
    document_id: string;
    chunk_id: string;
    score: number;
    rerank_score?: number;
  }>;
  latency_ms: {
    total?: number;
    embedding?: number;
    retrieval?: number;
    reranking?: number;
    generation?: number;
    [key: string]: number | undefined;
  };
  data_source_mode: string;
  retrieved_docs: number;
  has_context: boolean;
  cached?: boolean;
  source?: string;
}

export interface HealthStatus {
  status: string;
  index_loaded: boolean;
  bm25_loaded: boolean;
  llm_available: boolean;
  stt_available: boolean;
  data_source_mode?: string;
  chroma_chunks?: number;
  components: Record<string, unknown>;
}
