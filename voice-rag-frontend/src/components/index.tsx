import { useState } from 'react';
import {
  CheckCircle2, AlertCircle, BookOpen, Zap,
  ChevronDown, ChevronUp, Wifi, WifiOff, Send, Loader2,
} from 'lucide-react';
import type { QueryResponse, HealthStatus, LatencyBreakdown } from '../types/api.types';
import { confidenceLevel, confidenceColor, toPipelineStages } from '../types/rag.types';


// ── WaveformVisualizer ────────────────────────────────────────────────────────

interface WaveformProps { bars: number[] }

export function WaveformVisualizer({ bars }: WaveformProps) {
  if (!bars.length) return null;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 2, height: 40, width: '100%', maxWidth: 300 }}>
      {bars.map((v, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            height: `${Math.max(8, Math.abs(v) * 100)}%`,
            background: '#6366f1',
            borderRadius: 2,
            transition: 'height 0.05s ease',
          }}
        />
      ))}
    </div>
  );
}


// ── TranscriptPanel ───────────────────────────────────────────────────────────

interface TranscriptProps { transcript: string }

export function TranscriptPanel({ transcript }: TranscriptProps) {
  if (!transcript) return null;
  return (
    <div className="transcript-card">
      <span className="section-label">You said</span>
      <p className="transcript-text">"{transcript}"</p>
    </div>
  );
}


// ── ConfidenceBadge ───────────────────────────────────────────────────────────

interface ConfidenceProps { value: number }

export function ConfidenceBadge({ value }: ConfidenceProps) {
  const pct = Math.round(value * 100);
  const level = confidenceLevel(value);
  const color = confidenceColor(level);
  return (
    <span style={{ color, fontWeight: 600, fontSize: 13 }}>
      {pct}% confidence
    </span>
  );
}


// ── CitationList ──────────────────────────────────────────────────────────────

interface CitationProps { citations: QueryResponse['citations'] }

export function CitationList({ citations }: CitationProps) {
  if (!citations.length) return null;
  return (
    <div className="citations">
      <div className="section-label"><BookOpen size={13} /> Sources</div>
      {citations.slice(0, 5).map((c, i) => (
        <div key={c.chunk_id} className="citation-item">
          <span className="citation-num">{i + 1}</span>
          <div>
            <div className="citation-id">{c.document_id}</div>
            <div className="citation-score">score: {c.score.toFixed(3)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}


// ── LatencyBreakdownPanel ─────────────────────────────────────────────────────

interface LatencyProps { lat: LatencyBreakdown }

export function LatencyBreakdownPanel({ lat }: LatencyProps) {
  const [open, setOpen] = useState(false);
  const stages = toPipelineStages(lat);

  return (
    <div className="latency-card">
      <button className="latency-toggle" onClick={() => setOpen(!open)}>
        <Zap size={14} />
        <span>Total: <strong>{lat.total.toFixed(1)}ms</strong></span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="latency-grid">
          {stages.map(({ name, latencyMs, pct }) => (
            <div key={name} className="latency-row">
              <span className="latency-label">{name}</span>
              <span className="latency-val">{latencyMs.toFixed(1)}ms</span>
              <div className="latency-bar-wrap">
                <div className="latency-bar-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ── AnswerPanel ───────────────────────────────────────────────────────────────

interface AnswerProps { result: QueryResponse }

export function AnswerPanel({ result }: AnswerProps) {
  return (
    <div className="answer-card">
      <div className="answer-header">
        <div className="section-label">Answer</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {result.grounded && !result.refused && (
            <span className="grounded-badge"><CheckCircle2 size={12} /> Grounded</span>
          )}
          {result.refused && (
            <span className="refused-badge"><AlertCircle size={12} /> Refused</span>
          )}
          <ConfidenceBadge value={result.confidence} />
        </div>
      </div>
      <p className="answer-text">{result.answer}</p>
      {result.query.normalized && (
        <div className="query-meta">
          <span>🌐 {result.query.language}</span>
          <span>🎯 {result.query.intent}</span>
          {result.retrieval.n_final > 0 && (
            <span>📄 {result.retrieval.n_final} passages</span>
          )}
        </div>
      )}
      <CitationList citations={result.citations} />
      <LatencyBreakdownPanel lat={result.latency_ms} />
    </div>
  );
}


// ── StatusIndicator ───────────────────────────────────────────────────────────

interface StatusProps { health: HealthStatus | null }

export function StatusIndicator({ health }: StatusProps) {
  return (
    <div className="health-indicator">
      {health?.status === 'healthy'
        ? <><Wifi size={14} className="health-ok" /><span>Online</span></>
        : health?.status === 'degraded'
        ? <><Wifi size={14} className="health-warn" /><span>Degraded</span></>
        : <><WifiOff size={14} className="health-err" /><span>Offline</span></>}
    </div>
  );
}


// ── ErrorBanner ───────────────────────────────────────────────────────────────

interface ErrorProps { message: string; className?: string }

export function ErrorBanner({ message, className = '' }: ErrorProps) {
  return (
    <div className={`error-banner ${className}`}>
      <AlertCircle size={14} /> {message}
    </div>
  );
}


// ── TextQueryInput ────────────────────────────────────────────────────────────

interface TextQueryProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  maxLength?: number;
}

export function TextQueryInput({ value, onChange, onSubmit, loading, maxLength = 500 }: TextQueryProps) {
  return (
    <div className="text-panel">
      <textarea
        className="text-input"
        placeholder="Type your question here…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        rows={3}
        maxLength={maxLength}
        disabled={loading}
      />
      <div className="text-actions">
        <span className="char-count">{value.length}/{maxLength}</span>
        <button
          className="submit-btn"
          onClick={onSubmit}
          disabled={!value.trim() || loading}
        >
          {loading ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
          {loading ? 'Processing…' : 'Ask'}
        </button>
      </div>
    </div>
  );
}
