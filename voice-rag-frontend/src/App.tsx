import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Mic, MicOff, Square, Copy, ThumbsUp, ThumbsDown,
  LayoutDashboard, Database, MessageSquare, BarChart2,
  Settings, ChevronRight, Moon, Sun, Zap, FileText,
  Globe, ExternalLink, CheckCircle2, AlertCircle, Send,
  Loader2, Radio
} from 'lucide-react';
import { useVoiceRecorder } from './hooks/useVoiceRecorder';
import {
  submitTextQuery,
  submitVoiceQuery,
  submitVoiceTranscript,
  fetchHealth,
  fetchIngestStatus
} from './services/api';
import type { QueryResponse, HealthStatus } from './types/api.types';

// ─── Types ───────────────────────────────────────────────────────────────────
type NavItem = 'dashboard' | 'knowledge' | 'conversations' | 'analytics' | 'settings';
type Theme = 'dark' | 'light';

interface ConversationItem {
  id: string;
  question: string;
  time: string;
  answer?: string;
  mode?: string;
}

interface SuggestedQuestion {
  icon: string;
  text: string;
}

const SUGGESTED_QUESTIONS: SuggestedQuestion[] = [
  { icon: '🧠', text: 'What is retrieval-augmented generation?' },
  { icon: '⚡', text: 'How does hybrid search work in RAG?' },
  { icon: '📊', text: 'How do you evaluate a RAG pipeline?' },
  { icon: '🔍', text: 'What is a cross-encoder reranker?' },
  { icon: '💡', text: 'Difference between dense and sparse retrieval?' },
  { icon: '🛡️', text: 'How to prevent hallucinations in RAG?' },
];

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  showSources?: boolean;
  mode?: string;
  confidence?: number;
  docs?: number;
}

const INITIAL_CHAT: ChatMessage[] = [
  {
    role: 'assistant',
    content: '👋 Hello! I\'m your Voice RAG assistant powered by Groq & ChromaDB.\n\nYou can:\n- **Type** a question and press Send\n- **Click the mic** and speak — your words appear live as you talk\n- The system automatically searches your knowledge base and uses Groq AI to answer\n\nWhat would you like to know?'
  }
];

// ─── Waveform Component ───────────────────────────────────────────────────────
function AnimatedWaveform({ active, bars }: { active: boolean; bars: number[] }) {
  return (
    <div className="waveform-container">
      {Array.from({ length: 32 }).map((_, i) => {
        const barValue = bars[i] ?? 0;
        const height = active
          ? Math.max(4, Math.abs(barValue) * 60 + 4 + Math.sin(Date.now() / 200 + i) * 10)
          : 4 + Math.sin(i * 0.5) * 2;
        return (
          <div
            key={i}
            className={`waveform-bar ${active ? 'waveform-bar--active' : ''}`}
            style={{ height: `${height}px` }}
          />
        );
      })}
    </div>
  );
}

// ─── Source Chips ─────────────────────────────────────────────────────────────
function SourceChips({ mode, docs }: { mode?: string; docs?: number }) {
  if (!docs && mode === 'api') {
    return (
      <div className="source-chips">
        <span className="source-chip">
          <Globe size={11} /> Groq AI Knowledge
        </span>
      </div>
    );
  }
  if (docs && docs > 0) {
    return (
      <div className="source-chips">
        <span className="source-chip">
          <FileText size={11} /> {docs} ChromaDB chunks
        </span>
        {mode && (
          <span className="source-chip">
            <Globe size={11} /> Mode: {mode}
          </span>
        )}
      </div>
    );
  }
  return null;
}

// ─── Chat Bubble ──────────────────────────────────────────────────────────────
function ChatBubble({
  role, content, showSources, showActions, mode, docs
}: {
  role: 'user' | 'assistant';
  content: string;
  showSources?: boolean;
  showActions?: boolean;
  mode?: string;
  docs?: number;
}) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<null | boolean>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className={`chat-bubble-wrap ${role === 'user' ? 'chat-bubble-wrap--user' : ''}`}>
      {role === 'assistant' && (
        <div className="avatar-ai">
          <Zap size={12} />
        </div>
      )}
      <div className={`chat-bubble ${role === 'user' ? 'chat-bubble--user' : 'chat-bubble--ai'}`}>
        <div className="chat-content" dangerouslySetInnerHTML={{
          __html: content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br/>')
            .replace(/^\d+\.\s/gm, (m) => `<span class="list-num">${m}</span>`)
        }} />
        {showSources && <SourceChips mode={mode} docs={docs} />}
        {showActions && (
          <div className="chat-actions">
            <button
              className={`action-btn ${liked === true ? 'action-btn--active' : ''}`}
              onClick={() => setLiked(liked === true ? null : true)}
              title="Helpful"
            >
              <ThumbsUp size={13} />
            </button>
            <button
              className={`action-btn ${liked === false ? 'action-btn--active-neg' : ''}`}
              onClick={() => setLiked(liked === false ? null : false)}
              title="Not helpful"
            >
              <ThumbsDown size={13} />
            </button>
            <button className="action-btn" onClick={handleCopy} title="Copy">
              {copied ? <CheckCircle2 size={13} /> : <Copy size={13} />}
            </button>
          </div>
        )}
      </div>
      {role === 'user' && <div className="avatar-user">U</div>}
    </div>
  );
}

// ─── Live Transcript Box ──────────────────────────────────────────────────────
function LiveTranscriptBox({
  transcript,
  isListening,
}: {
  transcript: string;
  isListening: boolean;
}) {
  if (!isListening && !transcript) return null;
  return (
    <div className="live-transcript-box">
      <div className="live-transcript-header">
        <Radio size={12} className="live-dot" />
        <span>Listening…</span>
      </div>
      <div className="live-transcript-text">
        {transcript || <span className="live-transcript-placeholder">Speak now — your words will appear here</span>}
      </div>
    </div>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({ activeNav, onNav, indexStatus }: {
  activeNav: NavItem;
  onNav: (n: NavItem) => void;
  indexStatus: { chunks: number; mode: string };
}) {
  const navItems: { id: NavItem; icon: React.ReactNode; label: string }[] = [
    { id: 'dashboard', icon: <LayoutDashboard size={16} />, label: 'Dashboard' },
    { id: 'knowledge', icon: <Database size={16} />, label: 'Knowledge Base' },
    { id: 'conversations', icon: <MessageSquare size={16} />, label: 'Conversations' },
    { id: 'analytics', icon: <BarChart2 size={16} />, label: 'Analytics' },
    { id: 'settings', icon: <Settings size={16} />, label: 'Settings' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon-wrap">
          <Mic size={18} />
        </div>
        <div>
          <div className="logo-name">VoiceRAG</div>
          <div className="logo-sub">Groq + ChromaDB</div>
        </div>
      </div>

      <button className="new-chat-btn">
        <span>+</span> New Chat
      </button>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeNav === item.id ? 'nav-item--active' : ''}`}
            onClick={() => onNav(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
            {activeNav === item.id && <ChevronRight size={14} className="nav-chevron" />}
          </button>
        ))}
      </nav>

      <div className="usage-card">
        <div className="usage-header">
          <span className="usage-title">ChromaDB Index</span>
          <span className="usage-count">{indexStatus.chunks.toLocaleString()} chunks</span>
        </div>
        <div className="usage-bar-track">
          <div
            className="usage-bar-fill"
            style={{ width: `${Math.min(100, (indexStatus.chunks / 50000) * 100)}%` }}
          />
        </div>
        <div className="usage-reset">Mode: {indexStatus.mode}</div>
      </div>
    </aside>
  );
}

// ─── Right Sidebar ────────────────────────────────────────────────────────────
function RightSidebar({
  conversations,
  onSelectConversation,
  onSelectSuggestion,
}: {
  conversations: ConversationItem[];
  onSelectConversation: (c: ConversationItem) => void;
  onSelectSuggestion: (q: string) => void;
}) {
  return (
    <aside className="right-sidebar">
      {conversations.length > 0 && (
        <div className="rs-section">
          <div className="rs-header">
            <span>Recent Conversations</span>
          </div>
          <div className="rs-conv-list">
            {conversations.slice(0, 5).map((c) => (
              <button
                key={c.id}
                className="rs-conv-item"
                onClick={() => onSelectConversation(c)}
              >
                <div className="rs-conv-icon"><MessageSquare size={12} /></div>
                <div className="rs-conv-content">
                  <div className="rs-conv-q">{c.question}</div>
                  <div className="rs-conv-time">{c.time}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="rs-section">
        <div className="rs-header">
          <span>Suggested Questions</span>
        </div>
        <div className="rs-sugg-list">
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q.text}
              className="rs-sugg-item"
              onClick={() => onSelectSuggestion(q.text)}
            >
              <span className="rs-sugg-icon">{q.icon}</span>
              <span>{q.text}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeNav, setActiveNav] = useState<NavItem>('dashboard');
  const [theme, setTheme] = useState<Theme>('dark');
  const [isListening, setIsListening] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_CHAT);
  const [isTyping, setIsTyping] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [indexStatus, setIndexStatus] = useState({ chunks: 0, mode: 'hybrid' });
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const recorder = useVoiceRecorder();

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping, recorder.liveTranscript]);

  // Theme class on body
  useEffect(() => {
    document.body.classList.toggle('theme-light', theme === 'light');
  }, [theme]);

  // Fetch index status on mount
  useEffect(() => {
    fetchIngestStatus()
      .then((s) => setIndexStatus({ chunks: s.total_chunks, mode: s.data_source_mode }))
      .catch(() => {});
  }, []);

  // Handle voice recording result
  useEffect(() => {
    if (recorder.state === 'processing') {
      const finalText = recorder.finalTranscript.trim();

      if (finalText) {
        // We have browser STT transcript — use it directly
        setIsListening(false);
        addMessage('user', finalText);
        sendQuery(finalText, 'browser_stt');
        recorder.reset();
      } else if (recorder.audioBlob) {
        // No browser STT (or empty) — fall back to sending audio to Sarvam
        setIsListening(false);
        submitVoiceQuery(recorder.audioBlob)
          .then((r: QueryResponse) => {
            if (r.transcript) {
              addMessage('user', r.transcript);
            }
            handleQueryResponse(r);
            recorder.reset();
          })
          .catch(() => {
            addMessage('assistant', 'Sorry, I could not understand the audio. Please try again or type your question.');
            recorder.reset();
          });
      } else {
        recorder.reset();
        setIsListening(false);
      }
    }
  }, [recorder.state, recorder.audioBlob, recorder.finalTranscript]);

  const addMessage = useCallback((role: 'user' | 'assistant', content: string, meta?: Partial<ChatMessage>) => {
    setMessages((prev) => [...prev, { role, content, ...meta } as ChatMessage]);
  }, []);

  const handleQueryResponse = useCallback((r: QueryResponse) => {
    setIsTyping(false);
    addMessage('assistant', r.answer || 'I could not generate a response.', {
      showSources: true,
      mode: r.data_source_mode,
      docs: r.retrieved_docs,
    });
  }, [addMessage]);

  const sendQuery = useCallback(async (query: string, source: string = 'text') => {
    setIsTyping(true);
    try {
      let result: QueryResponse;
      if (source === 'browser_stt') {
        result = await submitVoiceTranscript(query);
      } else {
        result = await submitTextQuery(query);
      }
      handleQueryResponse(result);

      // Track conversation
      setConversations((prev) => [
        {
          id: Date.now().toString(),
          question: query.slice(0, 60) + (query.length > 60 ? '…' : ''),
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          answer: result.answer,
          mode: result.data_source_mode,
        },
        ...prev,
      ]);
    } catch (err) {
      setIsTyping(false);
      const msg = err instanceof Error ? err.message : 'An error occurred';
      addMessage('assistant', `❌ Error: ${msg}. Please check your API configuration and try again.`);
    }
  }, [handleQueryResponse, addMessage]);

  const handleQuestion = useCallback((question: string) => {
    if (!question.trim()) return;
    addMessage('user', question);
    sendQuery(question);
  }, [addMessage, sendQuery]);

  const handleTextSend = useCallback(() => {
    const q = textInput.trim();
    if (!q) return;
    setTextInput('');
    handleQuestion(q);
  }, [textInput, handleQuestion]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTextSend();
    }
  }, [handleTextSend]);

  const handleMicClick = useCallback(async () => {
    if (isListening) {
      setIsListening(false);
      recorder.stopRecording();
    } else {
      setIsListening(true);
      try {
        await recorder.startRecording();
      } catch {
        setIsListening(false);
      }
    }
  }, [isListening, recorder]);

  const handleSelectConversation = useCallback((c: ConversationItem) => {
    setMessages([
      INITIAL_CHAT[0],
      { role: 'user', content: c.question },
      { role: 'assistant', content: c.answer || 'No answer available.', showSources: true, mode: c.mode },
    ]);
  }, []);

  return (
    <div className={`app-shell ${theme}`}>
      <Sidebar activeNav={activeNav} onNav={setActiveNav} indexStatus={indexStatus} />

      <div className="main-column">
        {/* Top Nav */}
        <header className="topnav">
          <div className="topnav-status">
            <span className="status-dot" />
            <span className="status-text">RAG Online · Groq + ChromaDB</span>
          </div>
          <div className="topnav-right">
            <button className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle theme">
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <div className="user-avatar"><span>U</span></div>
            <span className="user-label">User</span>
          </div>
        </header>

        {/* Hero Card */}
        <div className="hero-card">
          <div className="hero-text">
            <h1 className="hero-heading">
              Speak. Type. <span className="hero-gradient">Get Answered.</span>
            </h1>
            <p className="hero-desc">
              Voice RAG powered by Groq AI & ChromaDB — your words appear live as you speak.
            </p>
          </div>
          <div className="hero-wave">
            <AnimatedWaveform active={false} bars={[]} />
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          <div className="chat-messages">
            {messages.map((m, i) => (
              <ChatBubble
                key={i}
                role={m.role}
                content={m.content}
                showSources={!!m.showSources && m.role === 'assistant'}
                showActions={m.role === 'assistant' && i > 0}
                mode={m.mode}
                docs={m.docs}
              />
            ))}
            {isTyping && (
              <div className="chat-bubble-wrap">
                <div className="avatar-ai"><Zap size={12} /></div>
                <div className="chat-bubble chat-bubble--ai chat-bubble--typing">
                  <span /><span /><span />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Voice Panel */}
        <div className="voice-panel">
          {/* Live Transcript Display */}
          {isListening && (
            <LiveTranscriptBox
              transcript={recorder.liveTranscript}
              isListening={isListening}
            />
          )}

          <div className="voice-wave-ring">
            <AnimatedWaveform active={isListening} bars={recorder.waveform} />
          </div>

          {/* Text Input Row */}
          <div className="text-input-row">
            <textarea
              ref={inputRef}
              className="text-input"
              placeholder="Type a question or click the mic to speak…"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isListening || isTyping}
            />
            <button
              className="send-btn"
              onClick={handleTextSend}
              disabled={!textInput.trim() || isListening || isTyping}
              title="Send"
            >
              {isTyping ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
            </button>
          </div>

          {/* Voice Controls */}
          <div className="voice-controls">
            <button
              className={`mic-button ${isListening ? 'mic-button--active' : ''}`}
              onClick={handleMicClick}
              aria-label={isListening ? 'Stop listening' : 'Start listening'}
              disabled={isTyping}
            >
              <div className={`mic-glow ${isListening ? 'mic-glow--active' : ''}`} />
              {isListening ? <MicOff size={26} /> : <Mic size={26} />}
            </button>

            {isListening && (
              <button
                className="stop-btn"
                onClick={() => { setIsListening(false); recorder.stopRecording(); }}
              >
                <Square size={14} /> Send
              </button>
            )}
          </div>

          <div className="voice-hint">
            {isListening
              ? <><span className="listening-dot" /> Listening… {recorder.useBrowserSTT ? 'Your words appear above' : 'Speak now'}</>
              : recorder.error
              ? <span className="voice-error">⚠ {recorder.error}</span>
              : 'Tap mic to speak — words appear live as you talk'}
          </div>
        </div>

        {/* Footer */}
        <footer className="app-footer">
          <div className="footer-warn">
            <AlertCircle size={12} />
            AI-generated responses. Please verify important information.
          </div>
          <div className="footer-copy">© 2025 VoiceRAG · Groq + ChromaDB</div>
        </footer>
      </div>

      <RightSidebar
        conversations={conversations}
        onSelectConversation={handleSelectConversation}
        onSelectSuggestion={handleQuestion}
      />
    </div>
  );
}
