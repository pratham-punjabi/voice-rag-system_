import { useState, useRef, useCallback, useEffect } from 'react';

export type WSStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface WSMessage {
  type: 'status' | 'transcript' | 'result' | 'error';
  message?: string;
  transcript?: string;
  data?: unknown;
  code?: string;
}

export interface UseWebSocketResult {
  status: WSStatus;
  lastMessage: WSMessage | null;
  connect: () => void;
  disconnect: () => void;
  sendBinary: (data: ArrayBuffer | Blob) => void;
  sendJSON: (payload: object) => void;
  error: string | null;
}

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/voice`;
const RECONNECT_DELAY_MS = 2000;

export function useWebSocket(): UseWebSocketResult {
  const [status, setStatus] = useState<WSStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const shouldReconnectRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    shouldReconnectRef.current = true;
    setStatus('connecting');
    setError(null);

    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      // Announce sample rate config
      ws.send(JSON.stringify({ action: 'config', sample_rate: 16000 }));
    };

    ws.onmessage = (evt) => {
      try {
        const msg: WSMessage = JSON.parse(evt.data as string);
        setLastMessage(msg);
      } catch {
        // Ignore non-JSON messages
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus('disconnected');
      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearTimeout(reconnectTimerRef.current);
    wsRef.current?.close();
    setStatus('disconnected');
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer | Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const sendJSON = useCallback((payload: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, []);

  return { status, lastMessage, connect, disconnect, sendBinary, sendJSON, error };
}
