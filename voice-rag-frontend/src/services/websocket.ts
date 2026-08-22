/**
 * Low-level WebSocket client for streaming audio to /api/voice.
 * Used by useWebSocket hook — import the hook for React components.
 */

export type WSEventType = 'open' | 'message' | 'close' | 'error';

export interface WSClientOptions {
  url: string;
  onMessage: (msg: object) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

export class VoiceWSClient {
  private ws: WebSocket | null = null;
  private opts: WSClientOptions;

  constructor(opts: WSClientOptions) {
    this.opts = opts;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.opts.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.opts.onOpen?.();
      this.sendJSON({ action: 'config', sample_rate: 16000 });
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data as string);
        this.opts.onMessage(data);
      } catch {
        // Non-JSON frame — ignore
      }
    };

    this.ws.onclose = () => this.opts.onClose?.();
    this.ws.onerror = (e) => this.opts.onError?.(e);
  }

  sendBinary(data: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  sendJSON(payload: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  stop(): void {
    this.sendJSON({ action: 'stop' });
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export function buildWSUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/api/voice`;
}
