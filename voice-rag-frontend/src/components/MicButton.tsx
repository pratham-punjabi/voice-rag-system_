import { Mic, MicOff, Loader2 } from 'lucide-react';
import type { RecordingState } from '../types/api.types';

interface Props {
  state: RecordingState;
  onClick: () => void;
  disabled?: boolean;
}

export function MicButton({ state, onClick, disabled }: Props) {
  const isRecording = state === 'recording';
  const isProcessing = state === 'processing';

  return (
    <button
      className={`mic-btn ${isRecording ? 'mic-btn--active' : ''} ${isProcessing ? 'mic-btn--processing' : ''}`}
      onClick={onClick}
      disabled={disabled || isProcessing}
      aria-label={isRecording ? 'Stop recording' : 'Start recording'}
    >
      {isProcessing
        ? <Loader2 size={32} className="spin" />
        : isRecording
        ? <MicOff size={32} />
        : <Mic size={32} />}
    </button>
  );
}
