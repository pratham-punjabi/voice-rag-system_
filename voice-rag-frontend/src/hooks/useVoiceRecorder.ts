import { useState, useRef, useCallback, useEffect } from 'react';
import type { RecordingState } from '../types/api.types';

export interface VoiceRecorderResult {
  state: RecordingState;
  waveform: number[];
  audioBlob: Blob | null;
  duration: number;
  liveTranscript: string;       // Live interim transcript shown on UI
  finalTranscript: string;      // Confirmed final transcript
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  reset: () => void;
  error: string | null;
  useBrowserSTT: boolean;       // true when Web Speech API is the primary STT
}

const TARGET_SAMPLE_RATE = 16000;

// Extend Window type for SpeechRecognition
declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

export function useVoiceRecorder(): VoiceRecorderResult {
  const [state, setState] = useState<RecordingState>('idle');
  const [waveform, setWaveform] = useState<number[]>([]);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState('');

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const durationRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const finalTranscriptRef = useRef('');

  // Detect if browser supports Web Speech API
  const hasBrowserSTT = typeof window !== 'undefined' &&
    ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);

  const drawWaveform = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteTimeDomainData(data);
    const samples = Array.from({ length: 40 }, (_, i) => {
      const idx = Math.floor(i * (data.length / 40));
      return (data[idx] - 128) / 128;
    });
    setWaveform(samples);
    animRef.current = requestAnimationFrame(drawWaveform);
  }, []);

  const startSpeechRecognition = useCallback((lang: string = 'en-US') => {
    if (!hasBrowserSTT) return;

    const SpeechRecognitionClass =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognitionClass();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = lang;
    recognition.maxAlternatives = 1;

    finalTranscriptRef.current = '';

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = '';
      let final = finalTranscriptRef.current;

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript + ' ';
          finalTranscriptRef.current = final;
        } else {
          interim += result[0].transcript;
        }
      }

      setFinalTranscript(final.trim());
      setLiveTranscript(interim ? `${final}${interim}` : final.trim());
    };

    recognition.onerror = (event) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.warn('Speech recognition error:', event.error);
      }
    };

    recognition.onend = () => {
      // Auto-restart if still recording (handles speech pauses)
      if (mediaRef.current?.state === 'recording') {
        try { recognition.start(); } catch (_) {}
      }
    };

    try {
      recognition.start();
      recognitionRef.current = recognition;
    } catch (err) {
      console.warn('Failed to start speech recognition:', err);
    }
  }, [hasBrowserSTT]);

  const startRecording = useCallback(async (lang: string = 'en-US') => {
    setError(null);
    setLiveTranscript('');
    setFinalTranscript('');
    finalTranscriptRef.current = '';

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: TARGET_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // Audio analyser for waveform visualization
      const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // MediaRecorder for audio blob (used for Sarvam STT)
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });
      mediaRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        setAudioBlob(blob);
        cancelAnimationFrame(animRef.current);
        setWaveform([]);
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
      };

      recorder.start(100);
      startTimeRef.current = Date.now();
      setState('recording');
      drawWaveform();

      durationRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 200);

      // Start Web Speech API for live transcript
      startSpeechRecognition(lang);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Microphone access denied';
      setError(msg);
      setState('error');
    }
  }, [drawWaveform, startSpeechRecognition]);

  const stopRecording = useCallback(() => {
    // Stop speech recognition
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (_) {}
      recognitionRef.current = null;
    }

    if (mediaRef.current?.state === 'recording') {
      mediaRef.current.stop();
    }
    if (durationRef.current) clearInterval(durationRef.current);
    setState('processing');
  }, []);

  const reset = useCallback(() => {
    setState('idle');
    setWaveform([]);
    setAudioBlob(null);
    setDuration(0);
    setError(null);
    setLiveTranscript('');
    setFinalTranscript('');
    finalTranscriptRef.current = '';
    cancelAnimationFrame(animRef.current);
  }, []);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animRef.current);
      if (durationRef.current) clearInterval(durationRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (_) {}
      }
    };
  }, []);

  return {
    state,
    waveform,
    audioBlob,
    duration,
    liveTranscript,
    finalTranscript,
    startRecording,
    stopRecording,
    reset,
    error,
    useBrowserSTT: hasBrowserSTT,
  };
}
