/**
 * Audio capture and processing utilities.
 * Handles MediaRecorder, AudioContext, PCM resampling, and WAV encoding.
 */

export const TARGET_SAMPLE_RATE = 16000;
export const TARGET_CHANNELS = 1;

/**
 * Request microphone access with optimal settings for STT.
 */
export async function getMicrophoneStream(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: TARGET_SAMPLE_RATE,
      channelCount: TARGET_CHANNELS,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
}

/**
 * Encode a Float32 PCM buffer as a WAV Blob at 16kHz mono 16-bit.
 */
export function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);         // PCM chunk size
  view.setUint16(20, 1, true);          // PCM format
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  writeString(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  // Convert float32 → int16
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Downsample a Float32Array from sourceSampleRate to targetSampleRate.
 * Simple linear interpolation — adequate for voice.
 */
export function resample(
  input: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number
): Float32Array {
  if (sourceSampleRate === targetSampleRate) return input;

  const ratio = sourceSampleRate / targetSampleRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? 0;
    const b = input[idx + 1] ?? 0;
    output[i] = a + frac * (b - a);
  }

  return output;
}

/**
 * Compute RMS amplitude for waveform visualisation.
 */
export function computeRMS(buffer: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) {
    const val = (buffer[i] - 128) / 128;
    sum += val * val;
  }
  return Math.sqrt(sum / buffer.length);
}

/**
 * Convert recorded audio Blob to WAV Blob at 16kHz.
 * Handles browsers that record in webm/ogg by decoding via AudioContext.
 */
export async function blobToWAV(blob: Blob): Promise<Blob> {
  // If already WAV, return as-is
  if (blob.type === 'audio/wav') return blob;

  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });

  try {
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const channelData = decoded.getChannelData(0); // mono
    const resampled = resample(channelData, decoded.sampleRate, TARGET_SAMPLE_RATE);
    return encodeWAV(resampled, TARGET_SAMPLE_RATE);
  } finally {
    await audioCtx.close();
  }
}
