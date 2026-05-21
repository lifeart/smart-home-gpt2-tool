// In-browser speech recognition — Whisper via transformers.js. No API.
//
// Mirrors the author's voice pipeline (README: "RU speech → Whisper →
// English → tool call") but runs entirely in the browser: the Whisper
// model is downloaded from HF and executed locally via ONNX Runtime Web.
// `task: 'translate'` means any spoken language is transcribed to English,
// which is what the GPT-2 tool-caller expects.

import { pipeline } from '@huggingface/transformers';

const WHISPER_MODEL = 'Xenova/whisper-base';

let asrPipe = null;
let asrLoading = null;

// Lazy-load the ASR pipeline. Whisper isn't bundled under /models/, so
// transformers.js probes the local path, gets a 404 (vite.config.js makes
// missing /models/ paths return a real 404) and falls back to the HF Hub.
export async function getASR(progressCb) {
  if (asrPipe) return asrPipe;
  if (!asrLoading) {
    asrLoading = pipeline('automatic-speech-recognition', WHISPER_MODEL, {
      dtype: 'q8',
      device: 'wasm',
      progress_callback: progressCb,
    }).then(
      (p) => { asrPipe = p; return p; },
      (e) => { asrLoading = null; throw e; },
    );
  }
  return asrLoading;
}

// ---------------- microphone recorder ----------------

let mediaRecorder = null;
let chunks = [];
let activeStream = null;

export function isRecording() {
  return !!mediaRecorder && mediaRecorder.state === 'recording';
}

export async function startRecording() {
  activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(activeStream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size) chunks.push(e.data);
  };
  mediaRecorder.start();
}

export function stopRecording() {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject(new Error('not recording'));
      return;
    }
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, {
        type: mediaRecorder.mimeType || 'audio/webm',
      });
      if (activeStream) activeStream.getTracks().forEach((t) => t.stop());
      mediaRecorder = null;
      activeStream = null;
      resolve(blob);
    };
    mediaRecorder.stop();
  });
}

// ---------------- audio decode → 16 kHz mono Float32 ----------------

async function blobTo16kMono(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(arrayBuf);
  } finally {
    ctx.close();
  }
  // Mix down to mono.
  const n = decoded.length;
  const mono = new Float32Array(n);
  for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
    const d = decoded.getChannelData(ch);
    for (let i = 0; i < n; i++) mono[i] += d[i];
  }
  if (decoded.numberOfChannels > 1) {
    for (let i = 0; i < n; i++) mono[i] /= decoded.numberOfChannels;
  }
  // Resample to 16 kHz (linear interpolation).
  const srcRate = decoded.sampleRate;
  if (srcRate === 16000) return mono;
  const ratio = srcRate / 16000;
  const outLen = Math.floor(n / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, n - 1);
    const frac = pos - i0;
    out[i] = mono[i0] * (1 - frac) + mono[i1] * frac;
  }
  return out;
}

// ---------------- transcribe ----------------

/**
 * Transcribe a recorded audio blob to English text.
 * @param {Blob} blob
 * @param {{progressCb?:Function, translate?:boolean}} [opts]
 * @returns {Promise<string>}
 */
export async function transcribe(blob, { progressCb, translate = true } = {}) {
  const asr = await getASR(progressCb);
  const audio = await blobTo16kMono(blob);
  if (!audio.length) throw new Error('empty recording');
  const out = await asr(audio, {
    task: translate ? 'translate' : 'transcribe',
    chunk_length_s: 30,
  });
  let text = (typeof out === 'string' ? out : (out && out.text) || '').trim();
  // Whisper emits bracketed markers for silence / non-speech — treat as empty
  // so the caller shows "no speech detected" instead of injecting a marker.
  if (/^[[(]?\s*(blank_audio|silence|inaudible|no audio|music)\s*[\])]?\.?$/i.test(text)) {
    text = '';
  }
  return text;
}
