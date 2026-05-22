import {
  AutoTokenizer,
  AutoModelForCausalLM,
  LogitsProcessorList,
  env,
} from '@huggingface/transformers';
import {
  extractCandidateNames,
  extractPromptSchemas,
  buildSchemaConstraint,
  JsonSchemaLogitsProcessor,
} from './grammar.js';
import {
  extractUserQuery,
  topKForQuery,
  rewriteCandidateList,
  pruneSchemas,
  getOrBuildIndex,
} from './retrieval.js';
import { PRESETS, PRESET_LIST } from './presets.js';
import { MODEL_CARDS, TOGGLE_HELP, BENCH_LEGEND, FOOTER, DTYPE_NOTES } from './help.js';
import { canonicalizeCall } from './canon.js';
import { startRecording, stopRecording, isRecording, transcribe } from './voice.js';
import { asset, LOCAL_MODEL_PATH } from './paths.js';
import './bench.js';
import './voice_bench.js';

// Model resolution: transformers.js probes the local path first
// (web/public/models/<id>) and, if it isn't there, downloads from the HF Hub.
// Both flags on = "use local if present, else fetch from Hugging Face".
// `LOCAL_MODEL_PATH` carries the deployment base prefix so the probe URL is
// correct under both '/' and '/smart-home-gpt2-tool/'. On the deployed build
// `public/models/` is absent, so the probe 404s and the Hub fallback kicks in.
env.allowLocalModels = true;
env.allowRemoteModels = true;
env.localModelPath = LOCAL_MODEL_PATH;

// Lazy-loaded tool registry (for constrained decoding).
let toolRegistry = null;
async function getRegistry() {
  if (!toolRegistry) {
    toolRegistry = await fetch(asset('eval/tool_registry.json')).then(r => r.json());
  }
  return toolRegistry;
}

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const outEl = $('out');
const benchEl = $('bench');
const runBtn = $('run');
const loadBtn = $('load');

let tokenizer = null;
let model = null;
let loadedKey = '';

// Update the status pill text + colour state. `kind` ∈ idle|busy|ok|err.
function setStatus(text, kind = 'idle') {
  statusEl.textContent = text;
  statusEl.classList.remove('is-busy', 'is-ok', 'is-err');
  if (kind === 'busy') statusEl.classList.add('is-busy');
  else if (kind === 'ok') statusEl.classList.add('is-ok');
  else if (kind === 'err') statusEl.classList.add('is-err');
  console.log('[status]', text);
}

function detectWebGPU() {
  return 'gpu' in navigator;
}

async function load() {
  const model_id = $('model').value;
  const device = $('device').value;
  const dtype = $('dtype').value;
  const key = `${model_id}|${device}|${dtype}`;
  if (key === loadedKey) {
    setStatus('already loaded · ready', 'ok');
    return;
  }

  if (device === 'webgpu' && !detectWebGPU()) {
    setStatus('WebGPU not available in this browser', 'err');
    return;
  }

  loadBtn.disabled = true;
  runBtn.disabled = true;
  setStatus(`loading ${device}/${dtype}…`, 'busy');
  const t0 = performance.now();
  try {
    tokenizer = await AutoTokenizer.from_pretrained(model_id, {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          setStatus(`tokenizer ${p.file} ${(p.progress ?? 0).toFixed(0)}%`, 'busy');
        }
      },
    });
    model = await AutoModelForCausalLM.from_pretrained(model_id, {
      device,
      dtype,
      progress_callback: (p) => {
        if (p.status === 'progress') {
          setStatus(
            `model ${p.file} ${(p.progress ?? 0).toFixed(0)}% (${(
              (p.loaded ?? 0) /
              1024 /
              1024
            ).toFixed(1)} MB)`,
            'busy',
          );
        }
      },
    });
    loadedKey = key;
    // Expose for bench harness
    window._bench_model = model;
    window._bench_tokenizer = tokenizer;
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    setStatus(`loaded in ${dt}s · ${device}/${dtype} · ready`, 'ok');
    runBtn.disabled = false;
    setGenerateReady(true);
  } catch (e) {
    console.error(e);
    setStatus(`load failed: ${e.message}`, 'err');
  } finally {
    loadBtn.disabled = false;
  }
}

// Toggle the "Generate" button's ready state + the hint next to it.
function setGenerateReady(ready) {
  const hint = document.querySelector('.generate-hint');
  if (hint) {
    hint.textContent = ready
      ? 'Model ready — pick a command and Generate'
      : 'Load a model first ↑';
    hint.style.color = ready ? 'var(--ok)' : '';
  }
}

async function generate() {
  if (!model || !tokenizer) return;
  let prompt = $('prompt').value;
  const max_new_tokens = parseInt($('max').value, 10);
  const temperature = parseFloat($('temp').value);
  const do_sample = temperature > 0;
  const constrained = $('constrained').checked;
  const retrievalOn = $('retrieval').checked;
  const topK = Math.max(1, Math.min(10, parseInt($('topk').value, 10) || 3));

  runBtn.disabled = true;
  runBtn.classList.add('is-generating');
  outEl.textContent = '';
  benchEl.textContent = '';
  const resultEl = $('result');
  if (resultEl) resultEl.hidden = false;
  const synthEl = $('synth-out');
  if (synthEl) {
    synthEl.innerHTML =
      '<div class="synth-card"><div class="synth-body synth-json">generating…</div></div>';
  }
  setStatus('generating…', 'busy');

  // Retrieval pre-rank (opt-in toggle): keep only the top-K most relevant
  // candidate schemas, dropping the rest. Iter 40 made this SCHEMA-PRESERVING
  // — `pruneSchemas` keeps the full typed schema of each survivor (better
  // arguments) instead of collapsing to a names-only list. It is a genuine
  // speed/accuracy trade and stays opt-in + default-OFF: the gold function
  // lands in the MiniLM top-8 for ~95% of queries (top-12: ~97%), so
  // pruning can drop the answer — see training/verify_retrieval_recall.py.
  let retrievalInfo = 'retrieval: OFF';
  if (retrievalOn) {
    try {
      const userQuery = extractUserQuery(prompt);
      if (userQuery) {
        const before = prompt.length;
        const { topNames, topScores } = await topKForQuery(userQuery, topK);
        const origCands = extractCandidateNames(prompt);
        const goldInTopK = origCands.length > 0
          ? origCands.some(n => topNames.includes(n))
          : null;
        prompt = pruneSchemas(prompt, topNames);
        retrievalInfo = `retrieval: pruned to top-${topK} [${topNames.join(', ')}]  ${before}→${prompt.length} chars  score(top1)=${topScores[0]?.toFixed(3) ?? 'n/a'}`;
        if (goldInTopK !== null) {
          retrievalInfo += `  orig-cands-in-topK: ${origCands.filter(n => topNames.includes(n)).length}/${origCands.length}`;
        }
      } else {
        retrievalInfo = 'retrieval: ON but no USER query extracted — running free';
      }
    } catch (e) {
      console.error('retrieval failed:', e);
      retrievalInfo = `retrieval: FAILED (${e.message}) — falling back to original prompt`;
    }
  }

  const inputs = await tokenizer(prompt, { return_tensors: 'pt' });
  const promptTokens = inputs.input_ids.dims[1];

  // Build optional constrained-decoding processor
  let logits_processor = null;
  let constraintInfo = '';
  let activeProc = null;
  if (constrained) {
    const cands = extractCandidateNames(prompt);
    if (cands.length > 0) {
      const registry = await getRegistry();
      const promptSchemas = extractPromptSchemas(prompt);
      const typedArgs = $('typedargs') ? $('typedargs').checked : true;
      const wideNames = $('widenames') ? $('widenames').checked : false;
      const constraint = buildSchemaConstraint(cands, registry, { promptSchemas, typedArgs, wideNames });
      const eosTokenId =
        tokenizer.eos_token_id ??
        (model.generation_config && model.generation_config.eos_token_id);
      const always = new Set();
      if (eosTokenId !== undefined && eosTokenId !== null) always.add(eosTokenId);
      activeProc = new JsonSchemaLogitsProcessor({
        tokenizer,
        promptLength: promptTokens,
        constraint,
        topK: 40,
        allowAlwaysTokens: always,
      });
      logits_processor = new LogitsProcessorList();
      logits_processor.push(activeProc);
      constraintInfo = `constrained: ${cands.length} candidates (${cands.slice(0, 4).join(', ')}${cands.length > 4 ? '…' : ''})`;
    } else {
      constraintInfo = 'constrained: ON but no candidates parsed from prompt — running free';
    }
  } else {
    constraintInfo = 'constrained: OFF';
  }

  const t0 = performance.now();
  let firstTokenAt = null;
  let genTokens = 0;

  const streamer = {
    callback_function: (text) => {
      if (firstTokenAt === null) firstTokenAt = performance.now();
      genTokens += 1;
      outEl.textContent += text;
    },
  };

  // transformers.js TextStreamer
  const { TextStreamer } = await import('@huggingface/transformers');
  const ts = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: streamer.callback_function,
  });

  const output = await model.generate({
    ...inputs,
    max_new_tokens,
    do_sample,
    temperature: do_sample ? temperature : 1.0,
    streamer: ts,
    ...(logits_processor ? { logits_processor } : {}),
  });

  const t1 = performance.now();
  const total = (t1 - t0) / 1000;
  const ttft = firstTokenAt ? (firstTokenAt - t0) / 1000 : null;
  const newTokens = output.dims[1] - promptTokens;
  const tps = newTokens / Math.max(total - (ttft ?? 0), 0.001);

  const lines = [
    `prompt tokens : ${promptTokens}`,
    `new tokens    : ${newTokens}`,
    `time-to-first : ${ttft !== null ? ttft.toFixed(2) + ' s' : 'n/a'}`,
    `total         : ${total.toFixed(2)} s`,
    `throughput    : ${tps.toFixed(1)} tok/s`,
    `backend       : ${$('device').value} · ${$('dtype').value}`,
    retrievalInfo,
    constraintInfo,
  ];
  if (activeProc) {
    const s = activeProc.stats;
    lines.push(
      `grammar steps : ${s.steps}, overhead ${(s.totalMs / Math.max(s.steps, 1)).toFixed(2)} ms/step (total ${s.totalMs.toFixed(0)} ms)`,
    );
  }
  benchEl.textContent = lines.join('\n');

  // Parse the generated tool call and show a canonicalized version. This is
  // a pure browser-side post-process (training/canon.py port) — normalizes
  // value formats (12h→24h time, day plural, float rounding). No API.
  await renderParsedCall(outEl.textContent);

  setStatus(`done · ${tps.toFixed(0)} tok/s`, 'ok');
  runBtn.disabled = false;
  runBtn.classList.remove('is-generating');
}

// Extract the first balanced {...} JSON object, parse to {name, arguments}.
function parseToolCall(text) {
  if (!text) return null;
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (esc) { esc = false; continue; }
    if (c === '\\' && inStr) { esc = true; continue; }
    if (c === '"') { inStr = !inStr; continue; }
    if (inStr) continue;
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) {
        try {
          const obj = JSON.parse(text.slice(start, i + 1));
          if (obj && typeof obj === 'object') {
            return {
              name: typeof obj.name === 'string' ? obj.name : null,
              arguments: (obj.arguments && typeof obj.arguments === 'object') ? obj.arguments : {},
            };
          }
        } catch { /* not valid JSON yet */ }
        return null;
      }
    }
  }
  return null;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// Pretty-print a JSON value to a syntax-highlighted HTML string. Tokens get
// .tok-* classes coloured by style.css. Pure formatting — operates on the
// already-parsed object so it can never inject markup from model output.
function highlightJson(value, indent = 0) {
  const pad = '  '.repeat(indent);
  const padIn = '  '.repeat(indent + 1);
  const punc = (c) => `<span class="tok-punc">${c}</span>`;
  if (value === null) return '<span class="tok-null">null</span>';
  if (typeof value === 'boolean')
    return `<span class="tok-bool">${value}</span>`;
  if (typeof value === 'number')
    return `<span class="tok-num">${escapeHtml(String(value))}</span>`;
  if (typeof value === 'string')
    return `<span class="tok-str">"${escapeHtml(value)}"</span>`;
  if (Array.isArray(value)) {
    if (value.length === 0) return punc('[]');
    const items = value
      .map((v) => padIn + highlightJson(v, indent + 1))
      .join(punc(',') + '\n');
    return punc('[') + '\n' + items + '\n' + pad + punc(']');
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value);
    if (keys.length === 0) return punc('{}');
    const rows = keys
      .map(
        (k) =>
          padIn +
          `<span class="tok-key">"${escapeHtml(k)}"</span>` +
          punc(': ') +
          highlightJson(value[k], indent + 1),
      )
      .join(punc(',') + '\n');
    return punc('{') + '\n' + rows + '\n' + pad + punc('}');
  }
  return escapeHtml(String(value));
}

// Render the parsed + canonicalized tool call under the raw output.
// Async because enum-snapping (Iter 38) needs the tool registry, which is
// lazy-fetched. If the registry can't load, snapping is skipped — value
// canonicalization still runs.
async function renderParsedCall(rawText) {
  const el = $('synth-out');
  if (!el) return;
  const call = parseToolCall(rawText);
  if (!call || !call.name) {
    el.innerHTML = `
      <div class="synth-card synth-err">
        <div class="synth-card-head">
          <span class="synth-title">No tool call</span>
        </div>
        <div class="synth-errmsg">Could not parse a valid tool call from the output. Try the “Raw model output” panel below, or pick a different command.</div>
      </div>`;
    return;
  }
  let registry = null;
  try {
    registry = await getRegistry();
  } catch (e) {
    console.warn('[canon] registry unavailable, skipping enum-snap:', e);
  }
  const canon = {
    name: call.name,
    arguments: canonicalizeCall(call.name, call.arguments, registry),
  };
  const same = JSON.stringify(call.arguments) === JSON.stringify(canon.arguments);
  const callObj = { name: canon.name, arguments: canon.arguments };
  const pretty = JSON.stringify(callObj, null, 2);

  el.innerHTML = `
    <div class="synth-card">
      <div class="synth-card-head">
        <span class="synth-title">Parsed tool call</span>
        <div style="display:flex;align-items:center;gap:0.5rem">
          <span class="synth-fn-pill">${escapeHtml(canon.name)}()</span>
          <button class="copy-btn" type="button" id="copy-call">⧉ Copy JSON</button>
        </div>
      </div>
      <div class="synth-body">
        <pre class="synth-json">${highlightJson(callObj)}</pre>
        ${
          same
            ? ''
            : '<p class="synth-note">argument values normalized — enum-snapped, time → 24h, day → singular, numbers rounded</p>'
        }
      </div>
    </div>`;

  const copyBtn = $('copy-call');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pretty);
        copyBtn.textContent = '✓ Copied';
        copyBtn.classList.add('copied');
        setTimeout(() => {
          copyBtn.textContent = '⧉ Copy JSON';
          copyBtn.classList.remove('copied');
        }, 1600);
      } catch (e) {
        console.error('clipboard write failed:', e);
        copyBtn.textContent = 'copy failed';
      }
    });
  }
}

loadBtn.addEventListener('click', load);
runBtn.addEventListener('click', generate);

// Expose retrieval helpers for the bench harness.
window._retrieval = {
  extractUserQuery,
  topKForQuery,
  rewriteCandidateList,
  getOrBuildIndex,
};

// Preset selector — populate from PRESET_LIST (grouped by category),
// then on change replace the prompt with the selected realistic command.
const presetEl = $('preset');
if (presetEl) {
  const byCat = {};
  for (const p of PRESET_LIST) {
    (byCat[p.category] = byCat[p.category] || []).push(p);
  }
  for (const [cat, items] of Object.entries(byCat)) {
    const group = document.createElement('optgroup');
    group.label = cat;
    for (const p of items) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `"${p.query}"`;
      group.appendChild(opt);
    }
    presetEl.appendChild(group);
  }
  presetEl.addEventListener('change', () => {
    const p = PRESETS[presetEl.value];
    if (p) $('prompt').value = p;
  });
  // Open the demo on a real rich-schema example so the format is visible.
  if (PRESET_LIST.length) {
    const first = PRESET_LIST[0];
    $('prompt').value = PRESETS[first.id];
    presetEl.value = first.id;
  }
}

// Replace the USER command inside the current prompt, keeping the SYSTEM
// function list and the ASSISTANT marker intact.
function setUserQuery(query) {
  const el = $('prompt');
  if (!el) return;
  let p = el.value;
  const uIdx = p.indexOf('USER:');
  const aIdx = p.indexOf('ASSISTANT:', uIdx >= 0 ? uIdx : 0);
  if (uIdx >= 0 && aIdx > uIdx) {
    el.value = p.slice(0, uIdx) + 'USER: ' + query + '\n\n\n' + p.slice(aIdx);
  } else {
    el.value = p + '\n\n\nUSER: ' + query + '\n\n\nASSISTANT: <functioncall> ';
  }
}

// 🎤 voice input — record, transcribe in-browser with Whisper, inject text.
const micBtn = $('mic');
const voiceStatus = $('voice-status');
if (micBtn) {
  micBtn.addEventListener('click', async () => {
    if (!isRecording()) {
      try {
        await startRecording();
        micBtn.textContent = '⏹ Stop & transcribe';
        micBtn.classList.add('recording');
        if (voiceStatus) voiceStatus.textContent = 'recording… say a command, then click stop';
      } catch (e) {
        if (voiceStatus) voiceStatus.textContent = `mic unavailable: ${e.message}`;
      }
      return;
    }
    // Currently recording → stop and transcribe.
    micBtn.disabled = true;
    micBtn.classList.remove('recording');
    micBtn.textContent = '🎤 Speak a command';
    try {
      if (voiceStatus) voiceStatus.textContent = 'transcribing…';
      const blob = await stopRecording();
      const text = await transcribe(blob, {
        progressCb: (pr) => {
          if (pr.status === 'progress' && voiceStatus) {
            voiceStatus.textContent =
              `loading Whisper · ${pr.file} ${(pr.progress ?? 0).toFixed(0)}%`;
          }
        },
      });
      if (text) {
        setUserQuery(text);
        if (model && tokenizer) {
          // Heard a command and a model is ready → run it automatically.
          if (voiceStatus) voiceStatus.textContent = `heard: "${text}" — generating…`;
          micBtn.disabled = false;
          await generate();
          if (voiceStatus) voiceStatus.textContent = `heard: "${text}"`;
        } else if (voiceStatus) {
          voiceStatus.textContent = `heard: "${text}" — click “Load model”, then Generate`;
        }
      } else if (voiceStatus) {
        voiceStatus.textContent = 'no speech detected — try again';
      }
    } catch (e) {
      console.error('voice transcription failed:', e);
      if (voiceStatus) voiceStatus.textContent = `transcription failed: ${e.message}`;
    } finally {
      micBtn.disabled = false;
    }
  });
}

// Model dropdown — render a 2-3 line card under the config row on change.
const modelEl = $('model');
const modelInfoEl = $('model-info');
function renderModelInfo() {
  if (!modelInfoEl || !modelEl) return;
  const card = MODEL_CARDS[modelEl.value];
  if (!card) {
    modelInfoEl.innerHTML = '';
    return;
  }
  const body = card.body.map((line) => `<li>${line}</li>`).join('');
  modelInfoEl.innerHTML = `<strong>${card.title}</strong><ul>${body}</ul>`;
}
if (modelEl) {
  modelEl.addEventListener('change', renderModelInfo);
  renderModelInfo();
}

// Dtype dropdown — show a live one-line explanation of the selected weight
// precision (download size + quality + backend support).
const dtypeEl = $('dtype');
const dtypeInfoEl = $('dtype-info');
function renderDtypeInfo() {
  if (!dtypeInfoEl || !dtypeEl) return;
  const note = DTYPE_NOTES[dtypeEl.value];
  dtypeInfoEl.textContent = note || '';
}
if (dtypeEl) {
  dtypeEl.addEventListener('change', renderDtypeInfo);
  renderDtypeInfo();
}

// Per-toggle info: attach an (i) icon to each toggle that expands a help panel.
// Each toggle is a <label class="toggle"> in the Advanced section; the help
// panel is inserted right after its label inside the .toggles grid.
function attachToggleHelp() {
  for (const [id, help] of Object.entries(TOGGLE_HELP)) {
    const input = $(id);
    if (!input) continue;
    const parentLabel = input.closest('label.toggle') || input.closest('label');
    if (!parentLabel) continue;
    if (parentLabel.querySelector('.help-toggle')) continue; // idempotent
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'help-toggle';
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-label', `Help for: ${help.label}`);
    btn.title = `Click for help: ${help.label}`;
    btn.textContent = 'ⓘ';
    parentLabel.appendChild(btn);
    const panel = document.createElement('div');
    panel.className = 'toggle-help';
    panel.hidden = true;
    panel.innerHTML = `
      <div><span class="th-tag">What</span> ${help.what}</div>
      <div><span class="th-tag">When</span> ${help.when}</div>
      <div><span class="th-tag">Effect</span> ${help.effect}</div>
    `;
    parentLabel.after(panel);
    const togglePanel = (e) => {
      // Stop the click bubbling to the <label> (which would flip the checkbox).
      if (e) { e.preventDefault(); e.stopPropagation(); }
      panel.hidden = !panel.hidden;
      btn.setAttribute('aria-expanded', String(!panel.hidden));
    };
    btn.addEventListener('click', togglePanel);
  }
}
attachToggleHelp();

// Bench legend (under the bench <pre>). Collapsible — default expanded.
function renderBenchLegend() {
  const benchSection = benchEl ? benchEl.parentElement : null;
  if (!benchSection) return;
  if (document.getElementById('bench-legend')) return;
  const legend = document.createElement('details');
  legend.id = 'bench-legend';
  legend.className = 'info-card legend-card';
  legend.open = true;
  const rows = BENCH_LEGEND.map(([k, v, good]) => {
    const goodSpan = good ? ` <span class="legend-good">(${good})</span>` : '';
    return `<div><code>${k}</code> — ${v}${goodSpan}</div>`;
  }).join('');
  legend.innerHTML = `<summary><strong>What each bench line means</strong> <span class="legend-hint">(click to collapse)</span></summary>${rows}`;
  benchSection.appendChild(legend);
}
renderBenchLegend();

// Footer with project + HF Hub links.
function renderFooter() {
  if (document.getElementById('site-footer')) return;
  const footer = document.createElement('footer');
  footer.id = 'site-footer';
  const links = FOOTER.links
    .map(
      (l) =>
        `<a href="${l.href}" target="_blank" rel="noopener" data-kind="${l.kind}"${l.probe ? ' data-probe="1"' : ''}>${l.label}</a>`,
    )
    .join(' · ');
  footer.innerHTML = `
    <p>${FOOTER.blurb}</p>
    <p class="links">${links}</p>
    <p class="meta">${FOOTER.meta}</p>
  `;
  (document.querySelector('.page') || document.body).appendChild(footer);

  // Optimistically probe any link flagged `probe: true` (e.g. v4 which may
  // not be uploaded to HF yet). If the HEAD request fails / 404s, append a
  // "· not yet published" hint next to the link. Pure UX — never blocks.
  for (const a of footer.querySelectorAll('a[data-probe="1"]')) {
    fetch(a.href, { method: 'HEAD', mode: 'no-cors' })
      .then((resp) => {
        // `no-cors` returns opaque responses — we can't read status. Best
        // we can do is treat a non-error as "probably exists". Network
        // errors throw, caught below.
        if (resp.type === 'opaque' || resp.ok) {
          // assume exists
        } else if (resp.status === 404) {
          a.after(document.createTextNode(' (not yet on HF)'));
          a.style.opacity = '0.6';
        }
      })
      .catch(() => {
        const note = document.createElement('span');
        note.textContent = ' (offline or not yet on HF)';
        note.style.opacity = '0.6';
        note.style.fontSize = '0.9em';
        a.after(note);
        a.style.opacity = '0.6';
      });
  }
}
renderFooter();

// Backend-aware dtype default: fp16 on WebGPU — half the download / GPU
// memory at the same accuracy as fp32 (fp16 weights are lossless; the
// WebGPU fp16 compute path is correct on onnxruntime-web >=1.26, which
// transformers.js 4.2 bundles — older builds produced garbage, see
// Iter 37). onnxruntime-web's WASM EP has no fp16 kernels, so when WebGPU
// is unavailable fall back to device=wasm + dtype=fp32.
if (!detectWebGPU()) {
  if ($('device')) $('device').value = 'wasm';
  if (dtypeEl) {
    dtypeEl.value = 'fp32';
    renderDtypeInfo();
  }
}
setStatus(
  detectWebGPU()
    ? 'WebGPU ready · click Load model'
    : 'WASM fallback · click Load model',
  detectWebGPU() ? 'idle' : 'err',
);

// --- Theme toggle ----------------------------------------------------------
// Persist an explicit light/dark choice in localStorage. Absent a choice the
// page follows the OS via the prefers-color-scheme block in style.css.
(function initTheme() {
  const root = document.documentElement;
  const btn = $('theme-toggle');
  const KEY = 'sh-gpt2-theme';
  const saved = (() => {
    try { return localStorage.getItem(KEY); } catch { return null; }
  })();
  if (saved === 'light' || saved === 'dark') {
    root.setAttribute('data-theme', saved);
  }
  if (!btn) return;
  btn.addEventListener('click', () => {
    const prefersDark =
      window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches;
    const current =
      root.getAttribute('data-theme') || (prefersDark ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch { /* storage blocked — non-fatal */ }
  });
})();
