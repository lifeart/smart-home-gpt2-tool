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
  getOrBuildIndex,
} from './retrieval.js';
import { PRESETS } from './presets.js';
import { MODEL_CARDS, TOGGLE_HELP, BENCH_LEGEND, FOOTER } from './help.js';
import './bench.js';
import './voice_bench.js';

env.allowLocalModels = true;
env.localModelPath = '/models/';

// Lazy-loaded tool registry (for constrained decoding).
let toolRegistry = null;
async function getRegistry() {
  if (!toolRegistry) {
    toolRegistry = await fetch('/eval/tool_registry.json').then(r => r.json());
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

function setStatus(text) {
  statusEl.textContent = text;
  console.log('[status]', text);
}

function detectWebGPU() {
  return 'gpu' in navigator;
}

async function load() {
  let model_id = $('model').value;
  const device = $('device').value;
  const dtype = $('dtype').value;
  const isLocal = model_id.startsWith('local:');
  if (isLocal) model_id = model_id.slice('local:'.length);
  env.allowRemoteModels = !isLocal;
  const key = `${(isLocal ? 'local:' : '') + model_id}|${device}|${dtype}`;
  if (key === loadedKey) {
    setStatus('already loaded');
    return;
  }

  if (device === 'webgpu' && !detectWebGPU()) {
    setStatus('WebGPU not available in this browser');
    return;
  }

  loadBtn.disabled = true;
  runBtn.disabled = true;
  setStatus(`loading ${model_id} (${device}/${dtype})…`);
  const t0 = performance.now();
  try {
    tokenizer = await AutoTokenizer.from_pretrained(model_id, {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          setStatus(`tokenizer ${p.file} ${(p.progress ?? 0).toFixed(0)}%`);
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
          );
        }
      },
    });
    loadedKey = key;
    // Expose for bench harness
    window._bench_model = model;
    window._bench_tokenizer = tokenizer;
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    setStatus(`loaded in ${dt}s · ${device}/${dtype}`);
    runBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus(`load failed: ${e.message}`);
  } finally {
    loadBtn.disabled = false;
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
  outEl.textContent = '';
  benchEl.textContent = '';
  setStatus('generating…');

  // Retrieval pre-rank: rewrite the candidate list inside the prompt.
  let retrievalInfo = 'retrieval: OFF';
  if (retrievalOn) {
    try {
      const userQuery = extractUserQuery(prompt);
      if (userQuery) {
        const { topNames, topScores } = await topKForQuery(userQuery, topK);
        const origCands = extractCandidateNames(prompt);
        const goldInTopK = origCands.length > 0
          ? origCands.some(n => topNames.includes(n))
          : null;
        prompt = rewriteCandidateList(prompt, topNames);
        retrievalInfo = `retrieval: top-${topK} = [${topNames.join(', ')}]  score(top1)=${topScores[0]?.toFixed(3) ?? 'n/a'}`;
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
  setStatus('done');
  runBtn.disabled = false;
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

// Preset selector — replace prompt with a per-domain SFT template.
const presetEl = $('preset');
if (presetEl) {
  presetEl.addEventListener('change', () => {
    const p = PRESETS[presetEl.value];
    if (p) $('prompt').value = p;
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

// Per-toggle info: attach a (i) icon next to each toggle that expands a help <div>.
function attachToggleHelp() {
  // Find the toggle row by anchoring on the constrained checkbox.
  const anchor = $('constrained');
  const toggleRow = anchor ? anchor.closest('.row') : null;
  if (!toggleRow) return;

  // Master "Show all help" / "Hide all help" link above the row.
  if (!document.getElementById('help-master')) {
    const wrap = document.createElement('div');
    wrap.id = 'help-master';
    const link = document.createElement('button');
    link.type = 'button';
    link.className = 'help-master-btn';
    link.textContent = 'ⓘ Show option help';
    let shown = false;
    link.addEventListener('click', (e) => {
      e.preventDefault();
      shown = !shown;
      for (const panel of document.querySelectorAll('.toggle-help')) {
        panel.hidden = !shown;
      }
      for (const b of document.querySelectorAll('.help-toggle')) {
        b.setAttribute('aria-expanded', String(shown));
      }
      link.textContent = shown ? 'ⓘ Hide option help' : 'ⓘ Show option help';
    });
    wrap.appendChild(link);
    toggleRow.before(wrap);
  }

  for (const [id, help] of Object.entries(TOGGLE_HELP)) {
    const input = $(id);
    if (!input) continue;
    const parentLabel = input.closest('label');
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
    // Insert after the toggle row's parent section so it can span full width.
    parentLabel.after(panel);
    const togglePanel = (e) => {
      // Prevent the click bubbling to the parent <label> from toggling the checkbox.
      if (e) { e.preventDefault(); e.stopPropagation(); }
      panel.hidden = !panel.hidden;
      btn.setAttribute('aria-expanded', String(!panel.hidden));
    };
    btn.addEventListener('click', togglePanel);
  }
}
attachToggleHelp();

// Bench legend (under the bench <pre>).
function renderBenchLegend() {
  const benchSection = benchEl ? benchEl.parentElement : null;
  if (!benchSection) return;
  if (document.getElementById('bench-legend')) return;
  const legend = document.createElement('div');
  legend.id = 'bench-legend';
  legend.className = 'info-card legend-card';
  const rows = BENCH_LEGEND.map(
    ([k, v]) => `<div><code>${k}</code> — ${v}</div>`,
  ).join('');
  legend.innerHTML = `<strong>What each bench line means</strong>${rows}`;
  benchSection.appendChild(legend);
}
renderBenchLegend();

// Footer with project + HF Hub links.
function renderFooter() {
  if (document.getElementById('site-footer')) return;
  const footer = document.createElement('footer');
  footer.id = 'site-footer';
  const links = FOOTER.links
    .map((l) => `<a href="${l.href}" target="_blank" rel="noopener">${l.label}</a>`)
    .join(' · ');
  footer.innerHTML = `<p>${FOOTER.blurb}</p><p class="links">${links}</p>`;
  document.querySelector('main').appendChild(footer);
}
renderFooter();

setStatus(detectWebGPU() ? 'WebGPU available · click Load' : 'WebGPU NOT available · WASM only');
