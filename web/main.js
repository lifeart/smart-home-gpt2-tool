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
      const constraint = buildSchemaConstraint(cands, registry, { promptSchemas, typedArgs });
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

setStatus(detectWebGPU() ? 'WebGPU available · click Load' : 'WebGPU NOT available · WASM only');
