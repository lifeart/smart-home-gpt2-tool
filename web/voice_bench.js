// Voice E2E benchmark on noisy Whisper transcripts.
// Loads `voice_pipeline_results.json` (30 RU→Whisper noisy EN items), runs
// the model under 4 configurations:
//   1. baseline  — fixed candidate list (gold + 4 random from same domain), no constrained
//   2. con       — same candidate list + constrained decoding
//   3. ret       — retrieval pre-rank (top-K), no constrained
//   4. ret_con   — retrieval pre-rank + constrained decoding
//
// Exposed on window.runVoiceBench / window._voiceProgress / window._voiceResults.

import { LogitsProcessorList } from '@huggingface/transformers';
import {
  extractCandidateNames,
  buildSchemaConstraint,
  JsonSchemaLogitsProcessor,
} from './grammar.js';
import {
  getOrBuildIndex,
  loadEncoder,
  embed,
  cosineTopK,
} from './retrieval.js';

const VOICE_URL = '/eval/voice_pipeline_results.json';
const TEST_URL = '/eval/sh_test.json';
const REGISTRY_URL = '/eval/tool_registry.json';

let _voice = null;
let _registry = null;
let _fnDomainMap = null; // gold_name -> domain
let _domainFns = null;   // domain -> [fn names]

// Stable PRNG so baseline runs are reproducible (gold + 4 random domain peers).
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function loadFixtures() {
  if (!_voice) {
    const r = await fetch(VOICE_URL).then(r => r.json());
    _voice = r.results;
  }
  if (!_registry) {
    _registry = await fetch(REGISTRY_URL).then(r => r.json());
  }
  if (!_fnDomainMap) {
    // Build fn -> domain map from sh_test.json (the only place we have domain labels).
    const test = await fetch(TEST_URL).then(r => r.json());
    _fnDomainMap = {};
    _domainFns = {};
    for (const it of test) {
      if (!_fnDomainMap[it.gold_name]) _fnDomainMap[it.gold_name] = it.domain;
      if (!_domainFns[it.domain]) _domainFns[it.domain] = new Set();
      _domainFns[it.domain].add(it.gold_name);
    }
    // Voice fixture has two gold names not present in sh_test.json — patch their
    // domains so the "gold + 4 random domain neighbours" baseline still finds
    // plausible peers (instead of falling back to misc).
    if (!_fnDomainMap['lock_door']) _fnDomainMap['lock_door'] = 'sec';
    if (!_fnDomainMap['open_curtains']) _fnDomainMap['open_curtains'] = 'blinds';
    // Convert sets to arrays
    for (const d of Object.keys(_domainFns)) {
      _domainFns[d] = Array.from(_domainFns[d]);
    }
    // Make sure those names also appear in their domain pool so they can be drawn
    // as peers for OTHER voice items in the same domain.
    if (!_domainFns['sec'].includes('lock_door')) _domainFns['sec'].push('lock_door');
    if (!_domainFns['blinds'].includes('open_curtains')) _domainFns['blinds'].push('open_curtains');
  }
  return { voice: _voice, registry: _registry, fnDomain: _fnDomainMap, domainFns: _domainFns };
}

function buildPrompt(noisyEn, candidates) {
  const tools = JSON.stringify(candidates, null, 2);
  return `SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
${tools}


USER: ${noisyEn}


ASSISTANT: <functioncall> `;
}

// "gold + 4 random from same domain" — matches Phase 2 voice bench methodology.
function buildBaselineCandidates(gold, fnDomain, domainFns, rng) {
  const domain = fnDomain[gold] || 'misc';
  const pool = (domainFns[domain] || []).filter(n => n !== gold);
  // Shuffle and pick 4
  const shuffled = pool.slice().sort(() => rng() - 0.5);
  const peers = shuffled.slice(0, 4);
  // Place gold at a fixed-ish index (index 2 of 5) so position bias is constant.
  const cands = [...peers];
  cands.splice(2, 0, gold);
  return cands.slice(0, 5);
}

function extractPredName(text) {
  const m = text.match(/[\"']?name[\"']?\s*:\s*[\"']([^\"'(\s,}]+)/);
  return m ? m[1] : null;
}

async function runOne(model, tokenizer, prompt, { useConstrained, registry, max_new_tokens = 64 }) {
  const inputs = await tokenizer(prompt, { return_tensors: 'pt' });
  const promptLength = inputs.input_ids.dims[1];

  let logits_processor = null;
  if (useConstrained) {
    const cands = extractCandidateNames(prompt);
    if (cands.length > 0) {
      const constraint = buildSchemaConstraint(cands, registry);
      const eosTokenId =
        tokenizer.eos_token_id ??
        (model.generation_config && model.generation_config.eos_token_id);
      const always = new Set();
      if (eosTokenId !== undefined && eosTokenId !== null) always.add(eosTokenId);
      const proc = new JsonSchemaLogitsProcessor({
        tokenizer,
        promptLength,
        constraint,
        topK: 40,
        allowAlwaysTokens: always,
      });
      logits_processor = new LogitsProcessorList();
      logits_processor.push(proc);
    }
  }

  const t0 = performance.now();
  const out = await model.generate({
    ...inputs,
    max_new_tokens,
    do_sample: false,
    ...(logits_processor ? { logits_processor } : {}),
  });
  const ms = performance.now() - t0;
  const ids = [];
  for (let i = promptLength; i < out.dims[1]; i++) ids.push(Number(out.data[i]));
  const text = tokenizer.decode(ids, { skip_special_tokens: true });
  return { text, ms };
}

/**
 * Run the voice E2E bench under one configuration.
 *
 * @param {{useRetrieval?: boolean, useConstrained?: boolean, K?: number,
 *          max_new_tokens?: number, verbose?: boolean, n?: number}} opts
 * @returns {Promise<{acc:number, recall:number|null, meanMs:number, n:number,
 *                    results:Array, configKey:string}>}
 */
async function runVoiceBench(opts = {}) {
  const {
    useRetrieval = true,
    useConstrained = true,
    K = 3,
    max_new_tokens = 64,
    verbose = true,
    n = null,
  } = opts;

  const configKey = `${useRetrieval ? 'ret' : 'no_ret'}_${useConstrained ? 'con' : 'no_con'}`;

  const { voice, registry, fnDomain, domainFns } = await loadFixtures();
  const model = window._bench_model;
  const tokenizer = window._bench_tokenizer;
  if (!model || !tokenizer) {
    throw new Error('Load a model first (use the Load button), then re-run.');
  }

  // Warm retrieval index if needed.
  let index = null;
  let encoder = null;
  if (useRetrieval) {
    if (verbose) console.log('[voice-bench] warming MiniLM index…');
    index = await getOrBuildIndex();
    encoder = await loadEncoder();
  }

  const items = n ? voice.slice(0, n) : voice;
  window._voiceProgress = { done: 0, total: items.length, config: configKey };
  window._voiceResults = [];

  const rng = mulberry32(42); // stable seed for baseline domain-neighbours

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const query = it.en_from_whisper;
    const gold = it.expected;

    let candidates;
    let goldInTopK = null;
    if (useRetrieval) {
      const [qv] = await embed(encoder, [query]);
      const top = cosineTopK(qv, index.vecs, K);
      candidates = top.map(t => index.names[t.idx]);
      goldInTopK = candidates.includes(gold);
    } else {
      candidates = buildBaselineCandidates(gold, fnDomain, domainFns, rng);
    }

    const prompt = buildPrompt(query, candidates);
    let res;
    try {
      res = await runOne(model, tokenizer, prompt, { useConstrained, registry, max_new_tokens });
    } catch (e) {
      console.error(`[voice-bench] item ${i} failed:`, e);
      res = { text: '', ms: 0, error: String(e) };
    }
    const pred = extractPredName(res.text);
    const ok = pred === gold;
    const row = {
      i,
      ru: it.ru_input,
      en: query,
      gold,
      pred,
      ok,
      candidates,
      goldInTopK,
      ms: res.ms,
      text: res.text,
    };
    window._voiceResults.push(row);
    window._voiceProgress.done = window._voiceResults.length;
    if (verbose) {
      console.log(
        `[voice ${configKey}] [${i + 1}/${items.length}] gold=${gold} pred=${pred} ${ok ? 'OK' : 'X'} ` +
          (useRetrieval ? `topK=[${candidates.join(',')}]` : `cands=[${candidates.join(',')}]`),
      );
    }
  }

  const results = window._voiceResults;
  const acc = results.filter(r => r.ok).length / results.length;
  const recall = useRetrieval
    ? results.filter(r => r.goldInTopK).length / results.length
    : null;
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / results.length;

  const summary = { configKey, acc, recall, meanMs, n: results.length, results };
  if (verbose) {
    console.log(`[voice ${configKey}] DONE acc=${(acc * 100).toFixed(1)}% recall=${recall !== null ? (recall * 100).toFixed(1) + '%' : 'n/a'} mean_ms=${meanMs.toFixed(0)}`);
  }
  return summary;
}

window.runVoiceBench = runVoiceBench;
console.log('[voice-bench] window.runVoiceBench ready. Usage: await runVoiceBench({useRetrieval:true,useConstrained:true,K:3})');
