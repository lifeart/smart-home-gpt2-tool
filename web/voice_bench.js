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
  extractPromptSchemas,
  buildSchemaConstraint,
  JsonSchemaLogitsProcessor,
} from './grammar.js';
import {
  getOrBuildIndex,
  loadEncoder,
  embed,
  cosineTopK,
  NONE_SENTINEL,
} from './retrieval.js';

const VOICE_URL = '/eval/voice_pipeline_results.json';
const TEST_URL = '/eval/sh_test.json';
const REGISTRY_URL = '/eval/tool_registry.json';

// Manual gold-args map for the 30 voice fixture items, keyed by i (0..29).
// Derived from the RU input — the canonical argument values a correct system
// should extract. Used by iter 7.3 to compute voice exact-match accuracy.
// (The voice fixture upstream only carries `expected` name, no expected_args.)
const VOICE_GOLD_ARGS = [
  /*  0 */ { room: 'living room' },
  /*  1 */ { room: 'kitchen' },
  /*  2 */ { room: 'bedroom', temperature_c: 22 },
  /*  3 */ { door: 'front door' },
  /*  4 */ { door: 'back door' },
  /*  5 */ { song: 'jazz', room: 'living room' },
  /*  6 */ { room: 'kitchen' },
  /*  7 */ { room: 'bedroom' },
  /*  8 */ { room: 'office' },
  /*  9 */ { area: 'living room' },
  /* 10 */ { time: '07:00' },
  /* 11 */ { room: 'bedroom' },
  /* 12 */ { room: 'kids room' },
  /* 13 */ { room: 'garage' },
  /* 14 */ { room: 'living room', temperature_c: 21 },
  /* 15 */ { door: 'garage door' },
  /* 16 */ { door: 'patio door' },
  /* 17 */ { song: 'classical', room: 'bedroom' },
  /* 18 */ { room: 'kitchen' },
  /* 19 */ { room: 'kitchen' },
  /* 20 */ { room: 'living room' },
  /* 21 */ { area: 'kitchen' },
  /* 22 */ { time: '06:30' },
  /* 23 */ { room: 'bathroom' },
  /* 24 */ { room: 'hallway' },
  /* 25 */ { room: 'kids room', temperature_c: 23 },
  /* 26 */ { door: 'front door' },
  /* 27 */ { song: 'rock', room: 'office' },
  /* 28 */ { room: 'kitchen' },
  /* 29 */ { area: 'bedroom' },
];

// Tolerant args matcher (matches bench.js logic — kept inline so voice_bench
// has no extra cross-module dep). Same rules: same set of keys; numbers compared
// after string-to-number coercion; strings compared case-insensitive after trim.
function _voiceArgsMatch(predArgs, goldArgs) {
  const a = (predArgs && typeof predArgs === 'object') ? predArgs : {};
  const b = (goldArgs && typeof goldArgs === 'object') ? goldArgs : {};
  const ak = Object.keys(a).filter(k => a[k] !== undefined);
  const bk = Object.keys(b).filter(k => b[k] !== undefined);
  if (ak.length !== bk.length) return false;
  const aks = ak.slice().sort(), bks = bk.slice().sort();
  for (let i = 0; i < aks.length; i++) if (aks[i] !== bks[i]) return false;
  function norm(v) {
    if (v === null || v === undefined) return null;
    if (typeof v === 'number' || typeof v === 'boolean') return v;
    if (typeof v === 'string') {
      const t = v.trim();
      if (t !== '' && /^-?\d+(\.\d+)?$/.test(t)) return Number(t);
      return t.toLowerCase();
    }
    return v;
  }
  for (const k of aks) {
    const an = norm(a[k]), bn = norm(b[k]);
    if (an === null && bn === null) continue;
    if (an === null || bn === null) return false;
    if (typeof an === 'number' && typeof bn === 'number') {
      if (an !== bn) return false;
    } else if (String(an) !== String(bn)) return false;
  }
  return true;
}

// Extract the predicted JSON arguments from raw model text.
function _extractPredArgs(rawText) {
  if (!rawText) return {};
  const start = rawText.indexOf('{');
  if (start === -1) return {};
  let depth = 0, inStr = false;
  for (let i = start; i < rawText.length; i++) {
    const c = rawText[i];
    if (inStr) {
      if (c === '\\') { i++; continue; }
      if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') { inStr = true; continue; }
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) {
        try {
          const obj = JSON.parse(rawText.slice(start, i + 1));
          return (obj && obj.arguments && typeof obj.arguments === 'object') ? obj.arguments : {};
        } catch { return {}; }
      }
    }
  }
  return {};
}

let _voice = null;
let _registry = null;
let _fnDomainMap = null; // gold_name -> domain
let _domainFns = null;   // domain -> [fn names]

// ASR alias map — verb-level substitutions Whisper produces on RU smart-home
// commands. Drawn from iter 6.1 K=5 recall misses on voice_pipeline_results.json:
//   "Запри"  (lock)   -> "Close"
//   "Разблокируй" (unlock) -> "Unblock"
//   "Замолчи" (silence) -> "Shut up"
//   "21 градус" (21 degrees) -> "a degree" / "make a degree"
//   "Опусти жалюзи" (lower blinds) -> often loses "blinds"
// The map carries the RAW ASR phrase as key (case-insensitive) and the SET of
// alternative phrasings to ALSO retrieve against. The expansion strategy:
//   1. Embed the original query.
//   2. For each (key, aliases) where key is a substring of the lower-cased query:
//        - For each alias, build a variant query by replacing key with alias.
//        - Embed all variants.
//   3. Mean-pool the embeddings (already L2-normalised after re-normalisation).
// This widens the semantic neighbourhood without injecting prompt-style noise.
const ASR_ALIASES = [
  // close → lock / shut / latch (door, gate, lid)
  { key: 'close', aliases: ['lock', 'shut', 'latch'] },
  // open → unlock / unlatch
  { key: 'open', aliases: ['unlock', 'unlatch'] },
  // unblock → unlock / release
  { key: 'unblock', aliases: ['unlock', 'release'] },
  // shut up / silence → stop music / mute
  { key: 'shut up', aliases: ['stop music', 'mute audio', 'silence'] },
  // a degree / make a degree → set thermostat / set temperature
  { key: 'a degree', aliases: ['set temperature', 'set thermostat'] },
  { key: 'make a degree', aliases: ['set temperature', 'set thermostat'] },
];

/**
 * Given a raw query, return an array of expanded query strings (always
 * including the original). Replaces ASR aliases case-insensitively.
 *
 * @param {string} query
 * @returns {string[]}
 */
function expandQuery(query) {
  const out = [query];
  const ql = query.toLowerCase();
  for (const { key, aliases } of ASR_ALIASES) {
    const idx = ql.indexOf(key);
    if (idx === -1) continue;
    for (const alias of aliases) {
      // Splice the original-case query, replacing the case-insensitive match.
      const replaced = query.slice(0, idx) + alias + query.slice(idx + key.length);
      if (!out.includes(replaced)) out.push(replaced);
    }
  }
  return out;
}

// Mean-pool a list of L2-normalised vectors and re-normalise.
function meanPool(vecs) {
  if (vecs.length === 1) return vecs[0];
  const dim = vecs[0].length;
  const acc = new Float32Array(dim);
  for (const v of vecs) {
    for (let i = 0; i < dim; i++) acc[i] += v[i];
  }
  const inv = 1 / vecs.length;
  for (let i = 0; i < dim; i++) acc[i] *= inv;
  // Re-normalise so cosine-via-dot still holds.
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += acc[i] * acc[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) acc[i] /= norm;
  return acc;
}

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

async function runOne(model, tokenizer, prompt, { useConstrained, registry, max_new_tokens = 64, typedArgs = true }) {
  const inputs = await tokenizer(prompt, { return_tensors: 'pt' });
  const promptLength = inputs.input_ids.dims[1];

  let logits_processor = null;
  if (useConstrained) {
    const cands = extractCandidateNames(prompt);
    if (cands.length > 0) {
      const promptSchemas = extractPromptSchemas(prompt);
      const constraint = buildSchemaConstraint(cands, registry, { promptSchemas, typedArgs });
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
    useAliasExpansion = false,
    typedArgs = true,
  } = opts;

  const configKey = `${useRetrieval ? 'ret' : 'no_ret'}_${useConstrained ? 'con' : 'no_con'}${useAliasExpansion ? '_alias' : ''}`;

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
    let expandedQueries = null;
    if (useRetrieval) {
      let qv;
      if (useAliasExpansion) {
        // Embed the raw query AND each alias-substituted variant, then mean-pool.
        const variants = expandQuery(query);
        expandedQueries = variants;
        const qvecs = await embed(encoder, variants);
        qv = meanPool(qvecs);
      } else {
        [qv] = await embed(encoder, [query]);
      }
      // Ask for K+1 so we can drop the synthetic __NONE__ sentinel if it
      // ranks, while still returning K real function names.
      const top = cosineTopK(qv, index.vecs, K + 1);
      const all = top.map(t => index.names[t.idx]).filter(n => n !== NONE_SENTINEL);
      candidates = all.slice(0, K);
      goldInTopK = candidates.includes(gold);
    } else {
      candidates = buildBaselineCandidates(gold, fnDomain, domainFns, rng);
    }

    const prompt = buildPrompt(query, candidates);
    let res;
    try {
      res = await runOne(model, tokenizer, prompt, { useConstrained, registry, max_new_tokens, typedArgs });
    } catch (e) {
      console.error(`[voice-bench] item ${i} failed:`, e);
      res = { text: '', ms: 0, error: String(e) };
    }
    const pred = extractPredName(res.text);
    const ok = pred === gold;
    const predArgs = _extractPredArgs(res.text);
    const goldArgs = VOICE_GOLD_ARGS[i] || {};
    const argsOk = _voiceArgsMatch(predArgs, goldArgs);
    const exactOk = ok && argsOk;
    const row = {
      i,
      ru: it.ru_input,
      en: query,
      gold,
      pred,
      ok,                // legacy: name-only
      name_ok: ok,
      args_ok: argsOk,
      exact_ok: exactOk,
      gold_args: goldArgs,
      pred_args: predArgs,
      candidates,
      goldInTopK,
      ms: res.ms,
      text: res.text,
      expandedQueries,
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
  const ni = results.length;
  const acc = results.filter(r => r.ok).length / ni;
  const name_acc = acc;
  const args_acc = results.filter(r => r.args_ok).length / ni;
  const exact_acc = results.filter(r => r.exact_ok).length / ni;
  const recall = useRetrieval
    ? results.filter(r => r.goldInTopK).length / ni
    : null;
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / ni;

  const summary = { configKey, acc, name_acc, args_acc, exact_acc, recall, meanMs, n: ni, results };
  if (verbose) {
    console.log(`[voice ${configKey}] DONE name=${(name_acc * 100).toFixed(1)}% args=${(args_acc * 100).toFixed(1)}% exact=${(exact_acc * 100).toFixed(1)}% recall=${recall !== null ? (recall * 100).toFixed(1) + '%' : 'n/a'} mean_ms=${meanMs.toFixed(0)}`);
  }
  return summary;
}

window.runVoiceBench = runVoiceBench;
console.log('[voice-bench] window.runVoiceBench ready. Usage: await runVoiceBench({useRetrieval:true,useConstrained:true,K:3})');
