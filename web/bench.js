// A/B benchmark for constrained × retrieval decoding modes.
//
// Modes:
//   'baseline'  — retrieval OFF, constrained OFF
//   'con'       — retrieval OFF, constrained ON
//   'ret'       — retrieval ON,  constrained OFF
//   'ret_con'   — retrieval ON,  constrained ON
//
// Exposed on window.runBench(opts?) for one-shot devtools / MCP invocation.

import { LogitsProcessorList } from '@huggingface/transformers';
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
  retrieveTopK,
  NONE_SENTINEL,
} from './retrieval.js';

const SAMPLE_URL = '/eval/sh_test_sample.json';
const FULL_URL = '/eval/sh_test.json';
const REGISTRY_URL = '/eval/tool_registry.json';

let _sample = null;
let _full = null;
let _registry = null;

async function loadFixtures(url = SAMPLE_URL) {
  let sample;
  if (url === SAMPLE_URL) {
    if (!_sample) _sample = await fetch(SAMPLE_URL).then(r => r.json());
    sample = _sample;
  } else if (url === FULL_URL) {
    if (!_full) _full = await fetch(FULL_URL).then(r => r.json());
    sample = _full;
  } else {
    sample = await fetch(url).then(r => r.json());
  }
  if (!_registry) {
    _registry = await fetch(REGISTRY_URL).then(r => r.json());
  }
  return { sample, registry: _registry };
}

function extractPredictedName(rawText) {
  const m = rawText.match(/\{\s*"name"\s*:\s*"([^"]+)"/);
  return m ? m[1] : null;
}

// Extract the predicted JSON object as a string (the {...} starting at first '{').
function extractPredictedJsonString(rawText) {
  const start = rawText.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inStr = false;
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
      if (depth === 0) return rawText.slice(start, i + 1);
    }
  }
  return null;
}

// Parse the predicted JSON. Returns {name, arguments} or null if unparseable.
function parsePredictedCall(rawText) {
  const js = extractPredictedJsonString(rawText);
  if (!js) return null;
  try {
    const obj = JSON.parse(js);
    if (typeof obj !== 'object' || obj === null) return null;
    return {
      name: typeof obj.name === 'string' ? obj.name : null,
      arguments: (obj.arguments && typeof obj.arguments === 'object') ? obj.arguments : {},
    };
  } catch {
    return null;
  }
}

// Parse gold (already a JSON string). Returns {name, arguments}.
function parseGoldCall(goldStr) {
  if (!goldStr || typeof goldStr !== 'string') return { name: null, arguments: {} };
  try {
    const obj = JSON.parse(goldStr);
    return {
      name: typeof obj.name === 'string' ? obj.name : null,
      arguments: (obj.arguments && typeof obj.arguments === 'object') ? obj.arguments : {},
    };
  } catch {
    return { name: null, arguments: {} };
  }
}

// Normalize a single value for tolerant comparison.
function normalizeScalar(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v;
  if (typeof v === 'string') {
    const t = v.trim();
    if (t !== '' && /^-?\d+(\.\d+)?$/.test(t)) return Number(t);
    return t.toLowerCase();
  }
  return v;
}

// Compare two argument objects. Rules:
//  - Same set of keys (filter undefined).
//  - For each key: number → ==-with-coercion; string → case-insensitive trimmed eq;
//                  list → set equality; object → recurse.
//  - Empty == empty.
function argsMatch(predArgs, goldArgs) {
  const a = (predArgs && typeof predArgs === 'object') ? predArgs : {};
  const b = (goldArgs && typeof goldArgs === 'object') ? goldArgs : {};
  const ak = Object.keys(a).filter(k => a[k] !== undefined);
  const bk = Object.keys(b).filter(k => b[k] !== undefined);
  if (ak.length !== bk.length) return false;
  const aks = ak.slice().sort();
  const bks = bk.slice().sort();
  for (let i = 0; i < aks.length; i++) if (aks[i] !== bks[i]) return false;
  for (const k of aks) {
    const av = a[k];
    const bv = b[k];
    if (Array.isArray(bv)) {
      if (!Array.isArray(av)) return false;
      if (av.length !== bv.length) return false;
      const as = av.map(normalizeScalar).slice().sort();
      const bs = bv.map(normalizeScalar).slice().sort();
      for (let i = 0; i < as.length; i++) {
        if (JSON.stringify(as[i]) !== JSON.stringify(bs[i])) return false;
      }
    } else if (bv !== null && typeof bv === 'object') {
      if (av === null || typeof av !== 'object' || Array.isArray(av)) return false;
      if (!argsMatch(av, bv)) return false;
    } else {
      const an = normalizeScalar(av);
      const bn = normalizeScalar(bv);
      if (an === null && bn === null) continue;
      if (an === null || bn === null) return false;
      if (typeof an === 'number' && typeof bn === 'number') {
        if (an !== bn) return false;
      } else {
        if (String(an) !== String(bn)) return false;
      }
    }
  }
  return true;
}

// Per-arg-type bucket.
function argValueType(key, value, fnSchema) {
  if (fnSchema && fnSchema.params && fnSchema.params[key]) {
    const declared = String(fnSchema.params[key]).toLowerCase();
    if (declared === 'integer' || declared === 'int') return 'number';
    if (declared === 'number' || declared === 'float') return 'number';
    if (declared === 'boolean' || declared === 'bool') return 'boolean';
    if (declared === 'string') return 'string';
    if (declared.includes('|') || declared.startsWith('enum')) return 'enum';
    return declared;
  }
  if (typeof value === 'number') return 'number';
  if (typeof value === 'boolean') return 'boolean';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'object' && value !== null) return 'object';
  return 'string';
}

function isValidJson(rawText) {
  const start = rawText.indexOf('{');
  if (start === -1) return false;
  let depth = 0;
  let inStr = false;
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
        try { JSON.parse(rawText.slice(start, i + 1)); return true; }
        catch { return false; }
      }
    }
  }
  return false;
}

async function runOne(model, tokenizer, prompt, { constrained, registry, max_new_tokens = 64, typedArgs = true, wideNames = false }) {
  const inputs = await tokenizer(prompt, { return_tensors: 'pt' });
  const promptLength = inputs.input_ids.dims[1];

  let processor = null;
  let stats = null;
  if (constrained) {
    const cands = extractCandidateNames(prompt);
    if (cands.length === 0) {
      processor = null;
    } else {
      const promptSchemas = extractPromptSchemas(prompt);
      const constraint = buildSchemaConstraint(cands, registry, { promptSchemas, typedArgs, wideNames });
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
      processor = new LogitsProcessorList();
      processor.push(proc);
      stats = proc.stats;
    }
  }

  const t0 = performance.now();
  const output = await model.generate({
    ...inputs,
    max_new_tokens,
    do_sample: false,
    ...(processor ? { logits_processor: processor } : {}),
  });
  const t1 = performance.now();

  const newTokens = output.dims[1] - promptLength;
  const genIds = [];
  const data = output.data;
  for (let i = promptLength; i < output.dims[1]; i++) {
    genIds.push(Number(data[i]));
  }
  const text = tokenizer.decode(genIds, { skip_special_tokens: true });
  return {
    text,
    newTokens,
    ms: t1 - t0,
    perTokMs: stats && stats.steps > 0 ? stats.totalMs / stats.steps : null,
  };
}

// Apply retrieval to a prompt (returns rewritten prompt + topK metadata).
// If useSentinel is true (default), uses the gated retriever (retrieveTopK)
// that returns gold_none=true when the top-1 score is below `threshold` or
// the top-1 hit is the __NONE__ sentinel — in that case we rewrite the
// candidate list to ["__NONE__"] so the model is asked to decline.
async function retrievalRewrite(prompt, topK, { useSentinel = true, threshold = 0.30 } = {}) {
  const userQuery = extractUserQuery(prompt);
  if (!userQuery) return { prompt, topNames: null, topScores: null, gold_none: false };
  if (!useSentinel) {
    const { topNames, topScores } = await topKForQuery(userQuery, topK);
    const newPrompt = rewriteCandidateList(prompt, topNames);
    return { prompt: newPrompt, topNames, topScores, gold_none: false };
  }
  const idx = await getOrBuildIndex();
  const r = await retrieveTopK(userQuery, idx.vecs, idx.names, topK, threshold);
  if (r.gold_none) {
    // Rewrite candidate list to a single-element [__NONE__]. The model is now
    // asked to decline; if `gold_name` is empty (irrelevance class) and pred
    // is __NONE__ the row will be graded correct.
    const newPrompt = rewriteCandidateList(prompt, [NONE_SENTINEL]);
    return { prompt: newPrompt, topNames: [NONE_SENTINEL], topScores: r.topScores, gold_none: true };
  }
  const newPrompt = rewriteCandidateList(prompt, r.names);
  return { prompt: newPrompt, topNames: r.names, topScores: r.topScores, gold_none: false };
}

const ALL_MODES = ['baseline', 'con', 'ret', 'ret_con'];

// "decline class" — gold is missing / empty / __NONE__ ⇒ correct pred is __NONE__.
function isDeclineGold(item) {
  const gn = item.gold_name;
  const g = item.gold;
  if (gn === '' || gn === null || gn === undefined) return true;
  if (gn === NONE_SENTINEL) return true;
  if (g === '' || g === '{}' || g === null || g === undefined) return true;
  return false;
}

function gradePrediction(predName, item) {
  if (isDeclineGold(item)) return predName === NONE_SENTINEL;
  return predName === item.gold_name;
}

// Run a single item across modes and return the row.
async function runOneItem(item, idx, { modes, registry, max_new_tokens, topK, verbose, useSentinel, threshold, typedArgs = true, wideNames = false }) {
  const model = window._bench_model;
  const tokenizer = window._bench_tokenizer;
  const goldCall = parseGoldCall(item.gold);
  const row = {
    i: idx, domain: item.domain, gold_name: item.gold_name,
    gold_args: goldCall.arguments,
    gold_arg_keys: Object.keys(goldCall.arguments || {}),
  };

  let retPrompt = null, retTopNames = null, retTopScores = null, retGoldNone = false;
  if (modes.includes('ret') || modes.includes('ret_con')) {
    try {
      const r = await retrievalRewrite(item.prompt, topK, { useSentinel, threshold });
      retPrompt = r.prompt;
      retTopNames = r.topNames;
      retTopScores = r.topScores;
      retGoldNone = r.gold_none;
      row.ret_topK = retTopNames;
      row.ret_top1_score = retTopScores ? retTopScores[0] : null;
      row.ret_gold_none = retGoldNone;
      // gold_in_topK: under sentinel-fired path, gold can still be "in" the
      // 1-element [__NONE__] list iff this is an irrelevance gold.
      if (retGoldNone) {
        row.gold_in_topK = isDeclineGold(item);
      } else {
        row.gold_in_topK = retTopNames ? retTopNames.includes(item.gold_name) : null;
      }
    } catch (e) {
      console.error('retrieval rewrite failed:', e);
      row.ret_error = String(e);
    }
  }

  for (const mode of modes) {
    const useRet = mode === 'ret' || mode === 'ret_con';
    const useCon = mode === 'con' || mode === 'ret_con';
    const promptForMode = useRet ? (retPrompt || item.prompt) : item.prompt;
    try {
      const out = await runOne(model, tokenizer, promptForMode, {
        constrained: useCon, registry, max_new_tokens, typedArgs, wideNames,
      });
      const name = extractPredictedName(out.text);
      const predCall = parsePredictedCall(out.text);
      const predArgs = predCall ? predCall.arguments : {};
      row[`${mode}_name`] = name;
      row[`${mode}_pred_args`] = predArgs;
      // Under retrieval+sentinel: if the candidate list was rewritten to
      // [__NONE__] and the model emitted that name, this is the decline class.
      // Grade against gold_name with decline-aware comparison.
      const nameOk = useRet ? gradePrediction(name, item) : (name === item.gold_name);
      row[`${mode}_ok`] = nameOk;                    // name-only (legacy metric)
      row[`${mode}_name_ok`] = nameOk;               // explicit alias
      const argsOk = argsMatch(predArgs, goldCall.arguments);
      row[`${mode}_args_ok`] = argsOk;
      row[`${mode}_exact_ok`] = nameOk && argsOk;
      row[`${mode}_json_ok`] = isValidJson(out.text);
      row[`${mode}_ms`] = out.ms;
      row[`${mode}_tokens`] = out.newTokens;
      row[`${mode}_text`] = out.text;
      if (mode === 'con' || mode === 'ret_con') {
        row[`${mode}_overhead_per_step_ms`] = out.perTokMs;
      }
    } catch (e) {
      console.error(`mode ${mode} failed:`, e);
      row[`${mode}_error`] = String(e);
      row[`${mode}_ok`] = false;
      row[`${mode}_name_ok`] = false;
      row[`${mode}_args_ok`] = false;
      row[`${mode}_exact_ok`] = false;
      row[`${mode}_json_ok`] = false;
      row[`${mode}_ms`] = 0;
    }
  }

  if (verbose) {
    const cells = modes.map(m => `${m}=${row[`${m}_name`]} ${row[`${m}_ok`] ? '✓' : '✗'}`).join('  ');
    console.log(`   ${cells}  gold-in-topK=${row.gold_in_topK ?? 'n/a'}`);
  }
  return row;
}

// Run bench on a given list of items (no fixture loading, no aggregation).
async function runBenchOnItems(items, {
  modes = ALL_MODES,
  max_new_tokens = 64,
  topK = 3,
  verbose = false,
  registry = null,
  useSentinel = true,
  threshold = 0.30,
  typedArgs = true,
  wideNames = false,
} = {}) {
  const model = window._bench_model;
  const tokenizer = window._bench_tokenizer;
  if (!model || !tokenizer) {
    throw new Error('Load a model first (use the Load button), then re-run.');
  }
  if (!registry) {
    const f = await loadFixtures();
    registry = f.registry;
  }
  if (modes.some(m => m === 'ret' || m === 'ret_con')) {
    await getOrBuildIndex();
  }
  const results = [];
  for (let i = 0; i < items.length; i++) {
    const row = await runOneItem(items[i], i, { modes, registry, max_new_tokens, topK, verbose, useSentinel, threshold, typedArgs, wideNames });
    results.push(row);
  }
  return { results };
}

function aggregateResults(results, modes = ALL_MODES) {
  const n_items = results.length;
  const sum = (k) => results.reduce((a, r) => a + (Number(r[k]) || 0), 0);
  const cnt = (pred) => results.filter(pred).length;

  const summary = { items: n_items, modes };
  for (const m of modes) {
    summary[`acc_${m}`] = cnt(r => r[`${m}_ok`]) / n_items;          // name-only acc (legacy)
    summary[`name_acc_${m}`] = cnt(r => r[`${m}_name_ok`]) / n_items;
    summary[`args_acc_${m}`] = cnt(r => r[`${m}_args_ok`]) / n_items;
    summary[`exact_acc_${m}`] = cnt(r => r[`${m}_exact_ok`]) / n_items;
    summary[`json_${m}`] = cnt(r => r[`${m}_json_ok`]) / n_items;
    summary[`mean_ms_${m}`] = sum(`${m}_ms`) / n_items;
  }
  if (modes.includes('ret') || modes.includes('ret_con')) {
    const withTop = results.filter(r => r.gold_in_topK !== null && r.gold_in_topK !== undefined);
    summary.gold_in_topK = withTop.length
      ? cnt(r => r.gold_in_topK === true) / withTop.length
      : null;
  }

  // Per-domain breakdown.
  const domains = Array.from(new Set(results.map(r => r.domain))).sort();
  const perDomain = {};
  for (const d of domains) {
    const sub = results.filter(r => r.domain === d);
    const row = { n: sub.length };
    for (const m of modes) {
      row[`acc_${m}`] = sub.filter(r => r[`${m}_ok`]).length / sub.length;
      row[`name_acc_${m}`] = sub.filter(r => r[`${m}_name_ok`]).length / sub.length;
      row[`args_acc_${m}`] = sub.filter(r => r[`${m}_args_ok`]).length / sub.length;
      row[`exact_acc_${m}`] = sub.filter(r => r[`${m}_exact_ok`]).length / sub.length;
    }
    perDomain[d] = row;
  }
  summary.per_domain = perDomain;
  return summary;
}

/**
 * Aggregate args-level stats. Returns per-mode metrics including:
 *   - name_acc, args_acc, exact_acc
 *   - args_acc_by_type: per-arg-value bucket (type -> {ok, n, acc})
 *   - args_acc_by_key_count: per-item bucket by # of gold-arg keys
 *   - per_domain_args: { domain: { exact_acc, args_acc, name_acc } }
 *
 * @param {Array<Object>} results
 * @param {Object} registry  parsed tool_registry.json
 * @param {string[]} modes
 */
function aggregateArgs(results, registry = null, modes = ALL_MODES) {
  const out = { items: results.length, modes: {} };

  for (const m of modes) {
    const mode = {
      name_acc: 0, args_acc: 0, exact_acc: 0,
      args_acc_by_type: {},
      args_acc_by_key_count: { '0': { ok: 0, n: 0 }, '1': { ok: 0, n: 0 }, '2': { ok: 0, n: 0 }, '3+': { ok: 0, n: 0 } },
      per_domain_args: {},
    };
    const n = results.length || 1;
    mode.name_acc = results.filter(r => r[`${m}_name_ok`]).length / n;
    mode.args_acc = results.filter(r => r[`${m}_args_ok`]).length / n;
    mode.exact_acc = results.filter(r => r[`${m}_exact_ok`]).length / n;

    // Per-arg-value-type accuracy.
    for (const r of results) {
      const goldArgs = r.gold_args || {};
      const predArgs = r[`${m}_pred_args`] || {};
      const fnSchema = (registry && registry[r.gold_name]) || null;
      for (const k of Object.keys(goldArgs)) {
        const t = argValueType(k, goldArgs[k], fnSchema);
        if (!mode.args_acc_by_type[t]) mode.args_acc_by_type[t] = { ok: 0, n: 0 };
        mode.args_acc_by_type[t].n++;
        if (Object.prototype.hasOwnProperty.call(predArgs, k)) {
          const single = argsMatch({ [k]: predArgs[k] }, { [k]: goldArgs[k] });
          if (single) mode.args_acc_by_type[t].ok++;
        }
      }
    }
    for (const t of Object.keys(mode.args_acc_by_type)) {
      const b = mode.args_acc_by_type[t];
      b.acc = b.n ? b.ok / b.n : 0;
    }

    // Args acc by gold key-count (whole-args-match, per-item).
    for (const r of results) {
      const kc = (r.gold_arg_keys || []).length;
      const bucket = kc >= 3 ? '3+' : String(kc);
      mode.args_acc_by_key_count[bucket].n++;
      if (r[`${m}_args_ok`]) mode.args_acc_by_key_count[bucket].ok++;
    }
    for (const b of Object.keys(mode.args_acc_by_key_count)) {
      const x = mode.args_acc_by_key_count[b];
      x.acc = x.n ? x.ok / x.n : 0;
    }

    // Per-domain.
    const domains = Array.from(new Set(results.map(r => r.domain))).sort();
    for (const d of domains) {
      const sub = results.filter(r => r.domain === d);
      mode.per_domain_args[d] = {
        n: sub.length,
        name_acc: sub.filter(r => r[`${m}_name_ok`]).length / sub.length,
        args_acc: sub.filter(r => r[`${m}_args_ok`]).length / sub.length,
        exact_acc: sub.filter(r => r[`${m}_exact_ok`]).length / sub.length,
      };
    }
    out.modes[m] = mode;
  }
  return out;
}

async function runBench({
  n = null,
  max_new_tokens = 64,
  verbose = true,
  modes = null,
  topK = 3,
  useSentinel = true,
  threshold = 0.30,
  typedArgs = true,
  wideNames = false,
} = {}) {
  modes = modes || ALL_MODES;
  const { sample, registry } = await loadFixtures();
  const items = n ? sample.slice(0, n) : sample;
  const model = window._bench_model;
  const tokenizer = window._bench_tokenizer;
  if (!model || !tokenizer) {
    throw new Error('Load a model first (use the Load button), then re-run.');
  }

  if (modes.some(m => m === 'ret' || m === 'ret_con')) {
    if (verbose) console.log('[bench] warming MiniLM index…');
    await getOrBuildIndex();
  }

  window._partialBench = [];

  const results = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (verbose) console.log(`[${i + 1}/${items.length}] ${item.domain} | ${item.gold_name}`);
    const row = await runOneItem(item, i, { modes, registry, max_new_tokens, topK, verbose, useSentinel, threshold, typedArgs, wideNames });
    results.push(row);
    window._partialBench.push(row);
  }

  const summary = aggregateResults(results, modes);
  const cnt = (pred) => results.filter(pred).length;

  const rescues = (modes.includes('baseline') && modes.includes('ret_con'))
    ? results.filter(r => !r.baseline_ok && r.ret_con_ok).slice(0, 5)
    : [];
  const ret_rescues_over_con = (modes.includes('con') && modes.includes('ret_con'))
    ? results.filter(r => !r.con_ok && r.ret_con_ok).slice(0, 5)
    : [];
  const top_misses = results.filter(r => r.gold_in_topK === false).slice(0, 5);

  if (verbose) {
    console.log('=== A/B SUMMARY (n=' + results.length + ') ===');
    console.table(summary);
  }
  return { summary, results, rescues, ret_rescues_over_con, top_misses };
}

// Chunked full bench: streams progress through window._fullProgress / _fullResults.
async function runFullBench({
  chunkSize = 50,
  modes = ALL_MODES,
  max_new_tokens = 64,
  topK = 3,
  url = FULL_URL,
  useSentinel = true,
  threshold = 0.30,
  typedArgs = true,
  wideNames = false,
} = {}) {
  const { sample: all, registry } = await loadFixtures(url);
  window._fullProgress = { done: 0, total: all.length, started: Date.now() };
  window._fullResults = [];
  // Pre-warm
  if (modes.some(m => m === 'ret' || m === 'ret_con')) {
    await getOrBuildIndex();
  }
  for (let off = 0; off < all.length; off += chunkSize) {
    const slice = all.slice(off, off + chunkSize);
    const { results } = await runBenchOnItems(slice, {
      modes, max_new_tokens, topK, verbose: false, registry, useSentinel, threshold, typedArgs, wideNames,
    });
    // Re-index items so .i is global, not per-chunk.
    for (let k = 0; k < results.length; k++) {
      results[k].i = off + k;
    }
    window._fullResults.push(...results);
    window._fullProgress.done = window._fullResults.length;
    console.log(`[fullbench] ${window._fullProgress.done}/${window._fullProgress.total} done`);
  }
  const summary = aggregateResults(window._fullResults, modes);
  const argsSummary = aggregateArgs(window._fullResults, registry, modes);
  summary.args = argsSummary;
  window._fullSummary = summary;
  window._fullArgsSummary = argsSummary;
  return summary;
}

window.runBench = runBench;
window.runBenchOnItems = runBenchOnItems;
window.runFullBench = runFullBench;
window.aggregateFullBench = (results, modes = ALL_MODES) => aggregateResults(results, modes);
window.aggregateArgs = (results, registry = null, modes = ALL_MODES) => aggregateArgs(results, registry, modes);
window._bench = {
  extractCandidateNames, extractPromptSchemas, buildSchemaConstraint,
  parsePredictedCall, parseGoldCall, argsMatch, argValueType, aggregateArgs,
};
console.log('[bench] window.runBench / window.runFullBench ready.');
