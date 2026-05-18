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
  buildSchemaConstraint,
  JsonSchemaLogitsProcessor,
} from './grammar.js';
import {
  extractUserQuery,
  topKForQuery,
  rewriteCandidateList,
  getOrBuildIndex,
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

async function runOne(model, tokenizer, prompt, { constrained, registry, max_new_tokens = 64 }) {
  const inputs = await tokenizer(prompt, { return_tensors: 'pt' });
  const promptLength = inputs.input_ids.dims[1];

  let processor = null;
  let stats = null;
  if (constrained) {
    const cands = extractCandidateNames(prompt);
    if (cands.length === 0) {
      processor = null;
    } else {
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
async function retrievalRewrite(prompt, topK) {
  const userQuery = extractUserQuery(prompt);
  if (!userQuery) return { prompt, topNames: null, topScores: null };
  const { topNames, topScores } = await topKForQuery(userQuery, topK);
  const newPrompt = rewriteCandidateList(prompt, topNames);
  return { prompt: newPrompt, topNames, topScores };
}

const ALL_MODES = ['baseline', 'con', 'ret', 'ret_con'];

// Run a single item across modes and return the row.
async function runOneItem(item, idx, { modes, registry, max_new_tokens, topK, verbose }) {
  const model = window._bench_model;
  const tokenizer = window._bench_tokenizer;
  const row = {
    i: idx, domain: item.domain, gold_name: item.gold_name,
  };

  let retPrompt = null, retTopNames = null, retTopScores = null;
  if (modes.includes('ret') || modes.includes('ret_con')) {
    try {
      const r = await retrievalRewrite(item.prompt, topK);
      retPrompt = r.prompt;
      retTopNames = r.topNames;
      retTopScores = r.topScores;
      row.ret_topK = retTopNames;
      row.ret_top1_score = retTopScores ? retTopScores[0] : null;
      row.gold_in_topK = retTopNames ? retTopNames.includes(item.gold_name) : null;
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
        constrained: useCon, registry, max_new_tokens,
      });
      const name = extractPredictedName(out.text);
      row[`${mode}_name`] = name;
      row[`${mode}_ok`] = name === item.gold_name;
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
    const row = await runOneItem(items[i], i, { modes, registry, max_new_tokens, topK, verbose });
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
    summary[`acc_${m}`] = cnt(r => r[`${m}_ok`]) / n_items;
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
    }
    perDomain[d] = row;
  }
  summary.per_domain = perDomain;
  return summary;
}

async function runBench({
  n = null,
  max_new_tokens = 64,
  verbose = true,
  modes = null,
  topK = 3,
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
    const row = await runOneItem(item, i, { modes, registry, max_new_tokens, topK, verbose });
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
      modes, max_new_tokens, topK, verbose: false, registry,
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
  window._fullSummary = summary;
  return summary;
}

window.runBench = runBench;
window.runBenchOnItems = runBenchOnItems;
window.runFullBench = runFullBench;
window.aggregateFullBench = (results, modes = ALL_MODES) => aggregateResults(results, modes);
window._bench = { extractCandidateNames, buildSchemaConstraint };
console.log('[bench] window.runBench / window.runFullBench ready.');
