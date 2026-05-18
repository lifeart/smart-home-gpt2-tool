// MiniLM-based retrieval pre-rank for smart-home function call.
//
// Strategy:
//   1. Build an index where each of the 100 functions is embedded as
//      "<name> — <description>. parameters: <p1, p2, …>"  (Float32 normalized)
//   2. At query time, embed the user query, dot-product against all 100,
//      return top-K function names.
//   3. The caller rewrites the candidate list in the prompt to those top-K.
//
// Uses Xenova/all-MiniLM-L6-v2 (~22 MB). Tries WebGPU first, falls back to
// WASM if the WebGPU pipeline fails to materialize for this model.

import { pipeline, env } from '@huggingface/transformers';

// Don't reuse env.localModelPath from main.js for the encoder — we want to
// fetch MiniLM from the HF Hub, not from /models/.
const MINILM_ID = 'Xenova/all-MiniLM-L6-v2';

let _encoder = null;
let _encoderDevice = null;

/**
 * Lazy-load the MiniLM encoder. Caches the pipeline.
 */
export async function loadEncoder() {
  if (_encoder) return _encoder;
  // Force remote-only fetch for MiniLM. Save & restore the global env flags
  // so the GPT-2 loader (which DOES use /models/) keeps working.
  const prevAllowRemote = env.allowRemoteModels;
  const prevAllowLocal = env.allowLocalModels;
  const prevUseCache = env.useBrowserCache;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  // Disable browser cache to avoid getting stuck on previously-cached HTML 404s
  // from any prior misconfigured load. Encoder load is one-shot per session.
  env.useBrowserCache = false;
  let lastErr = null;
  // Try (device, dtype) combinations:
  //   webgpu/fp32 — best accuracy, ~86 MB on the wire
  //   wasm/q8    — fallback, ~22 MB on the wire
  // For Phase 4 we want the small one; webgpu encoders can have ONNX compat
  // hiccups for some MiniLM exports, so fall through to wasm/q8 quickly.
  const attempts = [
    { device: 'webgpu', dtype: 'fp32' },
    { device: 'wasm',   dtype: 'q8'   },
    { device: 'wasm',   dtype: 'fp32' },
  ];
  for (const { device, dtype } of attempts) {
    try {
      const enc = await pipeline('feature-extraction', MINILM_ID, {
        device,
        dtype,
      });
      _encoder = enc;
      _encoderDevice = `${device}/${dtype}`;
      console.log(`[retrieval] MiniLM loaded on ${device}/${dtype}`);
      env.allowRemoteModels = prevAllowRemote;
      env.allowLocalModels = prevAllowLocal;
      env.useBrowserCache = prevUseCache;
      return enc;
    } catch (e) {
      console.warn(`[retrieval] MiniLM ${device}/${dtype} load failed:`, e?.message || e);
      lastErr = e;
    }
  }
  env.allowRemoteModels = prevAllowRemote;
  env.allowLocalModels = prevAllowLocal;
  env.useBrowserCache = prevUseCache;
  throw lastErr || new Error('MiniLM load failed on every device');
}

export function getEncoderDevice() {
  return _encoderDevice;
}

/**
 * Build the per-function index text.
 * @param {Record<string, {description: string, params: string[]}>} descriptions
 * @param {Record<string, {params: Record<string,string>, required: string[]}>} registry
 * @returns {{names: string[], texts: string[]}}
 */
export function buildFunctionIndex(descriptions, registry) {
  // Index every function we have a description for — this is a SUPERSET of
  // `registry` since some gold labels in sh_test.json are not in the registry.
  // The registry is still authoritative for constrained decoding (it carries
  // param shapes); the index is authoritative for retrieval (it carries
  // semantic descriptions).
  const allNames = new Set([...Object.keys(registry), ...Object.keys(descriptions)]);
  const names = Array.from(allNames).sort();
  const texts = names.map(name => {
    const d = descriptions[name] || {};
    const desc = d.description || '';
    const params = (d.params && d.params.length)
      ? d.params
      : Object.keys((registry[name] && registry[name].params) || {});
    const paramStr = params.length ? `parameters: ${params.join(', ')}` : 'no parameters';
    // Recipe: "<name> — <desc>. parameters: <…>. examples: <q1> | <q2> | <q3>"
    // The examples (drawn from training prompts) close the lexical gap between
    // formal function names/descriptions and how users phrase commands.
    const examples = Array.isArray(d.examples) ? d.examples.slice(0, 3) : [];
    const exStr = examples.length ? ` examples: ${examples.join(' | ')}` : '';
    return `${name} — ${desc} ${paramStr}${exStr}`.trim();
  });
  return { names, texts };
}

/**
 * Embed a batch of texts with mean pooling + L2 normalization.
 * Returns Array<Float32Array>.
 */
export async function embed(encoder, texts) {
  if (!Array.isArray(texts)) texts = [texts];
  // The pipeline supports batch input directly.
  const out = await encoder(texts, { pooling: 'mean', normalize: true });
  // `out` is a Tensor of shape [N, dim]. Slice into individual Float32Arrays.
  const dims = out.dims;
  const N = dims[0];
  const dim = dims[1];
  const data = out.data; // Float32Array length N*dim
  const vecs = [];
  for (let i = 0; i < N; i++) {
    const slice = new Float32Array(dim);
    slice.set(data.subarray(i * dim, (i + 1) * dim));
    vecs.push(slice);
  }
  return vecs;
}

/**
 * Cosine similarity top-K. Assumes both query and index vectors are L2-normalized,
 * so dot product == cosine.
 * @param {Float32Array} queryVec
 * @param {Float32Array[]} indexVecs
 * @param {number} k
 * @returns {Array<{idx: number, score: number}>}
 */
export function cosineTopK(queryVec, indexVecs, k = 3) {
  const scores = new Array(indexVecs.length);
  const dim = queryVec.length;
  for (let i = 0; i < indexVecs.length; i++) {
    const v = indexVecs[i];
    let s = 0;
    for (let d = 0; d < dim; d++) s += queryVec[d] * v[d];
    scores[i] = { idx: i, score: s };
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k);
}

// ---------- High-level helpers used by main.js / bench.js ----------

let _indexCache = null; // { names, texts, vecs }

/**
 * Build the function index (names+texts), and on first call also embed all
 * function texts. Subsequent calls return the cached index.
 *
 * @returns {Promise<{names:string[], texts:string[], vecs:Float32Array[]}>}
 */
export async function getOrBuildIndex() {
  if (_indexCache && _indexCache.vecs) return _indexCache;
  const [descriptions, registry] = await Promise.all([
    fetch('/eval/function_descriptions.json').then(r => r.json()),
    fetch('/eval/tool_registry.json').then(r => r.json()),
  ]);
  const { names, texts } = buildFunctionIndex(descriptions, registry);
  const encoder = await loadEncoder();
  const t0 = performance.now();
  const vecs = await embed(encoder, texts);
  const dt = performance.now() - t0;
  console.log(`[retrieval] embedded ${vecs.length} function texts in ${dt.toFixed(0)} ms (device=${_encoderDevice})`);
  _indexCache = { names, texts, vecs };
  return _indexCache;
}

/**
 * Extract the user query from the prompt — between the last `USER:` line and
 * the next blank line / `ASSISTANT:` marker.
 *
 * @param {string} prompt
 * @returns {string}
 */
export function extractUserQuery(prompt) {
  // Find last USER: occurrence
  const idx = prompt.lastIndexOf('USER:');
  if (idx === -1) return '';
  const after = prompt.slice(idx + 'USER:'.length);
  // Take up to the first `\n\n` or `ASSISTANT:` marker
  let end = after.length;
  const m1 = after.indexOf('\n\n');
  const m2 = after.indexOf('ASSISTANT:');
  if (m1 !== -1) end = Math.min(end, m1);
  if (m2 !== -1) end = Math.min(end, m2);
  return after.slice(0, end).trim();
}

/**
 * Given a user query string, return the top-K function names by cosine
 * similarity, plus their scores.
 *
 * @param {string} query
 * @param {number} k
 * @returns {Promise<{topNames:string[], topScores:number[], allNames:string[]}>}
 */
export async function topKForQuery(query, k = 3) {
  const idx = await getOrBuildIndex();
  const encoder = await loadEncoder();
  const [qvec] = await embed(encoder, [query]);
  const top = cosineTopK(qvec, idx.vecs, k);
  return {
    topNames: top.map(t => idx.names[t.idx]),
    topScores: top.map(t => t.score),
    allNames: idx.names,
  };
}

/**
 * Rewrite the candidate list inside a prompt to a new list of names.
 *
 * The original list is the JSON block matched by extractCandidateNames in
 * grammar.js — either a simple ["a","b","c"] array or a list of full schema
 * objects. We replace both forms with a clean string array (preferred form),
 * preserving the surrounding "Use them if required -\n" marker and the
 * "\n\nUSER:" trailer.
 *
 * @param {string} prompt
 * @param {string[]} newNames
 * @returns {string} rewritten prompt
 */
export function rewriteCandidateList(prompt, newNames) {
  // Match the same block as extractCandidateNames does.
  const re = /Use them if required -\s*\n(\[[\s\S]*?)(?=\n\nUSER:|\nUSER:|\n\n)/;
  const m = prompt.match(re);
  if (!m) return prompt; // nothing to rewrite
  const block = m[1];
  // Build replacement
  const newBlock = `[\n  ${newNames.map(n => `"${n}"`).join(',\n  ')}\n]`;
  // Replace exactly the matched block (m[1]).
  const start = m.index + m[0].indexOf(block);
  const end = start + block.length;
  return prompt.slice(0, start) + newBlock + prompt.slice(end);
}
