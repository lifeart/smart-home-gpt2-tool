// JSON-schema constrained decoder for smart-home GPT-2.
//
// Approach: a top-K logits reranker that uses a small JSON prefix-validator
// (string-level, not token-level). At each generation step:
//   1. Decode the full generated text from token IDs.
//   2. Take the top-K candidate tokens from the model.
//   3. For each candidate, append its decoded text to the running string and
//      check whether that new string is a valid PREFIX of *some* legal JSON
//      completion (per the schema cursor). Pick the highest-scoring valid token.
//
// This is simpler than a true token-level vocab mask (which would require
// pre-computing per-state allowed token sets), and works well because:
//   - Smart-home replies are short (~30 tokens)
//   - The hot path runs <0.5ms overhead per step at K=40
//   - The model is already trained to produce JSON, so the top tokens are
//     almost always valid; the mask only kicks in to fix occasional drift.
//
// Schema cursor state machine (string-level prefix validator):
//   START      -> expect "{"
//   AFTER_LB   -> expect "\"name\": \""
//   IN_NAME    -> expect one of the allowed function names (prefix-match)
//   AFTER_NAME -> expect "\", \"arguments\": {"
//   IN_ARGS    -> inside the arguments object; sub-states for keys/values
//   DONE       -> expect "}"  then EOS / </functioncall>
//
// We don't enforce numeric vs string value types; that's a future tightening.
// We DO enforce that argument keys are drawn from the function's declared
// `properties` (when the registry has a schema for that function).

import { LogitsProcessor } from '@huggingface/transformers';

// ---------- helpers ----------

/**
 * Extract the candidate function names from the SYSTEM prompt.
 * The prompt's tool list is either:
 *   - a JSON array of strings: ["foo", "bar", "baz"]
 *   - a (possibly truncated) JSON array of full schema objects with "name": "foo"
 *
 * @param {string} prompt
 * @returns {string[]}
 */
export function extractCandidateNames(prompt) {
  const m = prompt.match(/Use them if required -\s*\n(\[[\s\S]*?)(?:\n\n|\nUSER:)/);
  if (!m) return [];
  const block = m[1];
  // Simple string array?
  if (/^\[\s*"/.test(block.trim())) {
    try {
      return JSON.parse(block);
    } catch {
      // fall through to regex
    }
  }
  // Object schemas — even if truncated mid-object, find all top-level "name": "..."
  const names = [];
  const re = /"name"\s*:\s*"([^"]+)"/g;
  let mm;
  while ((mm = re.exec(block)) !== null) names.push(mm[1]);
  return Array.from(new Set(names));
}

/**
 * Extract per-function param schemas (type + enum list) from the SYSTEM
 * prompt's tool list. The prompt may carry richer info than the registry
 * (enums in particular). Returns `Map<fnName, { keys, types: Map<key, {type, enum?}> }>`.
 *
 * When the tool list is a simple string array (no object schemas), returns
 * an empty map — the caller falls back to the registry.
 *
 * Parsing strategy: capture each `{"name": "<n>", ..., "properties": { ... }, ...}`
 * block (the prompt is often truncated mid-object so we use bounded depth
 * tracking rather than JSON.parse on the whole block).
 *
 * @param {string} prompt
 * @returns {Map<string, {keys: string[], types: Map<string, {type: string, enum?: string[]}>}>}
 */
export function extractPromptSchemas(prompt) {
  const out = new Map();
  const m = prompt.match(/Use them if required -\s*\n(\[[\s\S]*?)(?:\n\n|\nUSER:)/);
  if (!m) return out;
  const block = m[1];
  // Find each "name": "<x>" occurrence; for each, find its enclosing object
  // and the nested "properties" map.
  const nameRe = /"name"\s*:\s*"([^"]+)"/g;
  let mm;
  while ((mm = nameRe.exec(block)) !== null) {
    const fnName = mm[1];
    // Find the "properties" sub-block that belongs to this function definition.
    // Heuristic: search forward from current position for the first "properties":
    // section, bounded by the next top-level fn name (or end).
    const startSearch = mm.index + mm[0].length;
    const nextName = nameRe.exec(block);
    const endSearch = nextName ? nextName.index : block.length;
    nameRe.lastIndex = mm.index + 1; // rewind so next iteration can find nextName
    if (nextName) nameRe.lastIndex = nextName.index;
    const region = block.slice(startSearch, endSearch);
    const propIdx = region.indexOf('"properties"');
    if (propIdx === -1) {
      // Try the simpler form: "parameters": { "key": "type-string|with|enums", ... }
      const paramsIdx = region.indexOf('"parameters"');
      if (paramsIdx !== -1) {
        // Find opening '{' of the parameters object
        const obraceRel = region.indexOf('{', paramsIdx);
        if (obraceRel !== -1) {
          // Walk balanced braces (string-aware) to find end of params obj
          let dpth = 0, isInStr = false, endRel = region.length;
          for (let j = obraceRel; j < region.length; j++) {
            const c = region[j];
            if (isInStr) {
              if (c === '\\') { j++; continue; }
              if (c === '"') isInStr = false;
              continue;
            }
            if (c === '"') { isInStr = true; continue; }
            if (c === '{') dpth++;
            else if (c === '}') {
              dpth--;
              if (dpth === 0) { endRel = j + 1; break; }
            }
          }
          const inner = region.slice(obraceRel + 1, endRel - 1);
          // Match "<key>": "<typeStr>"  (skip nested objects)
          const kvRe = /"([^"]+)"\s*:\s*"([^"]*)"/g;
          const types2 = new Map();
          const keys2 = [];
          let kvm;
          while ((kvm = kvRe.exec(inner)) !== null) {
            const key = kvm[1];
            const typeStr = kvm[2].toLowerCase();
            let t = null, enums = null;
            if (typeStr.includes('|')) {
              // alternation = enum
              enums = typeStr.split('|').map(s => s.trim()).filter(s => s);
              t = 'string';
            } else if (typeStr === 'int' || typeStr === 'integer') {
              t = 'integer';
            } else if (typeStr === 'number' || typeStr === 'float') {
              t = 'number';
            } else if (typeStr === 'boolean' || typeStr === 'bool') {
              t = 'boolean';
            } else if (typeStr === 'string' || typeStr === 'str') {
              t = 'string';
            } else if (typeStr === 'array') {
              t = 'array';
            } else {
              t = typeStr || null;
            }
            const info = { type: t };
            if (enums) info.enum = enums;
            keys2.push(key);
            types2.set(key, info);
          }
          out.set(fnName, { keys: keys2, types: types2 });
          continue;
        }
      }
      out.set(fnName, { keys: [], types: new Map() });
      continue;
    }
    // From "properties" find the opening `{` then walk braces with string-aware depth.
    const obraceRel = region.indexOf('{', propIdx);
    if (obraceRel === -1) { out.set(fnName, { keys: [], types: new Map() }); continue; }
    // Walk to matching `}` (or until region ends — prompts truncate mid-schema).
    let depth = 0;
    let inStr = false;
    let endRel = region.length;
    for (let j = obraceRel; j < region.length; j++) {
      const c = region[j];
      if (inStr) {
        if (c === '\\') { j++; continue; }
        if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') { inStr = true; continue; }
      if (c === '{') depth++;
      else if (c === '}') {
        depth--;
        if (depth === 0) { endRel = j + 1; break; }
      }
    }
    const propsBlock = region.slice(obraceRel + 1, endRel - 1);
    // For each "<key>": { ... } at the top level of propsBlock, parse the
    // value object's "type" and "enum".
    const types = new Map();
    const keys = [];
    // Scan top-level keys.
    let k = 0;
    while (k < propsBlock.length) {
      // skip whitespace/commas
      while (k < propsBlock.length && /[\s,]/.test(propsBlock[k])) k++;
      if (k >= propsBlock.length) break;
      if (propsBlock[k] !== '"') break;
      const ks = k + 1;
      const ke = propsBlock.indexOf('"', ks);
      if (ke === -1) break;
      const key = propsBlock.slice(ks, ke);
      k = ke + 1;
      // skip ws + ':'
      while (k < propsBlock.length && /\s/.test(propsBlock[k])) k++;
      if (propsBlock[k] !== ':') break;
      k++;
      while (k < propsBlock.length && /\s/.test(propsBlock[k])) k++;
      // expect '{'
      if (propsBlock[k] !== '{') break;
      // Walk balanced { ... }
      const objStart = k;
      depth = 0; inStr = false;
      let objEnd = propsBlock.length;
      for (let j = k; j < propsBlock.length; j++) {
        const c = propsBlock[j];
        if (inStr) {
          if (c === '\\') { j++; continue; }
          if (c === '"') inStr = false;
          continue;
        }
        if (c === '"') { inStr = true; continue; }
        if (c === '{') depth++;
        else if (c === '}') {
          depth--;
          if (depth === 0) { objEnd = j + 1; break; }
        }
      }
      const valBlock = propsBlock.slice(objStart, objEnd);
      // Pull out "type" and "enum"
      const tm = valBlock.match(/"type"\s*:\s*"([^"]+)"/);
      const em = valBlock.match(/"enum"\s*:\s*\[([\s\S]*?)\]/);
      const declType = tm ? tm[1].toLowerCase() : null;
      let enumVals = null;
      if (em) {
        // pull quoted strings out of the enum list
        const eRe = /"([^"]*)"/g;
        enumVals = [];
        let em2;
        while ((em2 = eRe.exec(em[1])) !== null) enumVals.push(em2[1]);
      }
      keys.push(key);
      const info = { type: declType };
      if (enumVals && enumVals.length) info.enum = enumVals;
      types.set(key, info);
      k = objEnd;
    }
    out.set(fnName, { keys, types });
  }
  return out;
}

/**
 * Build the per-query schema constraint from the candidate name list +
 * the global tool registry + (optional) prompt-derived schema.
 *
 * The constraint also carries per-key type info (when known) so the value
 * masker can refuse type-incompatible values (e.g. quote-opening when key
 * expects a number).
 *
 * @param {string[]} candidateNames
 * @param {Record<string, {params: Record<string,string>, required: string[]}>} registry
 * @param {{ promptSchemas?: Map<string, {keys: string[], types: Map<string, {type, enum?}>}>,
 *           typedArgs?: boolean }} [options]
 * @returns {{names, paramKeys: Map<string,string[]>, paramTypes: Map<string, Map<string, {type, enum?}>>, typedArgs: boolean }}
 */
export function buildSchemaConstraint(candidateNames, registry, options = {}) {
  const promptSchemas = options.promptSchemas || null;
  const typedArgs = options.typedArgs !== false; // default ON
  const paramKeys = new Map();
  const paramTypes = new Map();
  for (const n of candidateNames) {
    // Prefer prompt-derived schema (richer: includes enums); fall back to registry.
    let keys = null;
    const typesMap = new Map();
    if (promptSchemas && promptSchemas.has(n)) {
      const p = promptSchemas.get(n);
      keys = p.keys.slice();
      for (const [k, info] of p.types) {
        // Normalize type name
        let t = info.type;
        if (t === 'integer' || t === 'int') t = 'integer';
        else if (t === 'number' || t === 'float') t = 'number';
        else if (t === 'boolean' || t === 'bool') t = 'boolean';
        else if (t === 'string') t = 'string';
        else if (t === 'array') t = 'array';
        else if (t === 'object') t = 'object';
        else t = t || null;
        typesMap.set(k, { type: t, enum: info.enum || null });
      }
    }
    const regEntry = registry ? registry[n] : null;
    if ((!keys || keys.length === 0) && regEntry && regEntry.params) {
      // Use registry as a fallback for keys + types (no enum info).
      keys = Object.keys(regEntry.params);
      for (const k of keys) {
        const rt = String(regEntry.params[k]).toLowerCase();
        let t;
        if (rt === 'integer' || rt === 'int') t = 'integer';
        else if (rt === 'number' || rt === 'float') t = 'number';
        else if (rt === 'boolean' || rt === 'bool') t = 'boolean';
        else if (rt === 'string') t = 'string';
        else t = rt;
        if (!typesMap.has(k)) typesMap.set(k, { type: t, enum: null });
      }
    } else if (regEntry && regEntry.params) {
      // Merge: keep typesMap (enum from prompt) but fill registry-only types if missing.
      for (const k of Object.keys(regEntry.params)) {
        if (!typesMap.has(k)) {
          const rt = String(regEntry.params[k]).toLowerCase();
          let t;
          if (rt === 'integer' || rt === 'int') t = 'integer';
          else if (rt === 'number' || rt === 'float') t = 'number';
          else if (rt === 'boolean' || rt === 'bool') t = 'boolean';
          else if (rt === 'string') t = 'string';
          else t = rt;
          typesMap.set(k, { type: t, enum: null });
        }
      }
    }
    paramKeys.set(n, keys);
    paramTypes.set(n, typesMap);
  }
  return { names: candidateNames, paramKeys, paramTypes, typedArgs };
}

// ---------- prefix validator ----------

/**
 * Check whether the string `s` is a valid PREFIX of some legal JSON output
 * under the schema. Returns true iff there exists at least one continuation
 * that satisfies the grammar.
 *
 * Grammar (concrete bytes the model must produce):
 *   {"name": "<NAME>", "arguments": {<KV_PAIRS>}}
 * where:
 *   <NAME> is one of constraint.names
 *   <KV_PAIRS> is "" | "\"<KEY>\": <VALUE>" | "\"<KEY>\": <VALUE>, <KV_PAIRS>"
 *   <KEY> is one of paramKeys[name]  (or any string if paramKeys[name] is null)
 *   <VALUE> is any JSON value (string / number / bool / array / object)
 *
 * We're lenient on inner whitespace and value content; strict on structural
 * tokens and identifier characters (name & key).
 *
 * @param {string} s
 * @param {{names: string[], paramKeys: Map<string,string[]>}} constraint
 * @returns {boolean}
 */
export function isValidPrefix(s, constraint) {
  // Fast walk through the structural skeleton.
  let i = 0;
  // 1. Opening "{"
  if (i === s.length) return true;
  if (s[i] !== '{') return false;
  i++;
  // 2. Optional whitespace
  i = skipWs(s, i);
  if (i === s.length) return true;
  // 3. "\"name\""
  if (!matchLiteralPrefix(s, i, '"name"')) {
    return startsWithPrefix('"name"', s.slice(i));
  }
  // If s ends before the full literal:
  if (i + '"name"'.length > s.length) return true;
  i += '"name"'.length;
  // 4. ":"
  i = skipWs(s, i);
  if (i === s.length) return true;
  if (s[i] !== ':') return false;
  i++;
  i = skipWs(s, i);
  if (i === s.length) return true;
  // 5. opening quote of name value
  if (s[i] !== '"') return false;
  i++;
  // 6. Name body — must be a prefix of one of constraint.names
  const nameStart = i;
  // Find closing quote (no escapes inside our names)
  let nameEnd = s.indexOf('"', nameStart);
  let nameComplete = nameEnd !== -1;
  const partialName = s.slice(nameStart, nameComplete ? nameEnd : s.length);
  // Check there's at least one candidate that starts with partialName
  let matchingName = null;
  for (const cand of constraint.names) {
    if (cand.startsWith(partialName) || (nameComplete && cand === partialName)) {
      if (!nameComplete || cand === partialName) {
        matchingName = cand;
        if (nameComplete && cand === partialName) break;
      }
    }
  }
  if (!nameComplete) {
    // Need at least one prefix-matching candidate.
    return constraint.names.some(c => c.startsWith(partialName));
  }
  // Name complete; must match exactly.
  if (!constraint.names.includes(partialName)) return false;
  i = nameEnd + 1;
  // 7. ", \"arguments\": {"
  i = skipWs(s, i);
  if (i === s.length) return true;
  if (s[i] !== ',') return false;
  i++;
  i = skipWs(s, i);
  if (i === s.length) return true;
  const argsLit = '"arguments"';
  if (!matchLiteralPrefix(s, i, argsLit)) return false;
  if (i + argsLit.length > s.length) return true;
  i += argsLit.length;
  i = skipWs(s, i);
  if (i === s.length) return true;
  if (s[i] !== ':') return false;
  i++;
  i = skipWs(s, i);
  if (i === s.length) return true;
  if (s[i] !== '{') return false;
  i++;
  // 8. Inside arguments object. Allowed keys for this function:
  const allowedKeys = constraint.paramKeys.get(partialName);
  const typesMap = constraint.paramTypes ? constraint.paramTypes.get(partialName) : null;
  const typedArgs = !!constraint.typedArgs;
  // Walk KV pairs.
  while (true) {
    i = skipWs(s, i);
    if (i === s.length) return true;
    if (s[i] === '}') {
      // End of args
      i++;
      i = skipWs(s, i);
      if (i === s.length) return true;
      if (s[i] !== '}') return false;
      i++;
      // Anything after the final "}" is ok (e.g. trailing whitespace, EOS-text,
      // or "</functioncall>" or another newline — the stopping criterion takes over).
      return true;
    }
    if (s[i] !== '"') return false;
    i++;
    const keyStart = i;
    const keyEnd = s.indexOf('"', keyStart);
    const keyComplete = keyEnd !== -1;
    const partialKey = s.slice(keyStart, keyComplete ? keyEnd : s.length);
    if (allowedKeys !== null && allowedKeys !== undefined) {
      if (!keyComplete) {
        if (!allowedKeys.some(k => k.startsWith(partialKey))) return false;
        return true; // need more chars
      }
      if (!allowedKeys.includes(partialKey)) return false;
    } else {
      // Free-form: any string ok
      if (!keyComplete) return true;
    }
    i = keyEnd + 1;
    i = skipWs(s, i);
    if (i === s.length) return true;
    if (s[i] !== ':') return false;
    i++;
    i = skipWs(s, i);
    if (i === s.length) return true;

    // Typed-args masking: select per-key value validator if enabled and known.
    const keyInfo = typedArgs && typesMap ? typesMap.get(partialKey) : null;
    let next;
    if (keyInfo) {
      next = parseTypedValue(s, i, keyInfo);
    } else {
      next = parseValue(s, i);
    }
    if (next === -1) return false;
    if (next > s.length) return true;
    i = next;
    i = skipWs(s, i);
    if (i === s.length) return true;
    if (s[i] === ',') {
      i++;
      continue;
    } else if (s[i] === '}') {
      continue; // outer loop handles
    } else {
      return false;
    }
  }
}

/**
 * Parse a JSON value with a known type+enum constraint. Returns the same
 * three-way result as `parseValue`: index-after, -1 if invalid, s.length+1
 * if incomplete.
 *
 * @param {string} s
 * @param {number} i
 * @param {{type: string, enum?: string[]}} info
 */
function parseTypedValue(s, i, info) {
  if (i >= s.length) return s.length + 1;
  const c = s[i];
  const t = info.type;
  const enums = info.enum;
  // Enum dominates type — value must be a string matching one of the enums.
  if (enums && enums.length) {
    if (c !== '"') return -1;
    // Find closing quote.
    const close = s.indexOf('"', i + 1);
    if (close === -1) {
      // Streaming: partial. Check that the partial value is a prefix of some enum.
      const partial = s.slice(i + 1);
      // Case-insensitive prefix match (model may emit casing variants).
      if (enums.some(e => e.toLowerCase().startsWith(partial.toLowerCase()))) {
        return s.length + 1;
      }
      return -1;
    }
    const val = s.slice(i + 1, close);
    if (enums.some(e => e.toLowerCase() === val.toLowerCase())) return close + 1;
    return -1;
  }
  // Numeric types — no quote allowed; only -?digits(.digits)?
  if (t === 'integer' || t === 'number') {
    if (c === '"') return -1;
    if (c !== '-' && !(c >= '0' && c <= '9')) return -1;
    let j = i;
    if (s[j] === '-') j++;
    let sawDigit = false;
    while (j < s.length && s[j] >= '0' && s[j] <= '9') { j++; sawDigit = true; }
    if (j < s.length && s[j] === '.') {
      if (t === 'integer') {
        // integer should not have a decimal; tolerate for now (training data
        // sometimes has 24.4 for a "number" key).
      }
      j++;
      while (j < s.length && s[j] >= '0' && s[j] <= '9') { j++; sawDigit = true; }
    }
    if (j === s.length) {
      // Still streaming; allow continuation only if we have at least one digit
      // or only the leading '-'.
      return s.length + 1;
    }
    if (!sawDigit) return -1;
    return j;
  }
  if (t === 'boolean') {
    if (c === 't') return parseLiteralValue(s, i, 'true');
    if (c === 'f') return parseLiteralValue(s, i, 'false');
    return -1;
  }
  if (t === 'string') {
    if (c !== '"') return -1;
    return parseString(s, i);
  }
  // Unknown type — fall back to free-form.
  return parseValue(s, i);
}

function skipWs(s, i) {
  while (i < s.length && (s[i] === ' ' || s[i] === '\t' || s[i] === '\n' || s[i] === '\r')) i++;
  return i;
}

function matchLiteralPrefix(s, i, lit) {
  // Check that s[i..] either equals lit or is a prefix of lit (we're streaming).
  const tail = s.slice(i, i + lit.length);
  if (tail.length < lit.length) return lit.startsWith(tail);
  return tail === lit;
}

function startsWithPrefix(target, candidate) {
  // candidate is the unread tail of s; we need candidate to be a prefix of target
  return target.startsWith(candidate);
}

/**
 * Parse a JSON value starting at index i. Returns:
 *  - the index just after the value, on success
 *  - -1 on malformed
 *  - s.length + 1 if the value is incomplete (need more chars)
 */
function parseValue(s, i) {
  if (i >= s.length) return s.length + 1;
  const c = s[i];
  if (c === '"') return parseString(s, i);
  if (c === '{') return parseBraced(s, i, '{', '}');
  if (c === '[') return parseBraced(s, i, '[', ']');
  if (c === '-' || (c >= '0' && c <= '9')) return parseNumber(s, i);
  if (c === 't') return parseLiteralValue(s, i, 'true');
  if (c === 'f') return parseLiteralValue(s, i, 'false');
  if (c === 'n') return parseLiteralValue(s, i, 'null');
  return -1;
}

function parseString(s, i) {
  // s[i] === '"'
  let j = i + 1;
  while (j < s.length) {
    const c = s[j];
    if (c === '\\') {
      j += 2;
      continue;
    }
    if (c === '"') return j + 1;
    j++;
  }
  return s.length + 1; // incomplete
}

function parseNumber(s, i) {
  let j = i;
  if (s[j] === '-') j++;
  while (j < s.length && s[j] >= '0' && s[j] <= '9') j++;
  if (j < s.length && s[j] === '.') {
    j++;
    while (j < s.length && s[j] >= '0' && s[j] <= '9') j++;
  }
  // Number is always "possibly complete" — caller checks following structural char.
  // If we hit end-of-string while still scanning digits, treat as incomplete.
  if (j === s.length) return s.length + 1;
  return j;
}

function parseLiteralValue(s, i, lit) {
  const tail = s.slice(i, i + lit.length);
  if (tail.length < lit.length) {
    return lit.startsWith(tail) ? s.length + 1 : -1;
  }
  return tail === lit ? i + lit.length : -1;
}

function parseBraced(s, i, open, close) {
  // Simple depth-tracking; respects strings.
  let depth = 0;
  let j = i;
  let inStr = false;
  while (j < s.length) {
    const c = s[j];
    if (inStr) {
      if (c === '\\') {
        j += 2;
        continue;
      }
      if (c === '"') inStr = false;
      j++;
      continue;
    }
    if (c === '"') {
      inStr = true;
      j++;
      continue;
    }
    if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) return j + 1;
    }
    j++;
  }
  return s.length + 1;
}

// ---------- LogitsProcessor ----------

/**
 * Top-K rerank constrained decoder.
 *
 * @param {Object} opts
 * @param {any} opts.tokenizer        - the @huggingface/transformers tokenizer
 * @param {number} opts.promptLength  - number of input_ids in the prompt
 * @param {{names: string[], paramKeys: Map<string,string[]>}} opts.constraint
 * @param {number} [opts.topK=40]     - candidates considered per step
 * @param {Set<number>} [opts.allowAlwaysTokens] - token IDs always allowed (e.g. EOS)
 */
export class JsonSchemaLogitsProcessor extends LogitsProcessor {
  constructor({ tokenizer, promptLength, constraint, topK = 40, allowAlwaysTokens = null }) {
    super();
    this.tokenizer = tokenizer;
    this.promptLength = promptLength;
    this.constraint = constraint;
    this.topK = topK;
    this.allowAlwaysTokens = allowAlwaysTokens || new Set();
    // Per-step cache of decoded token text (token id -> text).
    this._tokenTextCache = new Map();
    this.stats = { steps: 0, totalMs: 0, masked: 0 };
  }

  _decodeToken(id) {
    let t = this._tokenTextCache.get(id);
    if (t === undefined) {
      t = this.tokenizer.decode([id], { skip_special_tokens: false });
      this._tokenTextCache.set(id, t);
    }
    return t;
  }

  _call(input_ids, logits) {
    const t0 = performance.now();
    // logits shape: [batch, vocab]
    const dims = logits.dims;
    const batch = dims[0];
    const vocab = dims[dims.length - 1];
    const data = logits.data; // Float32Array

    for (let b = 0; b < batch; b++) {
      // Generated tokens so far (excluding prompt)
      const seq = input_ids[b];
      const genIds = [];
      for (let k = this.promptLength; k < seq.length; k++) {
        const v = seq[k];
        genIds.push(typeof v === 'bigint' ? Number(v) : v);
      }
      const generatedText = genIds.length
        ? this.tokenizer.decode(genIds, { skip_special_tokens: false })
        : '';

      // Quick exit: if we've already passed the closing "}}" and produced
      // some trailing content, allow anything (model will hit EOS / max).
      const closed = /\}\s*\}\s*[\s\S]*$/.test(generatedText) &&
                     generatedText.replace(/\s+$/, '').endsWith('}');
      if (closed) {
        this.stats.steps++;
        this.stats.totalMs += performance.now() - t0;
        continue;
      }

      const base = b * vocab;
      // Find top-K indices by logit value (partial argsort).
      const K = Math.min(this.topK, vocab);
      const topIdxs = topKIndices(data, base, vocab, K);

      // Score each candidate token: append, check prefix validity.
      const allowed = new Set();
      for (const tok of topIdxs) {
        if (this.allowAlwaysTokens.has(tok)) {
          allowed.add(tok);
          continue;
        }
        const tokText = this._decodeToken(tok);
        if (tokText.length === 0) {
          // BOS/EOS-like — let it through; stopping criterion handles it.
          allowed.add(tok);
          continue;
        }
        if (isValidPrefix(generatedText + tokText, this.constraint)) {
          allowed.add(tok);
        }
      }

      if (allowed.size === 0) {
        // Fallback: keep the original top token to avoid stalling.
        allowed.add(topIdxs[0]);
      }

      // Mask everything not in `allowed`.
      let masked = 0;
      for (let v = 0; v < vocab; v++) {
        if (!allowed.has(v)) {
          data[base + v] = -Infinity;
          masked++;
        }
      }
      this.stats.masked += masked;
    }

    this.stats.steps++;
    this.stats.totalMs += performance.now() - t0;
    return logits;
  }
}

function topKIndices(data, base, vocab, K) {
  // Simple O(vocab * log K) partial sort using a min-heap of size K.
  const heap = []; // pairs of [val, idx]; root is min
  const push = (val, idx) => {
    heap.push([val, idx]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] > heap[i][0]) {
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      } else break;
    }
  };
  const replaceRoot = (val, idx) => {
    heap[0] = [val, idx];
    let i = 0;
    while (true) {
      const l = 2 * i + 1, r = 2 * i + 2;
      let smallest = i;
      if (l < heap.length && heap[l][0] < heap[smallest][0]) smallest = l;
      if (r < heap.length && heap[r][0] < heap[smallest][0]) smallest = r;
      if (smallest !== i) {
        [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
        i = smallest;
      } else break;
    }
  };
  for (let v = 0; v < vocab; v++) {
    const val = data[base + v];
    if (heap.length < K) push(val, v);
    else if (val > heap[0][0]) replaceRoot(val, v);
  }
  return heap.map(([, idx]) => idx).sort((a, b) => data[base + b] - data[base + a]);
}
