"""Iter 23 — Python port of web/grammar.js.

Faithful port of the top-K-rerank JSON-schema constrained decoder so we can
measure constrained-mode exact-match in HF benches (same metric the browser
gate uses). Mirrors `isValidPrefix`, `parseTypedValue`, `buildSchemaConstraint`,
`extractCandidateNames`, and `extractPromptSchemas`.

Tested against grammar.js by checking matching reject/accept on hand-crafted
inputs in __main__.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------- prompt parsing ----------------

_CAND_BLOCK_RE = re.compile(
    r"Use them if required -\s*\n(\[[\s\S]*?)(?:\n\n|\nUSER:)"
)


def extract_candidate_names(prompt: str) -> list[str]:
    m = _CAND_BLOCK_RE.search(prompt)
    if not m:
        return []
    block = m.group(1)
    bs = block.strip()
    if bs.startswith('["') or bs.startswith("[\n  \""):
        try:
            arr = json.loads(block)
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, str)]
        except Exception:
            pass
    # Fallback: regex over all "name": "<x>" occurrences (handles truncated
    # objects).
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', block)
    # Dedupe preserving order
    seen = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def extract_prompt_schemas(prompt: str) -> dict:
    """Returns {fn_name: {keys: list, types: {key: {type, enum}}}}.

    Mirrors web/grammar.js extractPromptSchemas. Returns empty for string-only
    candidate arrays.
    """
    out: dict = {}
    m = _CAND_BLOCK_RE.search(prompt)
    if not m:
        return out
    block = m.group(1)

    # Iterate over "name": "<X>" matches; for each, find its enclosing
    # function-schema region and extract `properties` (preferred) or
    # `parameters` (legacy form).
    name_re = re.compile(r'"name"\s*:\s*"([^"]+)"')
    name_matches = list(name_re.finditer(block))
    for idx, mm in enumerate(name_matches):
        fn = mm.group(1)
        start = mm.end()
        end = name_matches[idx + 1].start() if idx + 1 < len(name_matches) else len(block)
        region = block[start:end]

        # First try "properties"
        prop_idx = region.find('"properties"')
        keys: list[str] = []
        types: dict = {}
        if prop_idx >= 0:
            obrace = region.find("{", prop_idx)
            if obrace >= 0:
                end_rel = _walk_balanced(region, obrace, "{", "}")
                props_block = region[obrace + 1:end_rel - 1] if end_rel > obrace else region[obrace + 1:]
                keys, types = _parse_properties_block(props_block)
        elif '"parameters"' in region:
            # Legacy form: "parameters": { "<key>": "<typeStr|with|enums>", ... }
            params_idx = region.find('"parameters"')
            obrace = region.find("{", params_idx)
            if obrace >= 0:
                end_rel = _walk_balanced(region, obrace, "{", "}")
                inner = region[obrace + 1:end_rel - 1] if end_rel > obrace else region[obrace + 1:]
                # Match "<key>": "<typeStr>" pairs at top level.
                kv_re = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
                for kvm in kv_re.finditer(inner):
                    k = kvm.group(1)
                    type_str = kvm.group(2).lower()
                    info: dict = {}
                    if "|" in type_str:
                        info["enum"] = [s.strip() for s in type_str.split("|") if s.strip()]
                        info["type"] = "string"
                    elif type_str in ("int", "integer"):
                        info["type"] = "integer"
                    elif type_str in ("number", "float"):
                        info["type"] = "number"
                    elif type_str in ("boolean", "bool"):
                        info["type"] = "boolean"
                    elif type_str in ("string", "str"):
                        info["type"] = "string"
                    elif type_str == "array":
                        info["type"] = "array"
                    else:
                        info["type"] = type_str or None
                    keys.append(k)
                    types[k] = info
        out[fn] = {"keys": keys, "types": types}
    return out


def _walk_balanced(s: str, start: int, open_c: str, close_c: str) -> int:
    """Returns index AFTER the matching close char, or len(s) if unbalanced."""
    depth = 0
    in_str = False
    j = start
    while j < len(s):
        c = s[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == '"':
                in_str = False
            j += 1
            continue
        if c == '"':
            in_str = True
            j += 1
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s)


def _parse_properties_block(props: str) -> tuple[list[str], dict]:
    """Parse a JSON-schema 'properties' inner block: `"key": { ... }, ...`."""
    keys: list[str] = []
    types: dict = {}
    k = 0
    n = len(props)
    while k < n:
        while k < n and props[k] in " \t\n\r,":
            k += 1
        if k >= n or props[k] != '"':
            break
        ks = k + 1
        ke = props.find('"', ks)
        if ke < 0:
            break
        key = props[ks:ke]
        k = ke + 1
        while k < n and props[k] in " \t\n\r":
            k += 1
        if k >= n or props[k] != ":":
            break
        k += 1
        while k < n and props[k] in " \t\n\r":
            k += 1
        if k >= n or props[k] != "{":
            break
        end_rel = _walk_balanced(props, k, "{", "}")
        val_block = props[k:end_rel]
        tm = re.search(r'"type"\s*:\s*"([^"]+)"', val_block)
        em = re.search(r'"enum"\s*:\s*\[([\s\S]*?)\]', val_block)
        decl_type = tm.group(1).lower() if tm else None
        enum_vals = None
        if em:
            enum_vals = re.findall(r'"([^"]*)"', em.group(1))
        keys.append(key)
        info: dict = {"type": decl_type}
        if enum_vals:
            info["enum"] = enum_vals
        types[key] = info
        k = end_rel
    return keys, types


# ---------------- constraint builder ----------------

@dataclass
class SchemaConstraint:
    names: list[str]
    param_keys: dict = field(default_factory=dict)
    param_types: dict = field(default_factory=dict)
    typed_args: bool = True


def build_schema_constraint(
    candidate_names: list[str],
    registry: dict,
    *,
    prompt_schemas: Optional[dict] = None,
    typed_args: bool = True,
    wide_names: bool = True,
) -> SchemaConstraint:
    if wide_names and registry:
        s = set(candidate_names)
        for n in registry.keys():
            s.add(n)
        names = list(s)
    else:
        names = list(candidate_names)

    param_keys: dict = {}
    param_types: dict = {}
    for n in names:
        keys: Optional[list[str]] = None
        types: dict = {}
        if prompt_schemas and n in prompt_schemas:
            p = prompt_schemas[n]
            keys = list(p["keys"])
            for k, info in p["types"].items():
                t = info.get("type")
                if t in ("int", "integer"):
                    t = "integer"
                elif t in ("number", "float"):
                    t = "number"
                elif t in ("boolean", "bool"):
                    t = "boolean"
                elif t == "string":
                    t = "string"
                elif t == "array":
                    t = "array"
                elif t == "object":
                    t = "object"
                types[k] = {"type": t, "enum": info.get("enum")}
        reg_entry = registry.get(n) if registry else None
        if reg_entry and reg_entry.get("params"):
            reg_keys = list(reg_entry["params"].keys())
            if not keys:
                keys = list(reg_keys)
            else:
                set_k = set(keys)
                for k in reg_keys:
                    if k not in set_k:
                        keys.append(k)
            for k in reg_keys:
                if k not in types:
                    rt = str(reg_entry["params"][k]).lower()
                    if rt in ("int", "integer"):
                        t = "integer"
                    elif rt in ("number", "float"):
                        t = "number"
                    elif rt in ("boolean", "bool"):
                        t = "boolean"
                    elif rt == "string":
                        t = "string"
                    else:
                        t = rt
                    types[k] = {"type": t, "enum": None}
        param_keys[n] = keys
        param_types[n] = types
    return SchemaConstraint(
        names=names, param_keys=param_keys, param_types=param_types, typed_args=typed_args
    )


# ---------------- prefix validator ----------------

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _match_lit_prefix(s: str, i: int, lit: str) -> bool:
    """Returns True iff s[i:i+len(lit)] is a prefix of lit (or equals it)."""
    tail = s[i:i + len(lit)]
    if len(tail) < len(lit):
        return lit.startswith(tail)
    return tail == lit


def _parse_string(s: str, i: int) -> int:
    """Index AFTER the closing quote, len(s)+1 if incomplete, -1 if invalid."""
    # Caller has verified s[i] == '"'
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return j + 1
        j += 1
    return len(s) + 1


def _parse_number(s: str, i: int) -> int:
    j = i
    if s[j] == "-":
        j += 1
    while j < len(s) and "0" <= s[j] <= "9":
        j += 1
    if j < len(s) and s[j] == ".":
        j += 1
        while j < len(s) and "0" <= s[j] <= "9":
            j += 1
    if j == len(s):
        return len(s) + 1
    return j


def _parse_literal(s: str, i: int, lit: str) -> int:
    tail = s[i:i + len(lit)]
    if len(tail) < len(lit):
        return len(s) + 1 if lit.startswith(tail) else -1
    return i + len(lit) if tail == lit else -1


def _parse_braced(s: str, i: int, open_c: str, close_c: str) -> int:
    depth = 0
    j = i
    in_str = False
    while j < len(s):
        c = s[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == '"':
                in_str = False
            j += 1
            continue
        if c == '"':
            in_str = True
            j += 1
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s) + 1


def _parse_value(s: str, i: int) -> int:
    if i >= len(s):
        return len(s) + 1
    c = s[i]
    if c == '"':
        return _parse_string(s, i)
    if c == "{":
        return _parse_braced(s, i, "{", "}")
    if c == "[":
        return _parse_braced(s, i, "[", "]")
    if c == "-" or "0" <= c <= "9":
        return _parse_number(s, i)
    if c == "t":
        return _parse_literal(s, i, "true")
    if c == "f":
        return _parse_literal(s, i, "false")
    if c == "n":
        return _parse_literal(s, i, "null")
    return -1


def _parse_typed_value(s: str, i: int, info: dict) -> int:
    if i >= len(s):
        return len(s) + 1
    c = s[i]
    t = info.get("type")
    enums = info.get("enum")
    if enums:
        if c != '"':
            return -1
        close = s.find('"', i + 1)
        if close < 0:
            partial = s[i + 1:]
            if any(e.lower().startswith(partial.lower()) for e in enums):
                return len(s) + 1
            return -1
        val = s[i + 1:close]
        if any(e.lower() == val.lower() for e in enums):
            return close + 1
        return -1
    if t in ("integer", "number"):
        if c == '"':
            return -1
        if c != "-" and not ("0" <= c <= "9"):
            return -1
        j = i
        if s[j] == "-":
            j += 1
        saw_digit = False
        while j < len(s) and "0" <= s[j] <= "9":
            j += 1
            saw_digit = True
        if j < len(s) and s[j] == ".":
            j += 1
            while j < len(s) and "0" <= s[j] <= "9":
                j += 1
                saw_digit = True
        if j == len(s):
            return len(s) + 1
        if not saw_digit:
            return -1
        return j
    if t == "boolean":
        if c == "t":
            return _parse_literal(s, i, "true")
        if c == "f":
            return _parse_literal(s, i, "false")
        return -1
    if t == "string":
        if c != '"':
            return -1
        return _parse_string(s, i)
    return _parse_value(s, i)


def is_valid_prefix(s: str, c: SchemaConstraint) -> bool:
    """Port of web/grammar.js isValidPrefix."""
    i = 0
    if i == len(s):
        return True
    if s[i] != "{":
        return False
    i += 1
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    name_lit = '"name"'
    if not _match_lit_prefix(s, i, name_lit):
        return name_lit.startswith(s[i:])
    if i + len(name_lit) > len(s):
        return True
    i += len(name_lit)
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    if s[i] != ":":
        return False
    i += 1
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    if s[i] != '"':
        return False
    i += 1
    name_start = i
    name_end = s.find('"', name_start)
    name_complete = name_end != -1
    partial_name = s[name_start:name_end if name_complete else len(s)]
    if not name_complete:
        return any(cand.startswith(partial_name) for cand in c.names)
    if partial_name not in c.names:
        return False
    i = name_end + 1
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    if s[i] != ",":
        return False
    i += 1
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    args_lit = '"arguments"'
    if not _match_lit_prefix(s, i, args_lit):
        return False
    if i + len(args_lit) > len(s):
        return True
    i += len(args_lit)
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    if s[i] != ":":
        return False
    i += 1
    i = _skip_ws(s, i)
    if i == len(s):
        return True
    if s[i] != "{":
        return False
    i += 1
    # 8. KV pairs
    allowed_keys = c.param_keys.get(partial_name)
    types_map = c.param_types.get(partial_name) if c.typed_args else None
    while True:
        i = _skip_ws(s, i)
        if i == len(s):
            return True
        if s[i] == "}":
            i += 1
            i = _skip_ws(s, i)
            if i == len(s):
                return True
            if s[i] != "}":
                return False
            i += 1
            return True
        if s[i] != '"':
            return False
        i += 1
        key_start = i
        key_end = s.find('"', key_start)
        key_complete = key_end != -1
        partial_key = s[key_start:key_end if key_complete else len(s)]
        if allowed_keys is not None:
            if not key_complete:
                if not any(k.startswith(partial_key) for k in allowed_keys):
                    return False
                return True
            if partial_key not in allowed_keys:
                return False
        else:
            if not key_complete:
                return True
        i = key_end + 1
        i = _skip_ws(s, i)
        if i == len(s):
            return True
        if s[i] != ":":
            return False
        i += 1
        i = _skip_ws(s, i)
        if i == len(s):
            return True
        key_info = types_map.get(partial_key) if (types_map is not None) else None
        if key_info:
            nxt = _parse_typed_value(s, i, key_info)
        else:
            nxt = _parse_value(s, i)
        if nxt == -1:
            return False
        if nxt > len(s):
            return True
        i = nxt
        i = _skip_ws(s, i)
        if i == len(s):
            return True
        if s[i] == ",":
            i += 1
            continue
        elif s[i] == "}":
            continue
        else:
            return False


# ---------------- constrained generation ----------------

def constrained_generate(
    model,
    tokenizer,
    prompt: str,
    constraint: SchemaConstraint,
    device,
    *,
    max_new: int = 96,
    top_k: int = 40,
    allow_special_tokens: Optional[set[int]] = None,
) -> str:
    """Greedy decode + top-K-rerank under JSON-schema prefix validator.

    Mirrors web/grammar.js JsonSchemaLogitsProcessor: at each step take the K
    highest-logit tokens, decode each, append, and keep only those that keep
    the running output a valid prefix per `is_valid_prefix`. Pick the highest
    valid token. Fallback to top-1 if none valid (to avoid stalling).
    """
    import torch

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    gen_ids: list[int] = []
    allow_special = allow_special_tokens or set()

    # Cache decoded text per token id (small set in practice).
    tok_text_cache: dict[int, str] = {}

    def decode_tok(t: int) -> str:
        x = tok_text_cache.get(t)
        if x is None:
            x = tokenizer.decode([t], skip_special_tokens=False)
            tok_text_cache[t] = x
        return x

    closed_pat = re.compile(r"\}\s*\}\s*\Z")

    with torch.no_grad():
        for _ in range(max_new):
            if cur.shape[1] >= 1024:
                break
            out = model(cur)
            logits = out.logits[0, -1, :]
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False) if gen_ids else ""
            # Short-circuit if we've already closed the outer JSON.
            if generated_text and closed_pat.search(generated_text.rstrip()):
                break

            K = min(top_k, logits.shape[0])
            vals, idxs = torch.topk(logits, K)
            chosen = -1
            for rank in range(K):
                tok = int(idxs[rank].item())
                if tok in allow_special:
                    chosen = tok
                    break
                tok_str = decode_tok(tok)
                if not tok_str:
                    chosen = tok
                    break
                if is_valid_prefix(generated_text + tok_str, constraint):
                    chosen = tok
                    break
            if chosen < 0:
                chosen = int(idxs[0].item())  # fallback to top-1

            gen_ids.append(chosen)
            cur = torch.cat(
                [cur, torch.tensor([[chosen]], dtype=torch.long, device=device)],
                dim=1,
            )
            # Early stop: outer JSON closed (one extra step buffer for trailing ws).
            gtxt = tokenizer.decode(gen_ids, skip_special_tokens=False)
            if closed_pat.search(gtxt.rstrip()):
                break

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------- self-test ----------------

if __name__ == "__main__":
    # Lightweight check: known-good and known-bad strings under a simple
    # constraint. Mirrors what web/grammar.js would accept/reject.
    reg = {
        "set_light": {"params": {"room": "string", "level": "integer"}, "required": []},
        "lock_door": {"params": {"door": "string"}, "required": []},
    }
    c = build_schema_constraint(["set_light", "lock_door"], reg, wide_names=False)
    # Accepts
    assert is_valid_prefix('', c)
    assert is_valid_prefix('{', c)
    assert is_valid_prefix('{"name":"', c)
    assert is_valid_prefix('{"name":"set_light"', c)
    assert is_valid_prefix('{"name":"set_light","arguments":{}}', c)
    assert is_valid_prefix('{"name":"set_light","arguments":{"room":"bedroom"', c)
    assert is_valid_prefix('{"name":"set_light","arguments":{"room":"bedroom","level":50}}', c)
    # Rejects
    assert not is_valid_prefix('{"name":"unknown_fn"', c), "wrong name"
    assert not is_valid_prefix('{"name":"set_light","arguments":{"bad_key":"x"', c), "bad key"
    # Wrong typed value (integer key set to string)
    assert not is_valid_prefix('{"name":"set_light","arguments":{"level":"x"', c), "string for int"
    print("OK: grammar.py self-test pass")
