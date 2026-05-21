# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 23 — H1.2 constrained bench, self-contained for HF Jobs.

Bundles grammar.py + bench_common.py + bench_h1_con.py into one file so it
runs on `hf jobs uv run --flavor t4-small`.

Reads:
  - lifeart/smart-home-sft-v2 / sh_test.json    (test set, n=300)
  - lifeart/smart-home-sft-v2 / tool_registry.json
  - lifeart/smart-home-gpt2-v6                  (name model)
  - lifeart/smart-home-gpt2-v9                  (args model)

Writes:
  - bench_results.json                          (local)
  - lifeart/smart-home-sft-v2 / iter23_h1_con_results.json  (HF, best-effort)

Run locally:
    python training/bench_h1_con_cloud.py
Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        training/bench_h1_con_cloud.py
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ==================== scoring (port of web/bench.js) ====================

def _normalize_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t and re.fullmatch(r"-?\d+(\.\d+)?", t):
            try:
                return float(t) if "." in t else int(t)
            except ValueError:
                return t.lower()
        return t.lower()
    return v


def args_match(pred: Any, gold: Any) -> bool:
    a = pred if isinstance(pred, dict) else {}
    b = gold if isinstance(gold, dict) else {}
    ak = sorted(a.keys())
    bk = sorted(b.keys())
    if ak != bk:
        return False
    for k in ak:
        av, bv = a[k], b[k]
        if isinstance(bv, list):
            if not isinstance(av, list) or len(av) != len(bv):
                return False
            try:
                as_ = sorted(_normalize_scalar(x) for x in av)
                bs_ = sorted(_normalize_scalar(x) for x in bv)
                if as_ != bs_:
                    return False
            except TypeError:
                if json.dumps(av, sort_keys=True) != json.dumps(bv, sort_keys=True):
                    return False
        elif isinstance(bv, dict):
            if not isinstance(av, dict) or not args_match(av, bv):
                return False
        else:
            an, bn = _normalize_scalar(av), _normalize_scalar(bv)
            if an is None and bn is None:
                continue
            if an is None or bn is None:
                return False
            if isinstance(an, (int, float)) and isinstance(bn, (int, float)):
                if an != bn:
                    return False
            else:
                if str(an) != str(bn):
                    return False
    return True


def _extract_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_call(text: str) -> Optional[dict]:
    js = _extract_json(text)
    if not js:
        return None
    try:
        obj = json.loads(js)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "arguments": args}


def parse_args_only(text: str) -> Optional[dict]:
    js = _extract_json(text)
    if not js:
        return None
    try:
        obj = json.loads(js)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "arguments" in obj and isinstance(obj["arguments"], dict):
        return obj["arguments"]
    if "arguments" not in obj and "name" not in obj:
        return obj
    return None


def parse_gold(gold_str: str) -> dict:
    try:
        obj = json.loads(gold_str)
    except Exception:
        return {"name": None, "arguments": {}}
    if not isinstance(obj, dict):
        return {"name": None, "arguments": {}}
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "arguments": args}


def score(pred_name: Optional[str], pred_args: Any, gold_str: str) -> dict:
    gold = parse_gold(gold_str)
    name_ok = pred_name == gold["name"] and gold["name"] is not None
    a_ok = args_match(pred_args, gold["arguments"])
    return {"name_ok": name_ok, "args_ok": a_ok, "exact_ok": name_ok and a_ok}


def aggregate(rows: list[dict], prefix: str) -> dict:
    p = prefix + "_"
    n = len(rows)
    if not n:
        return {"n": 0}
    by_domain: dict[str, dict] = {}
    for r in rows:
        d = r.get("domain", "?")
        b = by_domain.setdefault(d, {"n": 0, "name": 0, "args": 0, "exact": 0})
        b["n"] += 1
        if r.get(p + "name_ok"):
            b["name"] += 1
        if r.get(p + "args_ok"):
            b["args"] += 1
        if r.get(p + "exact_ok"):
            b["exact"] += 1
    return {
        "n": n,
        "name_acc": sum(1 for r in rows if r.get(p + "name_ok")) / n,
        "args_acc": sum(1 for r in rows if r.get(p + "args_ok")) / n,
        "exact_acc": sum(1 for r in rows if r.get(p + "exact_ok")) / n,
        "by_domain": by_domain,
    }


def print_summary(label: str, s: dict) -> None:
    if not s.get("n"):
        print(f"  {label}: n=0")
        return
    print(
        f"  {label}: name={s['name_acc']*100:5.1f}% "
        f"args={s['args_acc']*100:5.1f}% "
        f"exact={s['exact_acc']*100:5.1f}% (n={s['n']})"
    )
    for d, b in sorted(s.get("by_domain", {}).items()):
        print(
            f"    {d:<10} name={b['name']/b['n']*100:5.1f}% "
            f"args={b['args']/b['n']*100:5.1f}% "
            f"exact={b['exact']/b['n']*100:5.1f}% "
            f"({b['exact']}/{b['n']})"
        )


ARGS_HINT_TMPL = "Note: The function name will be: {name}. Output the arguments only.\n\n\n"


def build_args_only_prompt(prompt: str, predicted_name: str) -> str:
    marker = "\n\n\nASSISTANT:"
    hint = ARGS_HINT_TMPL.format(name=predicted_name)
    if marker in prompt:
        head, tail = prompt.split(marker, 1)
        return head + "\n" + hint + "ASSISTANT:" + tail
    return prompt + " " + hint


# ==================== grammar (port of web/grammar.js) ====================

_CAND_BLOCK_RE = re.compile(r"Use them if required -\s*\n(\[[\s\S]*?)(?:\n\n|\nUSER:)")


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
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', block)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _walk_balanced(s: str, start: int, open_c: str, close_c: str) -> int:
    depth, in_str, j = 0, False, start
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
    keys: list[str] = []
    types: dict = {}
    k, n = 0, len(props)
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
        enum_vals = re.findall(r'"([^"]*)"', em.group(1)) if em else None
        keys.append(key)
        info: dict = {"type": decl_type}
        if enum_vals:
            info["enum"] = enum_vals
        types[key] = info
        k = end_rel
    return keys, types


def extract_prompt_schemas(prompt: str) -> dict:
    out: dict = {}
    m = _CAND_BLOCK_RE.search(prompt)
    if not m:
        return out
    block = m.group(1)
    name_re = re.compile(r'"name"\s*:\s*"([^"]+)"')
    name_matches = list(name_re.finditer(block))
    for idx, mm in enumerate(name_matches):
        fn = mm.group(1)
        start = mm.end()
        end = name_matches[idx + 1].start() if idx + 1 < len(name_matches) else len(block)
        region = block[start:end]
        prop_idx = region.find('"properties"')
        keys, types = [], {}
        if prop_idx >= 0:
            obrace = region.find("{", prop_idx)
            if obrace >= 0:
                end_rel = _walk_balanced(region, obrace, "{", "}")
                props_block = region[obrace + 1:end_rel - 1] if end_rel > obrace else region[obrace + 1:]
                keys, types = _parse_properties_block(props_block)
        elif '"parameters"' in region:
            params_idx = region.find('"parameters"')
            obrace = region.find("{", params_idx)
            if obrace >= 0:
                end_rel = _walk_balanced(region, obrace, "{", "}")
                inner = region[obrace + 1:end_rel - 1] if end_rel > obrace else region[obrace + 1:]
                kv_re = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
                for kvm in kv_re.finditer(inner):
                    k_ = kvm.group(1)
                    type_str = kvm.group(2).lower()
                    info: dict = {}
                    if "|" in type_str:
                        info["enum"] = [t.strip() for t in type_str.split("|") if t.strip()]
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
                    keys.append(k_)
                    types[k_] = info
        out[fn] = {"keys": keys, "types": types}
    return out


@dataclass
class SchemaConstraint:
    names: list[str]
    param_keys: dict = field(default_factory=dict)
    param_types: dict = field(default_factory=dict)
    typed_args: bool = True


def build_schema_constraint(candidate_names, registry, *, prompt_schemas=None, typed_args=True, wide_names=True):
    if wide_names and registry:
        s = set(candidate_names)
        for n in registry.keys():
            s.add(n)
        names = list(s)
    else:
        names = list(candidate_names)
    param_keys, param_types = {}, {}
    for n in names:
        keys, types = None, {}
        if prompt_schemas and n in prompt_schemas:
            p = prompt_schemas[n]
            keys = list(p["keys"])
            for k_, info in p["types"].items():
                t = info.get("type")
                if t in ("int", "integer"):
                    t = "integer"
                elif t in ("number", "float"):
                    t = "number"
                elif t in ("boolean", "bool"):
                    t = "boolean"
                types[k_] = {"type": t, "enum": info.get("enum")}
        reg_entry = registry.get(n) if registry else None
        if reg_entry and reg_entry.get("params"):
            reg_keys = list(reg_entry["params"].keys())
            if not keys:
                keys = list(reg_keys)
            else:
                sk = set(keys)
                for k_ in reg_keys:
                    if k_ not in sk:
                        keys.append(k_)
            for k_ in reg_keys:
                if k_ not in types:
                    rt = str(reg_entry["params"][k_]).lower()
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
                    types[k_] = {"type": t, "enum": None}
        param_keys[n] = keys
        param_types[n] = types
    return SchemaConstraint(names=names, param_keys=param_keys, param_types=param_types, typed_args=typed_args)


def _skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _match_lit_prefix(s, i, lit):
    tail = s[i:i + len(lit)]
    if len(tail) < len(lit):
        return lit.startswith(tail)
    return tail == lit


def _parse_string(s, i):
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


def _parse_number(s, i):
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


def _parse_literal(s, i, lit):
    tail = s[i:i + len(lit)]
    if len(tail) < len(lit):
        return len(s) + 1 if lit.startswith(tail) else -1
    return i + len(lit) if tail == lit else -1


def _parse_braced(s, i, open_c, close_c):
    depth, j, in_str = 0, i, False
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


def _parse_value(s, i):
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


def _parse_typed_value(s, i, info):
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
        saw = False
        while j < len(s) and "0" <= s[j] <= "9":
            j += 1
            saw = True
        if j < len(s) and s[j] == ".":
            j += 1
            while j < len(s) and "0" <= s[j] <= "9":
                j += 1
                saw = True
        if j == len(s):
            return len(s) + 1
        if not saw:
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


# ==================== generation ====================

def constrained_generate(model, tokenizer, prompt, constraint, device, *, max_new=96, top_k=40):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    gen_ids: list[int] = []
    tok_text_cache: dict[int, str] = {}

    def decode_tok(t):
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
            generated = tokenizer.decode(gen_ids, skip_special_tokens=False) if gen_ids else ""
            if generated and closed_pat.search(generated.rstrip()):
                break
            K = min(top_k, logits.shape[0])
            vals, idxs = torch.topk(logits, K)
            chosen = -1
            for rank in range(K):
                tok = int(idxs[rank].item())
                tok_str = decode_tok(tok)
                if not tok_str:
                    chosen = tok
                    break
                if is_valid_prefix(generated + tok_str, constraint):
                    chosen = tok
                    break
            if chosen < 0:
                chosen = int(idxs[0].item())
            gen_ids.append(chosen)
            cur = torch.cat([cur, torch.tensor([[chosen]], dtype=torch.long, device=device)], dim=1)
            gtxt = tokenizer.decode(gen_ids, skip_special_tokens=False)
            if closed_pat.search(gtxt.rstrip()):
                break
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def greedy_unconstrained(model, tokenizer, prompt, device, max_new=96):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    brace_depth = 0
    started = False
    L = cur.shape[1]
    for _ in range(max_new):
        if cur.shape[1] >= 1024:
            break
        out = model(cur)
        logits = out.logits[0, -1, :]
        nxt = int(logits.argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        tok_str = tokenizer.decode([nxt])
        for c in tok_str:
            if c == "{":
                brace_depth += 1
                started = True
            elif c == "}":
                brace_depth -= 1
        if started and brace_depth <= 0:
            break
        if nxt == newline and not started:
            break
    new_ids = cur[0, L:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def detect_clean_domain(candidate_names: list[str]) -> bool:
    return any("vacuum" in n for n in candidate_names)


# ==================== driver ====================

NAME_MODEL = os.environ.get("NAME_MODEL", "lifeart/smart-home-gpt2-v6")
ARGS_MODEL = os.environ.get("ARGS_MODEL", "lifeart/smart-home-gpt2-v9")
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = full sh_test


def fetch_dataset_json(filename: str) -> Any:
    p = hf_hub_download(DATA_REPO, filename, repo_type="dataset")
    return json.loads(Path(p).read_text())


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[cuda] {torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    test_file = os.environ.get("TEST_FILE", "sh_test.json")
    print(f"[data] {test_file} + tool_registry.json from {DATA_REPO}")
    test = fetch_dataset_json(test_file)
    registry = fetch_dataset_json("tool_registry.json")
    if LIMIT and LIMIT < len(test):
        test = test[:LIMIT]
    print(f"[test] {len(test)} items, {len(registry)} registry entries")

    print(f"[load] {NAME_MODEL}")
    name_tok = GPT2TokenizerFast.from_pretrained(NAME_MODEL)
    name_tok.pad_token = name_tok.eos_token
    name_model = GPT2LMHeadModel.from_pretrained(NAME_MODEL).to(device).eval()
    if ARGS_MODEL == NAME_MODEL:
        args_tok, args_model = name_tok, name_model
    else:
        print(f"[load] {ARGS_MODEL}")
        args_tok = GPT2TokenizerFast.from_pretrained(ARGS_MODEL)
        args_tok.pad_token = args_tok.eos_token
        args_model = GPT2LMHeadModel.from_pretrained(ARGS_MODEL).to(device).eval()

    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(test):
        prompt = s["prompt"]
        domain_label = s.get("domain", "?")
        cand_names = extract_candidate_names(prompt)
        prompt_schemas = extract_prompt_schemas(prompt)
        c = build_schema_constraint(
            cand_names, registry,
            prompt_schemas=prompt_schemas,
            typed_args=True, wide_names=True,
        )
        is_clean = detect_clean_domain(cand_names)

        text_b = constrained_generate(name_model, name_tok, prompt, c, device)
        call_b = parse_call(text_b)
        sb = score(
            call_b["name"] if call_b else None,
            call_b["arguments"] if call_b else {},
            s["gold"],
        )
        pred_name = call_b["name"] if call_b else None
        if pred_name:
            args_prompt = build_args_only_prompt(prompt, pred_name)
            text_h1 = greedy_unconstrained(args_model, args_tok, args_prompt, device)
            pred_args = parse_args_only(text_h1)
            if pred_args is None:
                call2 = parse_call(text_h1)
                pred_args = call2["arguments"] if call2 else {}
        else:
            pred_args = {}
        sh1 = score(pred_name, pred_args, s["gold"])

        if is_clean and pred_name:
            h12_args = call_b["arguments"] if call_b else {}
        else:
            h12_args = pred_args
        sh12 = score(pred_name, h12_args, s["gold"])

        rows.append({
            "i": i, "domain": domain_label, "gold": s["gold"],
            "is_clean_detected": is_clean,
            "base_pred_name": pred_name,
            "base_pred_args": call_b["arguments"] if call_b else {},
            "base_name_ok": sb["name_ok"], "base_args_ok": sb["args_ok"],
            "base_exact_ok": sb["exact_ok"],
            "h1_pred_args": pred_args,
            "h1_name_ok": sh1["name_ok"], "h1_args_ok": sh1["args_ok"],
            "h1_exact_ok": sh1["exact_ok"],
            "h12_pred_args": h12_args,
            "h12_name_ok": sh12["name_ok"], "h12_args_ok": sh12["args_ok"],
            "h12_exact_ok": sh12["exact_ok"],
        })

        if (i + 1) % 25 == 0 or i + 1 == len(test):
            b = sum(1 for r in rows if r["base_exact_ok"])
            h = sum(1 for r in rows if r["h1_exact_ok"])
            g = sum(1 for r in rows if r["h12_exact_ok"])
            print(
                f"  [{i+1}/{len(test)}] base_con={b/(i+1)*100:.1f}% "
                f"H1_con={h/(i+1)*100:.1f}% H1.2_con={g/(i+1)*100:.1f}% "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    sB = aggregate(rows, "base")
    sH1 = aggregate(rows, "h1")
    sH12 = aggregate(rows, "h12")
    print(f"\n=== Iter 23 H1.2 constrained ({NAME_MODEL.split('/')[-1]} + {ARGS_MODEL.split('/')[-1]}) ===")
    print(f"  Test: {len(test)} items, elapsed {time.time()-t0:.0f}s, device={device}")
    print()
    print_summary("baseline_con (v6 constrained one-shot)", sB)
    print()
    print_summary("H1_con (v6-con name + v9 args)", sH1)
    print()
    print_summary("H1.2_con (clean→base, else→H1)", sH12)

    label_clean = sum(1 for r in rows if r["domain"] == "clean")
    detected_clean = sum(1 for r in rows if r["is_clean_detected"])
    correct = sum(1 for r in rows if r["is_clean_detected"] and r["domain"] == "clean")
    if label_clean:
        print(
            f"\n  clean detection: precision="
            f"{correct/max(detected_clean,1)*100:.0f}%  "
            f"recall={correct/label_clean*100:.0f}%"
        )

    out = {
        "name_model": NAME_MODEL, "args_model": ARGS_MODEL,
        "n": len(test), "elapsed_s": time.time() - t0,
        "baseline_con_summary": sB,
        "h1_con_summary": sH1,
        "h12_con_summary": sH12,
        "rows": rows,
    }
    out_path = Path("bench_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=os.environ.get("RESULT_FILE", "iter23_h1_con_results.json"),
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message="H1.2 constrained candidates (v6+v9)",
            )
            print(f"[push] -> {DATA_REPO}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
