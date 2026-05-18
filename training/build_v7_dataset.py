# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 16.1 — build sh_train_v7.json = v6 base + ToolACE + 2k more xLAM + 500 Glaive.

Inputs (HF datasets):
  - lifeart/smart-home-sft-v2 / sh_train_v6.json  (20600 base rows)
  - Team-ACE/ToolACE / data.json                  (~11.3k multi-turn rows, take single-call → cap 3000)
  - minpeter/xlam-function-calling-60k-parsed     (60k single-call, take 2000 with stride/seed disjoint from v4)
  - glaiveai/glaive-function-calling-v2 / glaive-function-calling-v2.json (~113k chats, take 500 single-call)

Outputs:
  - data/sh_train_v7.json (~26000 rows)
  - push to lifeart/smart-home-sft-v2/sh_train_v7.json

Each new row:
  - prompt = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n<spec>\n\n\nUSER: <q>\n\n\nASSISTANT: <functioncall> "
  - gold   = {"name": <name>, "arguments": <args>}    (compact JSON)
  - gold_name, domain="external_<source>"

Dedup by (gold_name, user_query) lowercase 200 chars.
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "sh_train_v7.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")

PROMPT_TMPL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n{spec}\n\n\nUSER: {query}\n\n\n"
    "ASSISTANT: <functioncall> "
)

CAP_TOOLACE = 3000
CAP_XLAM = 2000
CAP_GLAIVE = 500
XLAM_V4_TAKEN = 3000          # v4 took the first 3000 SH-filtered matches
XLAM_SCAN_LIMIT = 60000       # full dataset
XLAM_SEED = 1729              # different from v4 (which used default order)
XLAM_STRIDE_OFFSET = 30000    # skip into second half so we don't overlap v4 SH-matched prefix


def extract_user_query(prompt: str) -> str:
    try:
        head = prompt.split("USER: ", 1)[1]
        return head.split("\n\n\nASSISTANT:")[0].strip()
    except Exception:  # noqa: BLE001
        return prompt[:120]


# -------------------- ToolACE -------------------- #

# Parses a single-call bracket invocation like:
#   [Market Trends API(trend_type="MARKET_INDEXES", country="us")]
# Returns (name, args) or None.
TOOLACE_NAME_RE = re.compile(
    r"^\s*\[\s*([^()\[\]]+?)\s*\((.*)\)\s*\]\s*$",
    re.S,
)


def _split_kv_args(arg_str: str) -> dict | None:
    """Split a bracket-call argument string into a dict.

    Strategy: find top-level `=` assignments at brace/bracket/quote depth 0.
    Values may be JSON literals (strings, numbers, true/false, lists, dicts).
    Returns dict on success, None on parse failure.
    """
    arg_str = arg_str.strip()
    if not arg_str:
        return {}
    # Find positions of `,` and `=` at depth 0
    depth = 0
    in_str = False
    str_ch = ""
    splits: list[int] = []
    eq_positions: list[int] = []
    i = 0
    while i < len(arg_str):
        c = arg_str[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == str_ch:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                str_ch = c
            elif c in "[{(":
                depth += 1
            elif c in "]})":
                depth -= 1
            elif depth == 0 and c == ",":
                splits.append(i)
            elif depth == 0 and c == "=":
                eq_positions.append(i)
        i += 1
    # Now split arg_str by top-level commas
    pieces: list[str] = []
    last = 0
    for s in splits:
        pieces.append(arg_str[last:s])
        last = s + 1
    pieces.append(arg_str[last:])
    out: dict = {}
    for p in pieces:
        # Find first top-level `=`
        depth = 0
        in_str = False
        str_ch = ""
        eq_idx = -1
        j = 0
        while j < len(p):
            c = p[j]
            if in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == str_ch:
                    in_str = False
            else:
                if c in ('"', "'"):
                    in_str = True
                    str_ch = c
                elif c in "[{(":
                    depth += 1
                elif c in "]})":
                    depth -= 1
                elif depth == 0 and c == "=":
                    eq_idx = j
                    break
            j += 1
        if eq_idx < 0:
            # Positional or junk — bail
            return None
        k = p[:eq_idx].strip()
        v_raw = p[eq_idx + 1 :].strip()
        # Drop trailing commas
        v_raw = v_raw.rstrip(",").strip()
        # Try JSON parse; if fails, try replacing single quotes with double and re-try; else keep as string
        v = None
        try:
            v = json.loads(v_raw)
        except Exception:  # noqa: BLE001
            try:
                v = json.loads(v_raw.replace("'", '"'))
            except Exception:  # noqa: BLE001
                # Strip surrounding quotes if present, otherwise keep as bare token
                if (
                    len(v_raw) >= 2
                    and v_raw[0] == v_raw[-1]
                    and v_raw[0] in ('"', "'")
                ):
                    v = v_raw[1:-1]
                else:
                    v = v_raw
        out[k] = v
    return out


def parse_toolace_call(assistant_value: str) -> tuple[str, dict] | None:
    """Parse a ToolACE assistant single-call line into (name, args).

    Returns None if not a single-call line.
    """
    m = TOOLACE_NAME_RE.match(assistant_value)
    if not m:
        return None
    name = m.group(1).strip()
    args_str = m.group(2)
    # Single-call: ensure no nested top-level call (count `(` at depth 0 in args matters less;
    # but the outer pattern already constrains shape). Still, reject if the rest contains
    # `, ` followed by another name-call at depth 0.
    args = _split_kv_args(args_str)
    if args is None:
        return None
    if not name or any(ch in name for ch in "[]"):
        return None
    return name, args


def adapt_toolace(cap: int) -> list[dict]:
    print(f"[ToolACE] loading Team-ACE/ToolACE/data.json …")
    p = hf_hub_download("Team-ACE/ToolACE", "data.json", repo_type="dataset")
    with open(p) as f:
        data = json.load(f)
    print(f"[ToolACE] {len(data)} multi-turn rows")
    rng = random.Random(11)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    out: list[dict] = []
    skipped_no_call = 0
    skipped_parse = 0
    skipped_no_funcs = 0
    for idx in indices:
        row = data[idx]
        system = row.get("system") or ""
        # Extract the JSON function list embedded inside the system prompt
        # ToolACE format: "...\nHere is a list of functions in JSON format that you can invoke:\n[{...}, {...}]"
        mark = "Here is a list of functions in JSON format that you can invoke:"
        if mark not in system:
            skipped_no_funcs += 1
            continue
        spec_raw = system.split(mark, 1)[1].strip()
        # Try parse as JSON array — the array may span until end of string
        spec = None
        # The array might be followed by trailing whitespace / "" — try progressively
        for cut in range(len(spec_raw), 0, -1):
            try:
                spec = json.loads(spec_raw[:cut])
                if isinstance(spec, list):
                    break
            except Exception:  # noqa: BLE001
                continue
        if not spec or not isinstance(spec, list):
            skipped_no_funcs += 1
            continue
        convs = row.get("conversations") or []
        # Walk convs; first user turn, then first assistant turn that looks like a single call
        user_q = None
        call = None
        for c in convs:
            if c.get("from") == "user" and user_q is None:
                user_q = (c.get("value") or "").strip()
            elif c.get("from") == "assistant" and user_q is not None:
                v = (c.get("value") or "").strip()
                if v.startswith("["):
                    parsed = parse_toolace_call(v)
                    if parsed is not None:
                        call = parsed
                        break
                # If assistant turn isn't a bracket call, skip the row (their first
                # assistant turn is text, e.g. clarifying question).
                break
        if user_q is None:
            skipped_no_call += 1
            continue
        if call is None:
            skipped_parse += 1
            continue
        name, args = call
        # Ensure the called name actually appears in the spec list (else hallucinated)
        spec_names = {s.get("name") for s in spec if isinstance(s, dict)}
        if name not in spec_names:
            skipped_parse += 1
            continue
        # Build prompt with full spec list (4-6 candidates is the right shape — most ToolACE
        # rows already have ~5 functions in their spec, we keep them as-is for diversity).
        spec_str = json.dumps(spec, indent=2)
        prompt = PROMPT_TMPL.format(spec=spec_str, query=user_q)
        gold = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out.append(
            {
                "prompt": prompt,
                "gold": gold,
                "gold_name": name,
                "domain": "external_toolace",
            }
        )
        if len(out) >= cap:
            break
    print(
        f"[ToolACE] adapted {len(out)} single-call rows "
        f"(skipped: no_funcs={skipped_no_funcs} parse={skipped_parse} "
        f"no_call={skipped_no_call})"
    )
    return out


# -------------------- xLAM (additional 2k) -------------------- #


def adapt_xlam_more(cap: int) -> list[dict]:
    """Take `cap` rows that v4 did NOT take.

    v4 used substring filter (smart-home keywords) and took first 3000 matches. Here we
    DROP the substring filter and instead sample from the deep end of the dataset
    (index ≥ XLAM_STRIDE_OFFSET) with a fresh RNG seed to maximize disjointness.
    """
    print("[xLAM+] loading minpeter/xlam-function-calling-60k-parsed …")
    ds = load_dataset(
        "minpeter/xlam-function-calling-60k-parsed", split="train"
    )
    n = len(ds)
    print(f"[xLAM+] {n} rows total; sampling from idx >= {XLAM_STRIDE_OFFSET}")
    rng = random.Random(XLAM_SEED)
    pool = list(range(XLAM_STRIDE_OFFSET, n))
    rng.shuffle(pool)
    out: list[dict] = []
    skipped_parse = 0
    skipped_multi = 0
    skipped_no_call = 0
    for idx in pool:
        row = ds[idx]
        msgs = row["messages"]
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:  # noqa: BLE001
                skipped_parse += 1
                continue
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        asst_msg = next(
            (m for m in msgs if m.get("role") == "assistant"), None
        )
        if not user_msg or not asst_msg:
            skipped_no_call += 1
            continue
        tool_calls = asst_msg.get("tool_calls") or []
        if not tool_calls:
            skipped_no_call += 1
            continue
        # Filter for single-call rows only (xLAM has some multi-call)
        if len(tool_calls) > 1:
            skipped_multi += 1
            continue
        first = tool_calls[0]
        fn = first.get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments")
        if not name:
            skipped_parse += 1
            continue
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:  # noqa: BLE001
                args = {}
        else:
            args = args_raw or {}
        # Parse tools spec — keep as list, slim each tool to {name, parameters} for brevity
        try:
            tools = (
                json.loads(row["tools"])
                if isinstance(row["tools"], str)
                else row["tools"]
            )
        except Exception:  # noqa: BLE001
            skipped_parse += 1
            continue
        spec_for_prompt: list[dict] = []
        for t in tools:
            fn_dict = t.get("function", t) if isinstance(t, dict) else None
            if isinstance(fn_dict, dict):
                spec_for_prompt.append(
                    {
                        "name": fn_dict.get("name"),
                        "parameters": fn_dict.get("parameters", {}),
                    }
                )
        if not spec_for_prompt:
            skipped_parse += 1
            continue
        spec_str = json.dumps(spec_for_prompt, indent=2)
        prompt = PROMPT_TMPL.format(
            spec=spec_str, query=user_msg.get("content", "").strip()
        )
        gold = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out.append(
            {
                "prompt": prompt,
                "gold": gold,
                "gold_name": name,
                "domain": "external_xlam_v7",
            }
        )
        if len(out) >= cap:
            break
    print(
        f"[xLAM+] adapted {len(out)} rows "
        f"(skipped: parse={skipped_parse} multi={skipped_multi} "
        f"no_call={skipped_no_call})"
    )
    return out


# -------------------- Glaive -------------------- #

GLAIVE_FUNCCALL_RE = re.compile(
    r"ASSISTANT:\s*<functioncall>\s*(\{.*?\})\s*<\|endoftext\|>",
    re.S,
)
GLAIVE_USER_RE = re.compile(
    r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", re.S,
)


def parse_glaive_call(call_blob: str) -> tuple[str, dict] | None:
    """Glaive embeds gold as `{"name":..., "arguments": '<inner json>'}` where
    `arguments` is itself a JSON string with single-quote wrapping.
    """
    # Glaive uses single-quoted JSON strings inside the outer dict. Use a tolerant
    # parser: find name via regex, args via the inner-quoted blob.
    m_name = re.search(r'"name"\s*:\s*"([^"]+)"', call_blob)
    if not m_name:
        return None
    name = m_name.group(1)
    m_args = re.search(r'"arguments"\s*:\s*[\'"](\{.*?\})[\'"]', call_blob, re.S)
    args: dict = {}
    if m_args:
        inner = m_args.group(1)
        try:
            args = json.loads(inner)
        except Exception:  # noqa: BLE001
            try:
                args = json.loads(inner.replace("'", '"'))
            except Exception:  # noqa: BLE001
                args = {}
    return name, args


def _extract_glaive_spec(system_text: str) -> list[dict] | None:
    """Glaive system block contains one or more function dicts separated by `\n\n`.

    Slice out the JSON dicts between the marker line and the end of the system text.
    """
    marker = "Use them if required -\n"
    if marker not in system_text:
        return None
    spec_raw = system_text.split(marker, 1)[1].strip()
    # Each top-level dict starts at `{` at column 0. Parse one-by-one.
    out: list[dict] = []
    i = 0
    n = len(spec_raw)
    while i < n:
        # Skip whitespace
        while i < n and spec_raw[i] in " \t\r\n":
            i += 1
        if i >= n or spec_raw[i] != "{":
            break
        # Find matching brace
        depth = 0
        j = i
        in_str = False
        str_ch = ""
        while j < n:
            c = spec_raw[j]
            if in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == str_ch:
                    in_str = False
            else:
                if c in ('"', "'"):
                    in_str = True
                    str_ch = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if j >= n:
            break
        blob = spec_raw[i : j + 1]
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:  # noqa: BLE001
            pass
        i = j + 1
    return out if out else None


def adapt_glaive(cap: int) -> list[dict]:
    print("[Glaive] loading glaiveai/glaive-function-calling-v2 …")
    p = hf_hub_download(
        "glaiveai/glaive-function-calling-v2",
        "glaive-function-calling-v2.json",
        repo_type="dataset",
    )
    with open(p) as f:
        data = json.load(f)
    print(f"[Glaive] {len(data)} rows total")
    rng = random.Random(31)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    out: list[dict] = []
    skipped_no_call = 0
    skipped_parse = 0
    skipped_no_spec = 0
    for idx in indices:
        row = data[idx]
        system_text = row.get("system") or ""
        chat = row.get("chat") or ""
        spec = _extract_glaive_spec(system_text)
        if not spec:
            skipped_no_spec += 1
            continue
        # First user query
        m_u = GLAIVE_USER_RE.search(chat)
        if not m_u:
            skipped_no_call += 1
            continue
        user_q = m_u.group(1).strip()
        # First functioncall
        m_c = GLAIVE_FUNCCALL_RE.search(chat)
        if not m_c:
            # Many Glaive rows are clarifying-question turns (no <functioncall>). Skip.
            skipped_no_call += 1
            continue
        call = parse_glaive_call(m_c.group(1))
        if not call:
            skipped_parse += 1
            continue
        name, args = call
        spec_names = {s.get("name") for s in spec if isinstance(s, dict)}
        if name not in spec_names:
            skipped_parse += 1
            continue
        spec_str = json.dumps(spec, indent=2)
        prompt = PROMPT_TMPL.format(spec=spec_str, query=user_q)
        gold = json.dumps(
            {"name": name, "arguments": args}, ensure_ascii=False
        )
        out.append(
            {
                "prompt": prompt,
                "gold": gold,
                "gold_name": name,
                "domain": "external_glaive",
            }
        )
        if len(out) >= cap:
            break
    print(
        f"[Glaive] adapted {len(out)} rows "
        f"(skipped: no_spec={skipped_no_spec} no_call={skipped_no_call} "
        f"parse={skipped_parse})"
    )
    return out


# -------------------- Merge -------------------- #


# Patterns for secrets that GitHub push-protection blocks. Drop any row containing one
# (ToolACE has at least one synthetic GitHub PAT in its prompts, etc.).
SECRET_RE = re.compile(
    r"gh[pousr]_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{40,}|AKIA[A-Z0-9]{12,}"
)


def dedupe_rows(rows: list[dict], seen: set[tuple[str, str]]) -> list[dict]:
    kept: list[dict] = []
    for r in rows:
        text = r.get("prompt", "") + " " + r.get("gold", "")
        if SECRET_RE.search(text):
            continue
        q = extract_user_query(r["prompt"]).lower()[:200]
        key = (r.get("gold_name", ""), q)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def load_v6() -> list[dict]:
    local = ROOT / "data" / "sh_train_v6.json"
    if local.exists():
        print(f"[v6] using local {local}")
        with local.open() as f:
            return json.load(f)
    print(f"[v6] fetching {DATA_REPO}/sh_train_v6.json")
    p = hf_hub_download(DATA_REPO, "sh_train_v6.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def main() -> None:
    random.seed(42)
    v6 = load_v6()
    print(f"[in] v6={len(v6)}")
    v6_counts = Counter(r.get("domain", "?") for r in v6)
    print(f"[in] v6 domain counts: {dict(v6_counts)}")

    # Seed the dedupe set with v6 keys
    seen: set[tuple[str, str]] = set()
    for r in v6:
        q = extract_user_query(r["prompt"]).lower()[:200]
        seen.add((r.get("gold_name", ""), q))

    # Build each new source
    toolace = adapt_toolace(CAP_TOOLACE)
    xlam_more = adapt_xlam_more(CAP_XLAM)
    glaive = adapt_glaive(CAP_GLAIVE)

    # Dedupe within each new source against v6 and each other
    toolace_kept = dedupe_rows(toolace, seen)
    xlam_kept = dedupe_rows(xlam_more, seen)
    glaive_kept = dedupe_rows(glaive, seen)
    print(
        f"[dedup] toolace {len(toolace)}→{len(toolace_kept)}, "
        f"xlam {len(xlam_more)}→{len(xlam_kept)}, "
        f"glaive {len(glaive)}→{len(glaive_kept)}"
    )

    merged = list(v6) + toolace_kept + xlam_kept + glaive_kept
    random.shuffle(merged)
    print(f"[out] total v7={len(merged)}")

    counts = Counter(r.get("domain", "?") for r in merged)
    print("\n=== final v7 dataset (by domain top 20) ===")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:<28} {v}")
    print(f"\n=== TOTAL {len(merged)} ===")

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_PATH),
                path_in_repo="sh_train_v7.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter16.1: v7 dataset ({len(merged)} rows) — "
                    f"v6 + {len(toolace_kept)} ToolACE + "
                    f"{len(xlam_kept)} xLAM + {len(glaive_kept)} Glaive"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
