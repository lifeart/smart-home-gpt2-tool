# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 20.1 — build sh_train_v9.json.

Sources:
  - lifeart/smart-home-sft-v2 / sh_train_v6r.json  (19084 refined rows)
  - acon96/Home-Assistant-Requests                 (~35.8k ShareGPT smart-home rows,
                                                    cap 5000 after parse)
  - nvidia/Nemotron-Post-Training-Dataset-v1, split=tool_calling
                                                   (310k rows, cap 5000 single-call)

Schema for every emitted row (matches v6r):
  prompt    = SYSTEM: ... functions ... USER: <q> ... ASSISTANT: <functioncall>
  gold      = '{"name":"X","arguments":{...}}'   (compact JSON)
  gold_name = "X"
  domain    = "external_ha" | "external_nemotron" | <existing v6r domain>

Granite multi-task relabel (Iter 20.1.e) is performed by training/relabel_granite.py
in a second pass against this concatenated file.

Output:
  data/sh_train_v9_base.json    (concat ~29k, pre-relabel)
  Push to lifeart/smart-home-sft-v2/sh_train_v9_base.json (optional during pilot)
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
OUT_PATH = ROOT / "data" / "sh_train_v9_base.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")

PROMPT_TMPL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n{spec}\n\n\nUSER: {query}\n\n\n"
    "ASSISTANT: <functioncall> "
)

CAP_HA = int(os.environ.get("CAP_HA", "5000"))
CAP_NEMOTRON = int(os.environ.get("CAP_NEMOTRON", "5000"))

PILOT = os.environ.get("PILOT", "0") == "1"

# Patterns for secrets that GitHub push-protection blocks. Drop any row containing one.
SECRET_RE = re.compile(
    r"gh[pousr]_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{40,}|AKIA[A-Z0-9]{12,}"
)


def extract_user_query(prompt: str) -> str:
    try:
        head = prompt.split("USER: ", 1)[1]
        return head.split("\n\n\nASSISTANT:")[0].strip()
    except Exception:  # noqa: BLE001
        return prompt[:120]


# -------------------- Home-Assistant-Requests adapter -------------------- #

HA_BLOCK_RE = re.compile(
    r"```homeassistant\s*(\{.*?\})\s*```",
    re.S,
)
# Service line e.g. "cover.close_cover()" or "light.turn_on(rgb_color,brightness)"
HA_SERVICE_LINE_RE = re.compile(r"([a-z_]+\.[a-z_]+)\(([^)]*)\)")


def parse_ha_services(system_text: str) -> list[dict] | None:
    """Extract the Services: line from HA system block and build a function spec.

    Each service `domain.action(p1,p2)` becomes a function named `domain.action`
    with the listed params as required strings.
    """
    idx = system_text.find("Services:")
    if idx < 0:
        return None
    tail = system_text[idx + len("Services:"):]
    # Stop at the "Devices:" marker (devices listing follows)
    end = tail.find("\nDevices:")
    if end >= 0:
        tail = tail[:end]
    funcs: dict[str, dict] = {}
    for m in HA_SERVICE_LINE_RE.finditer(tail):
        name = m.group(1).strip()
        params_str = m.group(2).strip()
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        if name in funcs:
            # Dedup: keep first appearance
            continue
        props = {p: {"type": "string"} for p in params}
        funcs[name] = {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": params,
            },
        }
    if not funcs:
        return None
    return list(funcs.values())


def parse_ha_call(assistant_text: str) -> tuple[str, dict] | None:
    """Extract the ```homeassistant {...}``` JSON block.

    The JSON has shape {"service": "cover.close", "target_device": "cover.kitchen", ...other_args}.
    We turn `service` into our `name` and the rest into `arguments`.
    """
    m = HA_BLOCK_RE.search(assistant_text)
    if not m:
        return None
    blob = m.group(1)
    try:
        obj = json.loads(blob)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("service")
    if not isinstance(name, str) or not name:
        return None
    args = {k: v for k, v in obj.items() if k != "service"}
    return name, args


def adapt_home_assistant(cap: int) -> list[dict]:
    print("[HA] loading acon96/Home-Assistant-Requests …")
    ds = load_dataset(
        "acon96/Home-Assistant-Requests", split="train", streaming=True
    )
    out: list[dict] = []
    skipped_no_sys = 0
    skipped_no_user = 0
    skipped_no_call = 0
    skipped_no_spec = 0
    skipped_unknown_name = 0
    for row in ds:
        convs = row.get("conversations") or []
        sys_msg = next((c for c in convs if c.get("from") == "system"), None)
        user_msg = next((c for c in convs if c.get("from") == "user"), None)
        asst_msg = next((c for c in convs if c.get("from") == "assistant"), None)
        if not sys_msg or not user_msg or not asst_msg:
            skipped_no_user += 1
            continue
        sys_text = sys_msg.get("value", "") or ""
        spec = parse_ha_services(sys_text)
        if not spec:
            skipped_no_spec += 1
            continue
        spec_names = {s["name"] for s in spec}
        call = parse_ha_call(asst_msg.get("value", "") or "")
        if not call:
            skipped_no_call += 1
            continue
        name, args = call
        # The HA block uses short service names ("cover.close") but Services: lists
        # full forms ("cover.close_cover"). Coerce: if not in spec_names, try with
        # "_cover"/"_open"/"_off"/"_on" common HA shorthands; or accept as-is.
        if name not in spec_names:
            # Try common alias mappings observed in the dataset
            alias_candidates = [
                name,
                name + "_cover",
                name + "_off",
                name + "_on",
                name + "_track",
                name + "_action",
                name + "_speed",
            ]
            matched = next(
                (a for a in alias_candidates if a in spec_names), None
            )
            if matched is None:
                # Reverse: check if any spec name starts with our `name` prefix
                pref = [n for n in spec_names if n.startswith(name + "_")]
                if len(pref) == 1:
                    matched = pref[0]
            if matched is None:
                skipped_unknown_name += 1
                continue
            name = matched
        spec_str = json.dumps(spec, indent=2)
        user_q = (user_msg.get("value") or "").strip()
        if not user_q:
            skipped_no_user += 1
            continue
        prompt = PROMPT_TMPL.format(spec=spec_str, query=user_q)
        gold = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out.append(
            {
                "prompt": prompt,
                "gold": gold,
                "gold_name": name,
                "domain": "external_ha",
            }
        )
        if len(out) >= cap:
            break
    print(
        f"[HA] adapted {len(out)} rows "
        f"(skipped: no_sys={skipped_no_sys} no_user={skipped_no_user} "
        f"no_spec={skipped_no_spec} no_call={skipped_no_call} "
        f"unknown_name={skipped_unknown_name})"
    )
    return out


# -------------------- Nemotron tool_calling adapter -------------------- #


def parse_nemotron_tools(metadata_str: str) -> list[dict] | None:
    """metadata is JSON: {"tools": [{"type":"function","function":{name,description,parameters}}, ...]}."""
    try:
        meta = json.loads(metadata_str)
    except Exception:  # noqa: BLE001
        return None
    tools = meta.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    out: list[dict] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters", {})
        if not name:
            continue
        out.append({"name": name, "parameters": params})
    return out if out else None


def adapt_nemotron(cap: int) -> list[dict]:
    print("[Nemotron] loading nvidia/Nemotron-Post-Training-Dataset-v1 (tool_calling) …")
    ds = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1",
        split="tool_calling",
        streaming=True,
    )
    out: list[dict] = []
    skipped_no_tools = 0
    skipped_no_user = 0
    skipped_no_call = 0
    skipped_multi_call = 0
    skipped_unknown_name = 0
    skipped_too_big = 0
    for row in ds:
        meta_str = row.get("metadata", "") or ""
        spec = parse_nemotron_tools(meta_str)
        if not spec:
            skipped_no_tools += 1
            continue
        spec_names = {s["name"] for s in spec}
        msgs = row.get("messages") or []
        # First user content
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        if not user_msg:
            skipped_no_user += 1
            continue
        user_q = (user_msg.get("content") or "").strip()
        # Find first assistant turn with tool_calls
        first_asst = next(
            (
                m
                for m in msgs
                if m.get("role") == "assistant" and m.get("tool_calls")
            ),
            None,
        )
        if not first_asst:
            skipped_no_call += 1
            continue
        tcs = first_asst.get("tool_calls") or []
        if len(tcs) > 1:
            skipped_multi_call += 1
            continue
        fn = tcs[0].get("function", {}) if tcs else {}
        name = fn.get("name")
        args_raw = fn.get("arguments")
        if not name:
            skipped_no_call += 1
            continue
        if name not in spec_names:
            skipped_unknown_name += 1
            continue
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:  # noqa: BLE001
                args = {}
        else:
            args = args_raw or {}
        # Slim spec to {name, parameters} — same as xLAM adapter shape
        slim_spec = [{"name": s["name"], "parameters": s.get("parameters", {})} for s in spec]
        spec_str = json.dumps(slim_spec, indent=2)
        prompt = PROMPT_TMPL.format(spec=spec_str, query=user_q)
        # Reject prompts that are too large after spec — keep it manageable for 1024-tok ctx.
        # The training tokenizer truncates to PAD-80 anyway, but emitting massive
        # specs is wasteful: cap raw prompt length to ~6000 chars (≈1500 tokens).
        if len(prompt) > 6000:
            skipped_too_big += 1
            continue
        gold = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out.append(
            {
                "prompt": prompt,
                "gold": gold,
                "gold_name": name,
                "domain": "external_nemotron",
            }
        )
        if len(out) >= cap:
            break
    print(
        f"[Nemotron] adapted {len(out)} rows "
        f"(skipped: no_tools={skipped_no_tools} no_user={skipped_no_user} "
        f"no_call={skipped_no_call} multi_call={skipped_multi_call} "
        f"unknown_name={skipped_unknown_name} too_big={skipped_too_big})"
    )
    return out


# -------------------- Merge -------------------- #


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


def load_v6r() -> list[dict]:
    local = ROOT / "data" / "sh_train_v6r.json"
    if local.exists():
        print(f"[v6r] using local {local}")
        with local.open() as f:
            return json.load(f)
    print(f"[v6r] fetching {DATA_REPO}/sh_train_v6r.json")
    p = hf_hub_download(DATA_REPO, "sh_train_v6r.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def main() -> None:
    random.seed(42)
    if PILOT:
        cap_ha, cap_nem = 50, 50
        print(f"[mode] PILOT — HA={cap_ha} Nemotron={cap_nem}")
    else:
        cap_ha, cap_nem = CAP_HA, CAP_NEMOTRON

    v6r = load_v6r()
    print(f"[in] v6r={len(v6r)}")
    v6r_counts = Counter(r.get("domain", "?") for r in v6r)
    print(f"[in] v6r top-10 domain counts: {dict(v6r_counts.most_common(10))}")

    seen: set[tuple[str, str]] = set()
    for r in v6r:
        q = extract_user_query(r["prompt"]).lower()[:200]
        seen.add((r.get("gold_name", ""), q))

    ha = adapt_home_assistant(cap_ha)
    nem = adapt_nemotron(cap_nem)

    if PILOT:
        print("\n=== HA samples ===")
        for r in ha[:3]:
            print("USER:", extract_user_query(r["prompt"])[:120])
            print("GOLD:", r["gold"][:200])
            print()
        print("\n=== Nemotron samples ===")
        for r in nem[:3]:
            print("USER:", extract_user_query(r["prompt"])[:120])
            print("GOLD:", r["gold"][:200])
            print()
        parse_rate_ha = len(ha) / cap_ha if cap_ha else 0
        parse_rate_nem = len(nem) / cap_nem if cap_nem else 0
        print(
            f"[pilot] HA parse rate ~{parse_rate_ha:.2%}, "
            f"Nemotron ~{parse_rate_nem:.2%}"
        )
        return

    ha_kept = dedupe_rows(ha, seen)
    nem_kept = dedupe_rows(nem, seen)
    print(
        f"[dedup] HA {len(ha)}→{len(ha_kept)}, "
        f"Nemotron {len(nem)}→{len(nem_kept)}"
    )

    merged = list(v6r) + ha_kept + nem_kept
    random.shuffle(merged)
    print(f"[out] total v9_base={len(merged)}")

    counts = Counter(r.get("domain", "?") for r in merged)
    print("\n=== v9_base by domain top 20 ===")
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
                path_in_repo="sh_train_v9_base.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter20.1: v9 base ({len(merged)} rows) — "
                    f"v6r + {len(ha_kept)} HA + {len(nem_kept)} Nemotron"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
