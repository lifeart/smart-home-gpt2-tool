# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 26.1 — H1.5_con: improved 3-way picker (CoT + schema-aware).

Iter 25 H1.4_con achieved 70.3% but picker missed 20 items where oracle3
had a correct answer. Diagnosis (see PLAN.md Iter 26): heavy A-bias (15/20
misses chose A when correct was B or C). Misses correlate with small arg
differences the picker didn't analyze (extra args, wrong enum values,
string-formatting drift like `living room` vs `living_room`).

This version:
  1. Drops the "prefer A on tie" instruction — instead asks the picker
     to ignore order and decide on merit only.
  2. Adds chain-of-thought reasoning: first identifies required args from
     the schema, then evaluates each option for extra / missing / wrong.
  3. Includes the function schema (from `tool_registry.json`) for the
     candidate's name in the picker prompt.

Reads:
  /tmp/iter23_h1p4_results.json  (has base, H1, llama_direct preds + golds)

Writes:
  /tmp/iter26_h1p5_results.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

from bench_common import aggregate, load_hf_token, load_test, parse_call, print_summary, score
from bench_h2_rerank import parse_user_and_candidates


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "tool_registry.json"


PICKER_SYSTEM_V2 = (
    "You evaluate competing tool-call options for a smart-home assistant. "
    "Your job is to decide which option BEST matches the user's intent and "
    "the function schema. There is no preference for option order — judge "
    "each on merit. Reason step by step: first identify what the user "
    "explicitly said, then identify the function's required arguments, "
    "then for each option check (a) is the function name correct? (b) are "
    "all required arguments present? (c) are any extra arguments invented "
    "that the user did NOT mention? (d) do enum/string/number values match "
    "what the user said? Only after this analysis, choose the option that "
    "introduces the fewest errors. Reply in the EXACT format requested."
)


def _registry_schema_str(reg: dict, fn_name: Optional[str]) -> str:
    if not fn_name or fn_name not in reg:
        return "(no schema available)"
    entry = reg[fn_name]
    params = entry.get("params") or {}
    required = entry.get("required") or []
    enums = entry.get("enums") or {}
    if not params:
        return "{} (no arguments)"
    lines = []
    for k, t in params.items():
        marker = " *REQUIRED*" if k in required else " (optional)"
        if k in enums and isinstance(enums[k], list) and enums[k]:
            sample = enums[k][:6]
            tail = f" enum~[{', '.join(sample)}{'...' if len(enums[k]) > 6 else ''}]"
        else:
            tail = ""
        lines.append(f"    {k}: {t}{marker}{tail}")
    return "{\n" + "\n".join(lines) + "\n  }"


def build_picker_msg_v2(user_query, cand_names, options, registry):
    """options: list of (label, call_dict). call_dict may be None."""
    parts = [
        f'User said: "{user_query}"',
        f"Candidate function list (model must pick from these): {json.dumps(cand_names)}",
        "",
    ]
    for label, call in options:
        if not call or not call.get("name"):
            parts.append(f"Option {label}: (no valid output)\n")
            continue
        name = call["name"]
        args = call.get("arguments") or {}
        schema = _registry_schema_str(registry, name)
        parts.append(
            f"Option {label}:\n"
            f"  proposed function: {name}\n"
            f"  proposed arguments: {json.dumps(args, separators=(',', ':'))}\n"
            f"  declared schema for {name}: {schema}"
        )
        parts.append("")
    parts.append(
        "Reason briefly (3-5 short lines) about which option introduces "
        "the fewest errors against the user query and the schema. Then output:\n\n"
        f"PICK: {'|'.join(label for label, _ in options)}\n"
        "REASON: <your reasoning, one paragraph>"
    )
    return "\n".join(parts)


PICK_RE = re.compile(r"PICK\s*:\s*([A-Z])", re.IGNORECASE)


def llama_pick_v2(token, msg, allowed):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PICKER_SYSTEM_V2},
            {"role": "user", "content": msg},
        ],
        "temperature": 0.0,
        "max_tokens": 350,
    }
    for attempt in range(3):
        try:
            r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=90)
        except requests.exceptions.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 200:
            try:
                txt = r.json()["choices"][0]["message"]["content"]
            except Exception:
                return None
            m = PICK_RE.search(txt or "")
            if m and m.group(1).upper() in allowed:
                return m.group(1).upper()
            return None
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/iter23_h1p4_results.json")
    ap.add_argument("--out", default="/tmp/iter26_h1p5_results.json")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN missing")
        return
    registry = json.loads(REGISTRY_PATH.read_text())
    print(f"[reg] {len(registry)} functions")

    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    print(f"[in] {len(rows)} rows")

    # For each row, rerun the picker with v2 prompt.
    # Skip rows where all three options are identical.
    work = []
    for r in rows:
        a = (r["base_pred_name"], json.dumps(r["base_pred_args"], sort_keys=True))
        b = (r["base_pred_name"], json.dumps(r["h1_pred_args"], sort_keys=True))
        c_ = r.get("llama_direct")
        c = ((c_ or {}).get("name"), json.dumps((c_ or {}).get("arguments", {}), sort_keys=True))
        s = {a, b, c}
        if len(s) == 1:
            r["llama_pick3_v2"] = "A"
            continue
        work.append(r)
    print(f"[work] {len(work)} rows need re-pick")

    def pick(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["llama_pick3_v2"] = "A"
            return r
        a_call = {"name": r["base_pred_name"], "arguments": r["base_pred_args"]}
        b_call = {"name": r["base_pred_name"], "arguments": r["h1_pred_args"]}
        c_call = r.get("llama_direct")
        msg = build_picker_msg_v2(
            user_query, cand_names,
            [("A", a_call), ("B", b_call), ("C", c_call)],
            registry,
        )
        p = llama_pick_v2(token, msg, {"A", "B", "C"})
        r["llama_pick3_v2"] = p or "A"
        return r

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(pick, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 30 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    for r in rows:
        p = r.get("llama_pick3_v2", "A")
        if p == "A":
            r["h1p5_name_ok"] = r["base_name_ok"]
            r["h1p5_args_ok"] = r["base_args_ok"]
            r["h1p5_exact_ok"] = r["base_exact_ok"]
        elif p == "B":
            r["h1p5_name_ok"] = r["h1_name_ok"]
            r["h1p5_args_ok"] = r["h1_args_ok"]
            r["h1p5_exact_ok"] = r["h1_exact_ok"]
        else:
            r["h1p5_name_ok"] = r["llama_direct_name_ok"]
            r["h1p5_args_ok"] = r["llama_direct_args_ok"]
            r["h1p5_exact_ok"] = r["llama_direct_exact_ok"]
        r["oracle3_exact_ok"] = (
            r["base_exact_ok"] or r["h1_exact_ok"] or r["llama_direct_exact_ok"]
        )

    sH4 = aggregate(rows, "h1p4")
    sH5 = aggregate(rows, "h1p5")
    n = len(rows)
    oracle = sum(1 for r in rows if r["oracle3_exact_ok"]) / n
    a_c = sum(1 for r in rows if r.get("llama_pick3_v2") == "A")
    b_c = sum(1 for r in rows if r.get("llama_pick3_v2") == "B")
    c_c = sum(1 for r in rows if r.get("llama_pick3_v2") == "C")

    print(f"\n=== Iter 26.1 H1.5_con: improved 3-way picker (CoT+schema) ===")
    print()
    print_summary("H1.4_con (old picker)", sH4)
    print()
    print_summary("H1.5_con (CoT+schema picker)", sH5)
    print(f"\n  oracle3: {oracle*100:.2f}%")
    print(f"  picks (v2): A={a_c}  B={b_c}  C={c_c}")

    # Picker accuracy comparison on disagreement-where-correct rows
    win = [r for r in rows if r["oracle3_exact_ok"] and not r["base_exact_ok"] or
           (r["oracle3_exact_ok"] and (r["base_exact_ok"] != r["h1_exact_ok"] or r["base_exact_ok"] != r["llama_direct_exact_ok"]))]
    win_all = [r for r in rows if r["oracle3_exact_ok"]]
    v4_correct = sum(1 for r in win_all if r["h1p4_exact_ok"])
    v5_correct = sum(1 for r in win_all if r["h1p5_exact_ok"])
    print(f"\n  on oracle3-true rows ({len(win_all)}): v4={v4_correct}  v5={v5_correct}")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "summaries": {
            "h1p4_con": sH4, "h1p5_con": sH5, "oracle3": oracle,
        },
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
