# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 33 — H1.13: 2-pass synthesis with self-critique refinement.

Synth v2 (Iter 32) reached 78.7% alone — the strongest single mechanism.
This adds a second pass: the model sees its own pass-1 answer, the user
query, the schema, and the candidate calls again, and is asked to audit
each argument (needed key? correct value?) and emit a corrected call.

Reads ../results/iter32_synth2.json (has `synth` = pass-1 output).
Writes ../results/iter33_h1p13.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Optional

import requests

from bench_common import aggregate, args_match, load_hf_token, load_test, parse_call, parse_gold, print_summary
from bench_h2_rerank import parse_user_and_candidates
from bench_h1p5_picker_v2 import _registry_schema_str, REGISTRY_PATH
from canon import canonicalize_args


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def candidates_of(r):
    """The four GPT-2/Llama candidates (canonicalized) for one row."""
    ld = r.get("llama_direct") or {}
    lao = r.get("llama_args_only") or {}
    return [
        ("A", {"name": r["base_pred_name"],
               "arguments": canonicalize_args(r["base_pred_args"])}),
        ("B", {"name": r["base_pred_name"],
               "arguments": canonicalize_args(r["h1_pred_args"])}),
        ("C", {"name": ld.get("name"),
               "arguments": canonicalize_args(ld.get("arguments") or {})}),
        ("D", {"name": lao.get("name"),
               "arguments": canonicalize_args(lao.get("arguments") or {})}),
    ]


REFINE_SYSTEM = (
    "You audit and correct a smart-home tool call. You are given the user "
    "query, the function schema, several candidate calls from other "
    "systems, and a DRAFT answer. Audit the draft argument by argument:\n"
    "- Is every key the user's request implies present? (labels, modes, "
    "messages, area/room are often wrongly dropped — restore them if the "
    "user implied them or most candidates include them.)\n"
    "- Is each value correct and complete? (full multi-word values; right "
    "enum; right number; 24h time.)\n"
    "- Is any key spurious (not in schema, not implied)? Remove it.\n"
    "Output the corrected call as ONE JSON object "
    "{\"name\":...,\"arguments\":{...}} — no prose, no markdown. If the draft "
    "is already correct, output it unchanged."
)


def build_refine_msg(user_query, cand_names, candidates, draft, registry):
    parts = [
        f'User said: "{user_query}"',
        f"Allowed function names: {json.dumps(cand_names)}",
        "",
    ]
    shown = set()
    for _, c in candidates:
        nm = (c or {}).get("name")
        if nm and nm not in shown:
            shown.add(nm)
            parts.append(f"Schema for {nm}: {_registry_schema_str(registry, nm)}")
    parts.append("")
    parts.append("Candidate calls (evidence):")
    for label, c in candidates:
        if c and c.get("name"):
            parts.append(f"  {label}: {json.dumps({'name': c['name'], 'arguments': c.get('arguments') or {}}, separators=(',', ':'))}")
    parts.append("")
    draft_str = json.dumps(
        {"name": (draft or {}).get("name"),
         "arguments": (draft or {}).get("arguments") or {}},
        separators=(",", ":"),
    )
    parts.append(f"DRAFT answer to audit: {draft_str}")
    parts.append("")
    parts.append("Output the corrected tool call as one JSON object.")
    return "\n".join(parts)


def llama_refine(token, msg) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": msg},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
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
            return parse_call(txt or "")
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


def cscore(name, args, gold_str):
    gold = parse_gold(gold_str)
    name_ok = name == gold["name"] and gold["name"] is not None
    a_ok = args_match(canonicalize_args(args or {}), canonicalize_args(gold["arguments"]))
    return name_ok, a_ok, name_ok and a_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/iter32_synth2.json")
    ap.add_argument("--out", default="../results/iter33_h1p13.json")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        return
    registry = json.loads(REGISTRY_PATH.read_text())
    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    n = len(rows)
    print(f"[in] {n} rows")

    t0 = time.time()

    def refine(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        draft = r.get("synth")
        if not cand_names or not user_query or not draft:
            r["refined"] = draft
            return r
        msg = build_refine_msg(user_query, cand_names, candidates_of(r), draft, registry)
        out = llama_refine(token, msg)
        r["refined"] = out if out else draft
        return r

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(refine, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{n}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Score: synth (pass 1) vs refined (pass 2)
    synth_c = refined_c = 0
    changed = changed_better = changed_worse = 0
    for r in rows:
        s = r.get("synth")
        rf = r.get("refined")
        _, _, s_ok = cscore((s or {}).get("name"), (s or {}).get("arguments") or {}, r["gold"])
        no, ao, rf_ok = cscore((rf or {}).get("name"), (rf or {}).get("arguments") or {}, r["gold"])
        r["h1p13_name_ok"] = no
        r["h1p13_args_ok"] = ao
        r["h1p13_exact_ok"] = rf_ok
        if s_ok:
            synth_c += 1
        if rf_ok:
            refined_c += 1
        sk = json.dumps(canonicalize_args((s or {}).get("arguments") or {}), sort_keys=True)
        rk = json.dumps(canonicalize_args((rf or {}).get("arguments") or {}), sort_keys=True)
        sn = (s or {}).get("name")
        rn = (rf or {}).get("name")
        if sk != rk or sn != rn:
            changed += 1
            if rf_ok and not s_ok:
                changed_better += 1
            elif s_ok and not rf_ok:
                changed_worse += 1

    sH13 = aggregate(rows, "h1p13")
    print(f"\n=== Iter 33 H1.13: 2-pass synthesis (synth → refine) ===")
    print()
    print_summary("H1.13 (refined)", sH13)
    print(f"\n  --- progression ---")
    print(f"  synth v2 (pass 1):   {synth_c/n*100:.2f}%  ({synth_c}/{n})")
    print(f"  H1.13 (refined):     {refined_c/n*100:.2f}%  ({refined_c}/{n})")
    print(f"  refine changed {changed} items: "
          f"{changed_better} fixed, {changed_worse} broken, "
          f"{changed-changed_better-changed_worse} neutral")

    Path(args.out).write_text(json.dumps({
        "n": n, "synth_pass1_acc": synth_c / n,
        "h1p13_summary": sH13, "h1p13_acc": refined_c / n,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
