# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 30 — H1.11: Llama synthesis from candidates (not selection).

The picker plateaued at ~72% while oracle4+canon = 84.3%. Picking can
never exceed the oracle. But the hard items often have gold args SPREAD
across candidates (one has the right time, another the right intensity),
and merge-oracle analysis shows +2.3 pp of assembly headroom on top of
the 84.3% — plus a 70B synthesizer can also FIX values no candidate got.

H1.11: give Llama-70B the user query, the function schema, and all four
candidate calls, and ask it to PRODUCE the single best call (it may merge
arguments across candidates or correct a value). The synthesized call is
then:
  (a) scored alone (canonicalized), and
  (b) added as a 5th candidate, with a final 5-way pick.

Reads ../results/iter26_h1p6.json
Writes ../results/iter30_h1p11.json
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

from bench_common import aggregate, args_match, load_hf_token, load_test, parse_call, parse_gold, print_summary
from bench_h2_rerank import parse_user_and_candidates
from bench_h1p5_picker_v2 import _registry_schema_str, build_picker_msg_v2, llama_pick_v2, REGISTRY_PATH
from canon import canonicalize_args, canonicalize_call


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


SYNTH_SYSTEM = (
    "You produce the single correct smart-home tool call. You are given "
    "the user query, the chosen function's schema, and several candidate "
    "tool calls produced by other systems. The candidates are EVIDENCE: "
    "each may be fully correct, partially correct, or wrong. Your job:\n"
    "- Decide the correct function name (must be in the candidate list).\n"
    "- Assemble the correct arguments. Take the best value for each key "
    "from whichever candidate got it right, or correct a value none got "
    "right, based on the user query and the schema.\n"
    "- KEY COMPLETENESS IS CRITICAL: include every argument key that the "
    "user's request implies. If a key appears in MOST candidate calls it "
    "is almost certainly part of the correct answer — do not drop it. "
    "Common dropped keys: labels ('oven', 'pasta'), modes ('heat'), "
    "messages, area/room. Only omit a key if it is clearly spurious.\n"
    "- Extract FULL multi-word values exactly as the user phrased them "
    "(e.g. 'basement gym', not 'gym'; 'back_door', not 'door').\n"
    "- Do not invent a key that no candidate has and the user did not "
    "mention.\n"
    "- Output ONE JSON object {\"name\":...,\"arguments\":{...}} — no prose, "
    "no markdown."
)


def build_synth_msg(user_query, cand_names, candidates, registry):
    """candidates: list of (label, call_dict)."""
    parts = [
        f'User said: "{user_query}"',
        f"Allowed function names: {json.dumps(cand_names)}",
        "",
    ]
    # Show schema for every distinct proposed name
    shown = set()
    for _, c in candidates:
        nm = (c or {}).get("name")
        if nm and nm not in shown:
            shown.add(nm)
            parts.append(f"Schema for {nm}: {_registry_schema_str(registry, nm)}")
    parts.append("")
    parts.append("Candidate tool calls (evidence):")
    for label, c in candidates:
        if not c or not c.get("name"):
            parts.append(f"  {label}: (no valid output)")
        else:
            parts.append(
                f"  {label}: {json.dumps({'name': c['name'], 'arguments': c.get('arguments') or {}}, separators=(',', ':'))}"
            )
    parts.append("")
    parts.append(
        "Produce the single correct tool call as one JSON object on one line."
    )
    return "\n".join(parts)


def llama_synth(token, msg) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYNTH_SYSTEM},
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


def cscore(name, args, gold_str, registry=None):
    """Score a call. With `registry`, argument values are enum-snapped to
    their registry enum before comparison (Iter 38, +3 pp — "gym" ->
    "basement gym", "living_room" -> "living room"). Both sides are snapped
    + canonicalized, so it measures enum-class equality."""
    gold = parse_gold(gold_str)
    name_ok = name == gold["name"] and gold["name"] is not None
    a_ok = args_match(
        canonicalize_call(name, args or {}, registry),
        canonicalize_call(gold["name"], gold["arguments"], registry),
    )
    return name_ok, a_ok, name_ok and a_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/iter26_h1p6.json")
    ap.add_argument("--out", default="../results/iter30_h1p11.json")
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

    def candidates_of(r):
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

    # Step 1: synthesize
    t0 = time.time()

    def synth(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["synth"] = None
            return r
        msg = build_synth_msg(user_query, cand_names, candidates_of(r), registry)
        r["synth"] = llama_synth(token, msg)
        return r

    print(f"\n[1/2] synthesizing {n} items")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(synth, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{n}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Score synth alone (canonicalized)
    synth_correct = 0
    for r in rows:
        c = r.get("synth")
        if not c:
            r["synth_name_ok"] = r["synth_args_ok"] = r["synth_exact_ok"] = False
            continue
        no, ao, eo = cscore(c.get("name"), c.get("arguments") or {}, r["gold"], registry)
        r["synth_name_ok"] = no
        r["synth_args_ok"] = ao
        r["synth_exact_ok"] = eo
        if eo:
            synth_correct += 1
    print(f"\n  synth alone (canon): {synth_correct/n*100:.2f}% ({synth_correct}/{n})")

    # Oracles
    def cc(r, lbl):  # canonicalized candidate score
        return r.get(f"cc_{lbl}", False)
    # Recompute canon candidate scores
    for r in rows:
        for lbl, c in candidates_of(r):
            _, _, eo = cscore(c["name"], c["arguments"], r["gold"], registry)
            r[f"cc_{lbl}_exact_ok"] = eo
    oracle4 = sum(
        1 for r in rows
        if any(r[f"cc_{l}_exact_ok"] for l in "ABCD")
    ) / n
    oracle5 = sum(
        1 for r in rows
        if any(r[f"cc_{l}_exact_ok"] for l in "ABCD") or r["synth_exact_ok"]
    ) / n
    print(f"  oracle4 (canon):       {oracle4*100:.2f}%")
    print(f"  oracle5 (+synth):      {oracle5*100:.2f}%  ({(oracle5-oracle4)*100:+.2f} pp)")
    synth_unique = sum(
        1 for r in rows
        if r["synth_exact_ok"] and not any(r[f"cc_{l}_exact_ok"] for l in "ABCD")
    )
    print(f"  synth-unique new correct: {synth_unique}")

    # Step 2: final 5-way pick {A,B,C,D,E=synth}
    print(f"\n[2/2] final 5-way pick")
    work = []
    for r in rows:
        cands = candidates_of(r)
        s = r.get("synth")
        cands.append(("E", {"name": (s or {}).get("name"),
                            "arguments": canonicalize_args((s or {}).get("arguments") or {})}))
        r["_cands5"] = cands
        keys = {(c["name"], json.dumps(c["arguments"], sort_keys=True)) for _, c in cands}
        if len(keys) == 1:
            r["pick5"] = "A"
            continue
        work.append(r)
    print(f"  {len(work)} rows need pick")

    def pick5(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["pick5"] = "A"
            return r
        msg = build_picker_msg_v2(user_query, cand_names, r["_cands5"], registry)
        p = llama_pick_v2(token, msg, {"A", "B", "C", "D", "E"})
        r["pick5"] = p or "A"
        return r

    t1 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(pick5, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 30 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t1:.0f}s", flush=True)
    print(f"  done in {time.time()-t1:.0f}s")

    label_score = {
        "A": "cc_A_exact_ok", "B": "cc_B_exact_ok", "C": "cc_C_exact_ok",
        "D": "cc_D_exact_ok", "E": "synth_exact_ok",
    }
    for r in rows:
        p = r.get("pick5", "A")
        r["h1p11_exact_ok"] = r[label_score[p]]
        r["h1p11_name_ok"] = r["h1p11_exact_ok"]
        del r["_cands5"]

    sSynth = aggregate(rows, "synth")
    sH11 = aggregate(rows, "h1p11")
    from collections import Counter
    picks = Counter(r.get("pick5") for r in rows)

    print(f"\n=== Iter 30 H1.11: synthesis + 5-way pick ===")
    print()
    print_summary("synth alone (canon)", sSynth)
    print()
    print_summary("H1.11 (synth in pool, 5-way pick)", sH11)
    print(f"\n  oracle4 canon:   {oracle4*100:.2f}%")
    print(f"  oracle5 +synth:  {oracle5*100:.2f}%")
    print(f"  picks: {dict(picks)}")
    print(f"\n  --- progression ---")
    print(f"  H1.6_con +canon:  72.33%")
    print(f"  H1.11 synth pick: {sH11['exact_acc']*100:.2f}%")
    # Also report: just take synth always (no pick)
    print(f"  synth-always:     {synth_correct/n*100:.2f}%")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "synth_summary": sSynth, "h1p11_summary": sH11,
        "oracle4_canon": oracle4, "oracle5": oracle5,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
