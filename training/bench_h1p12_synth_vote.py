# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 31 — H1.12: multi-sample synthesis voting + 2-way final pick.

H1.11 synthesizer (Llama produces a call from 4 candidates as evidence)
hit 72.7% alone / 74.0% in a 5-way pick. The synthesizer is the strongest
single mechanism found. This iteration strengthens it two ways:

  1. Self-consistency: synthesize 3 extra samples at temperature, vote by
     canonical (name, args). The temp-0 synth from H1.11 is the 4th vote.
     High-agreement synth answers are very reliable.
  2. Final 2-way pick: {synth_voted, H1.6+canon pick}. A 2-way decision
     is far easier for the picker than 5-way (H1.3's 2-way picker scored
     80% on disagreements vs ~70% for wider pools).

Reads ../results/iter30_h1p11.json (has temp-0 synth + all candidates +
the h1p6 4-way pick label).
Writes ../results/iter31_h1p12.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

from bench_common import aggregate, args_match, load_hf_token, load_test, parse_call, parse_gold, print_summary
from bench_h2_rerank import parse_user_and_candidates
from bench_h1p5_picker_v2 import build_picker_msg_v2, llama_pick_v2, REGISTRY_PATH
from bench_h1p11_synth import SYNTH_SYSTEM as _SYS, build_synth_msg as _BUILD
from canon import canonicalize_args


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


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


def llama_synth_sampled(token, msg, temperature, seed=None) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": msg},
        ],
        "temperature": temperature,
        "max_tokens": 220,
    }
    if seed is not None:
        payload["seed"] = seed
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


def ckey(call):
    if not call or not call.get("name"):
        return "__none__"
    return json.dumps(
        {"n": call["name"], "a": canonicalize_args(call.get("arguments") or {})},
        sort_keys=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/iter30_h1p11.json")
    ap.add_argument("--out", default="../results/iter31_h1p12.json")
    ap.add_argument("--extra-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.5)
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
    print(f"[in] {n} rows. extra synth samples={args.extra_samples} T={args.temperature}")

    # Step 1: extra synth samples
    t0 = time.time()

    def sample(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["synth_samples"] = []
            return r
        msg = _BUILD(user_query, cand_names, candidates_of(r), registry)
        outs = []
        for s in range(args.extra_samples):
            c = llama_synth_sampled(token, msg, args.temperature, seed=1000 * i + s)
            if c:
                outs.append(c)
        r["synth_samples"] = outs
        return r

    print(f"\n[1/2] {args.extra_samples} extra synth samples / item")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(sample, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{n}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Vote: temp-0 synth + extra samples. Majority by canonical key.
    for r in rows:
        votes = []
        if r.get("synth"):
            votes.append(r["synth"])
        votes.extend(r.get("synth_samples", []))
        if not votes:
            r["synth_voted"] = None
            continue
        keyed = [(ckey(v), v) for v in votes]
        counts = Counter(k for k, _ in keyed)
        best_key, best_n = counts.most_common(1)[0]
        # tie-break: prefer the temp-0 synth's key if it's tied for top
        top_keys = {k for k, c in counts.items() if c == best_n}
        if r.get("synth") and ckey(r["synth"]) in top_keys:
            chosen = r["synth"]
        else:
            chosen = next(v for k, v in keyed if k == best_key)
        r["synth_voted"] = chosen
        r["synth_agreement"] = best_n / len(votes)

    # Score synth_voted
    sv_correct = 0
    for r in rows:
        c = r.get("synth_voted")
        if not c:
            r["sv_name_ok"] = r["sv_args_ok"] = r["sv_exact_ok"] = False
            continue
        no, ao, eo = cscore(c.get("name"), c.get("arguments") or {}, r["gold"])
        r["sv_name_ok"] = no
        r["sv_args_ok"] = ao
        r["sv_exact_ok"] = eo
        if eo:
            sv_correct += 1
    print(f"\n  synth_voted alone: {sv_correct/n*100:.2f}% ({sv_correct}/{n})")

    # Step 2: 2-way final pick {synth_voted, h1p6-pick}.
    # h1p6 pick label is in row as llama_pick4 (A/B/C/D over canon candidates).
    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for r in rows:
        cands = candidates_of(r)
        p4 = r.get("llama_pick4", "A")
        r["_h1p6_call"] = cands[label_to_idx[p4]][1]

    work = []
    for r in rows:
        sv = r.get("synth_voted")
        h6 = r.get("_h1p6_call")
        if ckey(sv) == ckey(h6):
            r["pick2"] = "S"
            continue
        work.append(r)
    print(f"\n[2/2] 2-way pick {{synth_voted, h1p6}} — {len(work)} rows")

    def pick2(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["pick2"] = "S"
            return r
        opts = [("A", r.get("synth_voted")), ("B", r.get("_h1p6_call"))]
        msg = build_picker_msg_v2(user_query, cand_names, opts, registry)
        p = llama_pick_v2(token, msg, {"A", "B"})
        r["pick2"] = "S" if p == "A" else ("H" if p == "B" else "S")
        return r

    t1 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(pick2, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 30 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t1:.0f}s", flush=True)
    print(f"  done in {time.time()-t1:.0f}s")

    # Score H1.12
    for r in rows:
        if r.get("pick2", "S") == "S":
            c = r.get("synth_voted")
        else:
            c = r.get("_h1p6_call")
        no, ao, eo = cscore((c or {}).get("name"), (c or {}).get("arguments") or {}, r["gold"])
        r["h1p12_name_ok"] = no
        r["h1p12_args_ok"] = ao
        r["h1p12_exact_ok"] = eo
        del r["_h1p6_call"]

    # Oracle including synth_voted
    for r in rows:
        cands = candidates_of(r)
        for lbl, c in cands:
            _, _, eo = cscore(c["name"], c["arguments"], r["gold"])
            r[f"_cc_{lbl}"] = eo
    oracle6 = sum(
        1 for r in rows
        if any(r[f"_cc_{l}"] for l in "ABCD") or r["sv_exact_ok"]
        or r.get("synth_exact_ok")
    ) / n

    sSV = aggregate(rows, "sv")
    sH12 = aggregate(rows, "h1p12")
    picks = Counter(r.get("pick2") for r in rows)

    print(f"\n=== Iter 31 H1.12: synth voting + 2-way final pick ===")
    print()
    print_summary("synth_voted alone", sSV)
    print()
    print_summary("H1.12 (2-way pick)", sH12)
    print(f"\n  oracle (all incl synth variants): {oracle6*100:.2f}%")
    print(f"  pick2: {dict(picks)}")
    print(f"\n  --- progression ---")
    print(f"  H1.11 (synth 5-way):  74.00%")
    print(f"  synth_voted alone:    {sv_correct/n*100:.2f}%")
    print(f"  H1.12 (2-way):        {sH12['exact_acc']*100:.2f}%")

    for r in rows:
        for l in "ABCD":
            r.pop(f"_cc_{l}", None)
    Path(args.out).write_text(json.dumps({
        "n": n, "synth_voted_summary": sSV, "h1p12_summary": sH12,
        "oracle": oracle6, "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
