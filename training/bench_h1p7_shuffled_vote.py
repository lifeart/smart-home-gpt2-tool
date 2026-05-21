# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 26.3 — H1.7_con: shuffled-label multi-sample voting picker.

H1.5_con (3-way CoT picker) = 71.0%. H1.6_con (4-way CoT picker) = 69.7%
— degraded by candidate-pool growth. Analysis showed H1.5 picker has
heavy A-bias (15/20 misses chose A wrongly). Hypothesis: positional bias
+ single-shot reasoning is the bottleneck. Mitigations:

  1. Shuffle option labels per call so the model can't lean on "A is
     usually right".
  2. Multi-sample: run the picker 3 times (different shuffles), majority-
     vote BY CANDIDATE (not by label).
  3. Use temperature 0.3 — enough variance to break ties via shuffled
     reasoning, not so high that calls diverge.

Reads /tmp/iter26_h1p6.json (has all 4 candidates per row).
Writes /tmp/iter26_h1p7.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests

from bench_common import aggregate, load_hf_token, load_test, print_summary, score
from bench_h2_rerank import parse_user_and_candidates
from bench_h1p5_picker_v2 import (
    PICKER_SYSTEM_V2, build_picker_msg_v2, llama_pick_v2, REGISTRY_PATH,
)


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


PICK_RE = re.compile(r"PICK\s*:\s*([A-Z])", re.IGNORECASE)


def llama_pick_sampled(token, msg, allowed, temperature=0.3, seed=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PICKER_SYSTEM_V2},
            {"role": "user", "content": msg},
        ],
        "temperature": temperature,
        "max_tokens": 350,
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
    ap.add_argument("--in", dest="inp", default="/tmp/iter26_h1p6.json")
    ap.add_argument("--out", default="/tmp/iter26_h1p7.json")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--samples", type=int, default=3,
                    help="Picker calls per item with different shuffles.")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--n-options", type=int, default=4, choices=[3, 4],
                    help="3-way (base/H1/llama_direct) or 4-way (+LAO).")
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        return
    registry = json.loads(REGISTRY_PATH.read_text())
    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    print(f"[in] {len(rows)} rows  options={args.n_options} samples={args.samples}")

    def get_candidates(r):
        cands = [
            ("base", {"name": r["base_pred_name"], "arguments": r["base_pred_args"]},
             r["base_exact_ok"]),
            ("h1",   {"name": r["base_pred_name"], "arguments": r["h1_pred_args"]},
             r["h1_exact_ok"]),
            ("llama_direct", r.get("llama_direct"), r["llama_direct_exact_ok"]),
        ]
        if args.n_options == 4:
            cands.append(("llama_args_only", r.get("llama_args_only"), r["llama_args_only_exact_ok"]))
        return cands

    # For each item, run `samples` picker calls with shuffled labels.
    # Map each call's PICK back to the candidate name. Majority vote.
    work = []
    for r in rows:
        cands = get_candidates(r)
        # Dedupe by canonical key — if all candidates are identical, skip pick.
        seen = set()
        unique_idx = []
        for i, (n_, c, _) in enumerate(cands):
            key = (
                (c or {}).get("name"),
                json.dumps((c or {}).get("arguments", {}), sort_keys=True),
            )
            if key not in seen:
                seen.add(key)
                unique_idx.append(i)
        if len(unique_idx) == 1:
            r["picker_winner"] = cands[0][0]
            continue
        work.append(r)
    print(f"[work] {len(work)} rows need pick")

    rng_master = random.Random(42)

    def pick(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["picker_winner"] = "base"
            return r
        cands = get_candidates(r)
        # Run `samples` calls. Each call shuffles label assignment.
        votes = {n_: 0 for (n_, _, _) in cands}
        label_pool = "ABCD"[:args.n_options]
        for s in range(args.samples):
            local_rng = random.Random(rng_master.randint(0, 10**9))
            perm = list(range(len(cands)))
            local_rng.shuffle(perm)
            shuffled = [cands[idx] for idx in perm]
            opt_list = [
                (label_pool[i], c) for i, (_, c, _) in enumerate(shuffled)
            ]
            msg = build_picker_msg_v2(user_query, cand_names, opt_list, registry)
            picked_label = llama_pick_sampled(
                token, msg, set(label_pool),
                temperature=args.temperature,
            )
            if picked_label is None:
                continue
            picked_label_idx = label_pool.index(picked_label)
            picked_cand_name = shuffled[picked_label_idx][0]
            votes[picked_cand_name] += 1
        # Majority vote — tiebreak by "base" then "h1" then "llama_direct" then "llama_args_only"
        order = ["base", "h1", "llama_direct", "llama_args_only"]
        winner = max(order, key=lambda n_: (votes.get(n_, 0), -order.index(n_)))
        r["picker_winner"] = winner
        r["picker_votes"] = votes
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

    # Score the result
    correct = 0
    for r in rows:
        w = r.get("picker_winner", "base")
        ok_map = {
            "base": r["base_exact_ok"],
            "h1": r["h1_exact_ok"],
            "llama_direct": r["llama_direct_exact_ok"],
            "llama_args_only": r.get("llama_args_only_exact_ok", False),
        }
        ok = ok_map[w]
        r["h1p7_exact_ok"] = ok
        r["h1p7_name_ok"] = ok_map[w]  # imprecise but exact-ok ⊆ name-ok always
        if ok:
            correct += 1

    n = len(rows)
    sH7 = aggregate(rows, "h1p7")
    oracle3 = sum(1 for r in rows if r.get("oracle3_exact_ok")) / n
    oracle4 = sum(1 for r in rows if r.get("oracle4_exact_ok", False)) / n
    from collections import Counter
    winners = Counter(r.get("picker_winner") for r in rows)

    print(f"\n=== Iter 26.3 H1.7_con: shuffled label + multi-sample vote ===")
    print(f"  n_options={args.n_options} samples={args.samples} T={args.temperature}")
    print()
    print_summary(f"H1.7_con", sH7)
    print(f"\n  oracle3:    {oracle3*100:.2f}%")
    print(f"  oracle4:    {oracle4*100:.2f}%")
    print(f"  H1.5_con:   71.00%")
    print(f"  H1.6_con:   69.67%")
    print(f"  H1.7_con:   {correct/n*100:.2f}%  ({correct}/{n})")
    print(f"  winners: {dict(winners)}")

    Path(args.out).write_text(json.dumps({
        "n": n, "n_options": args.n_options, "samples": args.samples,
        "summary": sH7, "oracle3": oracle3, "oracle4": oracle4,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
