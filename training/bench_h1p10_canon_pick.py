# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 29.2 — H1.10: re-pick over canonicalized candidates.

Iter 29 showed value canonicalization lifts oracle4 81.7% → 84.3% and
H1.6+canon (old picker decisions, re-scored) → 72.3%. But those picker
decisions were made on RAW candidates — the picker saw "3 PM" / "24.44"
and may have judged a format-differing option as wrong.

This re-runs the 4-way CoT picker on CANONICALIZED candidate values, so
the picker compares clean values. Then scores under canon.

Reads ../results/iter26_h1p6.json
Writes ../results/iter29_h1p10.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests

from bench_common import aggregate, args_match, load_hf_token, load_test, parse_gold, print_summary
from bench_h2_rerank import parse_user_and_candidates
from bench_h1p5_picker_v2 import build_picker_msg_v2, llama_pick_v2, REGISTRY_PATH
from canon import canonicalize_args


def cscore(name, cargs, gold_str):
    gold = parse_gold(gold_str)
    name_ok = name == gold["name"] and gold["name"] is not None
    cgold = canonicalize_args(gold["arguments"])
    a_ok = args_match(cargs, cgold)
    return name_ok, a_ok, name_ok and a_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/iter26_h1p6.json")
    ap.add_argument("--out", default="../results/iter29_h1p10.json")
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

    # Build canonicalized candidates per row.
    for r in rows:
        ld = r.get("llama_direct") or {}
        lao = r.get("llama_args_only") or {}
        r["_cands"] = {
            "A": {"name": r["base_pred_name"],
                  "arguments": canonicalize_args(r["base_pred_args"])},
            "B": {"name": r["base_pred_name"],
                  "arguments": canonicalize_args(r["h1_pred_args"])},
            "C": {"name": ld.get("name"),
                  "arguments": canonicalize_args(ld.get("arguments") or {})},
            "D": {"name": lao.get("name"),
                  "arguments": canonicalize_args(lao.get("arguments") or {})},
        }
        # Score each canonicalized candidate
        for lbl, c in r["_cands"].items():
            no, ao, eo = cscore(c["name"], c["arguments"], r["gold"])
            r[f"cc_{lbl}_name_ok"] = no
            r[f"cc_{lbl}_args_ok"] = ao
            r[f"cc_{lbl}_exact_ok"] = eo

    # Which rows need a pick (canonicalized candidates not all identical)?
    work = []
    for r in rows:
        keys = {
            (c["name"], json.dumps(c["arguments"], sort_keys=True))
            for c in r["_cands"].values()
        }
        if len(keys) == 1:
            r["canon_pick"] = "A"
            continue
        work.append(r)
    print(f"[work] {len(work)} rows need pick")

    def pick(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["canon_pick"] = "A"
            return r
        opts = [(lbl, r["_cands"][lbl]) for lbl in ("A", "B", "C", "D")]
        msg = build_picker_msg_v2(user_query, cand_names, opts, registry)
        p = llama_pick_v2(token, msg, {"A", "B", "C", "D"})
        r["canon_pick"] = p or "A"
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

    # Score
    for r in rows:
        p = r.get("canon_pick", "A")
        r["h1p10_name_ok"] = r[f"cc_{p}_name_ok"]
        r["h1p10_args_ok"] = r[f"cc_{p}_args_ok"]
        r["h1p10_exact_ok"] = r[f"cc_{p}_exact_ok"]
        r["oracle4c_exact_ok"] = any(
            r[f"cc_{lbl}_exact_ok"] for lbl in ("A", "B", "C", "D")
        )
        del r["_cands"]  # don't serialize

    sH10 = aggregate(rows, "h1p10")
    oracle4c = sum(1 for r in rows if r["oracle4c_exact_ok"]) / n
    from collections import Counter
    picks = Counter(r.get("canon_pick") for r in rows)

    print(f"\n=== Iter 29.2 H1.10: 4-way picker over canonicalized candidates ===")
    print()
    print_summary("H1.10 (canon candidates + canon pick)", sH10)
    print(f"\n  oracle4 (canon): {oracle4c*100:.2f}%")
    print(f"  picks: {dict(picks)}")
    print(f"\n  --- progression ---")
    print(f"  H1.5_con (raw):        69-71%")
    print(f"  H1.6_con +canon:       72.33%")
    print(f"  H1.10 (canon+repick):  {sH10['exact_acc']*100:.2f}%")
    print(f"  oracle4 canon:         {oracle4c*100:.2f}%")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "h1p10_summary": sH10,
        "oracle4_canon": oracle4c,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
