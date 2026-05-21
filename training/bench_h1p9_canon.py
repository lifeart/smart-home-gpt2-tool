# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Iter 29 — H1.9: apply value canonicalization to all candidates.

Loads iter26_h1p6.json (base / h1 / llama_direct / llama_args_only preds),
canonicalizes every candidate's argument values (training/canon.py), then
re-scores and re-computes the oracle. Also re-scores the existing picker
decisions (h1p5 = 3-way CoT, h1p6 = 4-way) under canonicalization.

Pure post-processing — no API calls, no GPU.

Run:
    python training/bench_h1p9_canon.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_common import aggregate, args_match, parse_gold, print_summary
from canon import canonicalize_args


def cscore(name, args, gold_str):
    """Score with canonicalized prediction args."""
    gold = parse_gold(gold_str)
    cargs = canonicalize_args(args or {})
    name_ok = name == gold["name"] and gold["name"] is not None
    # Canonicalize gold too — it is already canonical, but this makes the
    # comparison symmetric and harmless (gold "15:00" stays "15:00").
    cgold = canonicalize_args(gold["arguments"])
    a_ok = args_match(cargs, cgold)
    return name_ok, a_ok, name_ok and a_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/iter26_h1p6.json")
    ap.add_argument("--out", default="../results/iter29_canon.json")
    args = ap.parse_args()

    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    n = len(rows)
    print(f"[in] {n} rows")

    cand_specs = [
        ("base", lambda r: (r["base_pred_name"], r["base_pred_args"])),
        ("h1", lambda r: (r["base_pred_name"], r["h1_pred_args"])),
        ("llama_direct", lambda r: (
            (r.get("llama_direct") or {}).get("name"),
            (r.get("llama_direct") or {}).get("arguments", {}),
        )),
        ("llama_args_only", lambda r: (
            (r.get("llama_args_only") or {}).get("name"),
            (r.get("llama_args_only") or {}).get("arguments", {}),
        )),
    ]

    # Re-score every candidate, with and without canon.
    for r in rows:
        for tag, getter in cand_specs:
            name, a = getter(r)
            no, ao, eo = cscore(name, a, r["gold"])
            r[f"c_{tag}_name_ok"] = no
            r[f"c_{tag}_args_ok"] = ao
            r[f"c_{tag}_exact_ok"] = eo

    # Per-candidate before/after
    print("\n=== Per-candidate exact: raw → canonicalized ===")
    for tag, _ in cand_specs:
        raw = sum(1 for r in rows if r.get(f"{tag}_exact_ok"))
        can = sum(1 for r in rows if r.get(f"c_{tag}_exact_ok"))
        print(f"  {tag:<16} {raw/n*100:5.1f}% → {can/n*100:5.1f}%  "
              f"({raw} → {can}, {can-raw:+d})")

    # Oracle before/after
    raw_oracle = sum(
        1 for r in rows
        if r["base_exact_ok"] or r["h1_exact_ok"]
        or r["llama_direct_exact_ok"] or r["llama_args_only_exact_ok"]
    )
    can_oracle = sum(
        1 for r in rows
        if r["c_base_exact_ok"] or r["c_h1_exact_ok"]
        or r["c_llama_direct_exact_ok"] or r["c_llama_args_only_exact_ok"]
    )
    print(f"\n  oracle4 raw:  {raw_oracle/n*100:.2f}%  ({raw_oracle})")
    print(f"  oracle4 canon: {can_oracle/n*100:.2f}%  ({can_oracle}, "
          f"{can_oracle-raw_oracle:+d})")

    # Re-score the existing pickers (h1p5 = 3-way, h1p6 = 4-way) under canon.
    # The pick label is stored; map to the canonicalized candidate score.
    pick_label_to_tag = {
        "A": "base", "B": "h1", "C": "llama_direct", "D": "llama_args_only",
    }
    for r in rows:
        # h1p5 used llama_pick3_v2 (A/B/C)
        p5 = r.get("llama_pick3_v2", "A")
        r["c_h1p5_exact_ok"] = r[f"c_{pick_label_to_tag[p5]}_exact_ok"]
        r["c_h1p5_name_ok"] = r[f"c_{pick_label_to_tag[p5]}_name_ok"]
        r["c_h1p5_args_ok"] = r[f"c_{pick_label_to_tag[p5]}_args_ok"]
        # h1p6 used llama_pick4 (A/B/C/D)
        p6 = r.get("llama_pick4", "A")
        r["c_h1p6_exact_ok"] = r[f"c_{pick_label_to_tag[p6]}_exact_ok"]
        r["c_h1p6_name_ok"] = r[f"c_{pick_label_to_tag[p6]}_name_ok"]
        r["c_h1p6_args_ok"] = r[f"c_{pick_label_to_tag[p6]}_args_ok"]

    sH5_raw = aggregate(rows, "h1p5")
    sH5_can = aggregate(rows, "c_h1p5")
    sH6_can = aggregate(rows, "c_h1p6")
    print()
    print_summary("H1.5_con (raw)", sH5_raw)
    print()
    print_summary("H1.5_con + canon", sH5_can)
    print()
    print_summary("H1.6_con + canon", sH6_can)

    print(f"\n  --- summary ---")
    print(f"  H1.5_con raw:    {sH5_raw['exact_acc']*100:.2f}%")
    print(f"  H1.5_con +canon: {sH5_can['exact_acc']*100:.2f}%")
    print(f"  H1.6_con +canon: {sH6_can['exact_acc']*100:.2f}%")
    print(f"  oracle4 +canon:  {can_oracle/n*100:.2f}%")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "oracle4_raw": raw_oracle / n,
        "oracle4_canon": can_oracle / n,
        "h1p5_raw": sH5_raw["exact_acc"],
        "h1p5_canon": sH5_can["exact_acc"],
        "h1p6_canon": sH6_can["exact_acc"],
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
