# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 23 H1.3 — Llama-70B picks between v6-baseline and H1 per item.

Reads /tmp/h1_n300.json (rows from bench_h1_two_stage.py) and, for each row,
asks Llama-70B to compare (baseA name+args) vs (H1 name+args) given the user
query and candidate list. Picks one. Scores winner vs gold.

Reported: baseline (= baseA), H1, oracle (= max(baseA, H1)), H1.3 (Llama
picker). No GPU needed — pure post-processing on the saved rows.

Run:
    python training/bench_h1p3_llama_pick.py --in /tmp/h1_n300.json
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

from bench_common import (
    aggregate,
    load_hf_token,
    parse_gold,
    print_summary,
    score,
)
from bench_h2_rerank import parse_user_and_candidates  # reuse parser


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PICK_SYSTEM = (
    "You compare two model-emitted tool calls for the same user query and "
    "pick the better one. Reply only in the exact required format. The "
    "better call is the one whose function name correctly addresses the user "
    "intent AND whose arguments are most faithful to the query. If both are "
    "equivalent, prefer A. Never invent a third option."
)


def build_pick_msg(
    user_query: str,
    cand_names: list[str],
    a_name: Optional[str],
    a_args: dict,
    b_name: Optional[str],
    b_args: dict,
) -> str:
    return (
        "User said: \"" + user_query + "\"\n"
        f"Candidate functions: {json.dumps(cand_names)}\n\n"
        "Option A:\n"
        f"  name: {a_name!r}\n"
        f"  arguments: {json.dumps(a_args, separators=(',', ':'))}\n\n"
        "Option B:\n"
        f"  name: {b_name!r}\n"
        f"  arguments: {json.dumps(b_args, separators=(',', ':'))}\n\n"
        "Decide which option is correct. If both are equally correct (e.g. "
        "identical, or both wrong in the same way), pick A. If only one is "
        "correct, pick that one. Reply in EXACT format:\n"
        "PICK: A|B\n"
        "REASON: <one line>"
    )


PICK_RE = re.compile(r"PICK\s*:\s*([AB])", re.IGNORECASE)


def llama_pick(token: str, user_msg: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 60,
    }
    for attempt in range(3):
        try:
            r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 200:
            try:
                txt = r.json()["choices"][0]["message"]["content"]
            except Exception:
                return None
            m = PICK_RE.search(txt or "")
            if m:
                return m.group(1).upper()
            return None
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/h1_n300.json")
    ap.add_argument("--out", default="iter23_h1p3_results.json")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all rows from input file")
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN missing")
        return

    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    if args.limit and args.limit < len(rows):
        rows = rows[: args.limit]
    print(f"[in] {args.inp}: {len(rows)} rows")

    # Prepare per-row work
    # For each row we need: user_query, cand_names (re-parsed from gold's prompt).
    # Problem: the saved row doesn't preserve the original prompt. Reload sh_test.json
    # and join by index.
    from bench_common import load_test
    test = load_test()

    # Items in rows already follow test order (i=0..n-1).
    # Items where baseA and H1 predictions are identical → no need to call Llama;
    # both correct or both wrong, the "pick A" rule applies → use baseA.
    work: list[tuple[int, dict]] = []
    for r in rows:
        i = r["i"]
        item = test[i]
        same_name = r["baseA_pred_name"] == r["h1_pred_name"]
        same_args = json.dumps(r["baseA_pred_args"], sort_keys=True) == json.dumps(
            r["h1_pred_args"], sort_keys=True
        )
        if same_name and same_args:
            r["needs_pick"] = False
            r["llama_pick"] = "A"  # identical → A
            continue
        r["needs_pick"] = True
        work.append((i, {"row": r, "prompt": item["prompt"]}))

    print(f"[work] {len(work)} rows need Llama pick "
          f"(others are A==B, default to A)")

    def task(args_):
        i, payload = args_
        r = payload["row"]
        prompt = payload["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["llama_pick"] = "A"
            r["llama_pick_error"] = "parse_failed"
            return r
        msg = build_pick_msg(
            user_query, cand_names,
            r["baseA_pred_name"], r["baseA_pred_args"],
            r["h1_pred_name"], r["h1_pred_args"],
        )
        pick = llama_pick(token, msg)
        r["llama_pick"] = pick or "A"
        if pick is None:
            r["llama_pick_error"] = "router_failed"
        return r

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(task, w) for w in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Compute final score per row
    for r in rows:
        p = r.get("llama_pick", "A")
        if p == "A":
            r["h1p3_name_ok"] = r["baseA_name_ok"]
            r["h1p3_args_ok"] = r["baseA_args_ok"]
            r["h1p3_exact_ok"] = r["baseA_exact_ok"]
        else:
            r["h1p3_name_ok"] = r["h1_name_ok"]
            r["h1p3_args_ok"] = r["h1_args_ok"]
            r["h1p3_exact_ok"] = r["h1_exact_ok"]
        r["oracle_exact_ok"] = r["baseA_exact_ok"] or r["h1_exact_ok"]

    sA = aggregate(rows, "baseA")
    sH1 = aggregate(rows, "h1")
    sH1p3 = aggregate(rows, "h1p3")
    n = len(rows)
    oracle = sum(1 for r in rows if r["oracle_exact_ok"]) / n
    # H1.2 (clean-only fallback) — also report for comparison
    h12_correct = 0
    for r in rows:
        ok = r["baseA_exact_ok"] if r["domain"] == "clean" else r["h1_exact_ok"]
        if ok:
            h12_correct += 1
    h12 = h12_correct / n

    print()
    print(f"=== Iter 23 H1.3: Llama-70B picker between v6-baseline and H1 ===")
    print()
    print_summary("baseline (v6 one-shot)", sA)
    print()
    print_summary("H1 (two-stage)", sH1)
    print(f"\n  H1.2 (clean-only domain fallback): exact = {h12*100:.2f}% ({h12_correct}/{n})")
    print(f"  oracle (best of {{baseA, H1}} per item): exact = {oracle*100:.2f}%")
    print()
    print_summary("H1.3 (Llama picks A or B)", sH1p3)

    # Diagnostics: how often did Llama pick correctly?
    a_picked = sum(1 for r in rows if r.get("llama_pick") == "A")
    b_picked = sum(1 for r in rows if r.get("llama_pick") == "B")
    print(f"\n  Llama picks: A={a_picked}/{n}  B={b_picked}/{n}")
    # When A and B disagreed on exact-correctness, did Llama pick the right one?
    disagree = [r for r in rows
                if r["baseA_exact_ok"] != r["h1_exact_ok"]]
    if disagree:
        correct_picks = 0
        for r in disagree:
            picked_correct = (r["llama_pick"] == "A" and r["baseA_exact_ok"]) or \
                             (r["llama_pick"] == "B" and r["h1_exact_ok"])
            if picked_correct:
                correct_picks += 1
        print(f"  Picker accuracy on disagreement rows ({len(disagree)}): "
              f"{correct_picks}/{len(disagree)} = {correct_picks/len(disagree)*100:.1f}%")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "summaries": {
            "baseline_v6": sA,
            "h1": sH1,
            "h1p2_clean_fallback_exact": h12,
            "oracle_exact": oracle,
            "h1p3_llama_pick": sH1p3,
        },
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
