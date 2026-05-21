# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 23 H1.3_con — Llama-70B picks between base_con and H1_con per item.

Reads /tmp/iter23_h1_con_results.json (from cloud bench), runs Llama-70B
verifier rerank, prints exact-match for H1.3_con.

H1.2_con (domain-gate) is the no-external-dep ship config. H1.3_con shows
the ceiling if we allow a runtime Llama call.
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

from bench_common import aggregate, load_hf_token, load_test, print_summary
from bench_h2_rerank import parse_user_and_candidates


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PICK_SYSTEM = (
    "You compare two model-emitted tool calls for the same user query and "
    "pick the better one. Reply only in the exact required format. The "
    "better call is the one whose function name correctly addresses the user "
    "intent AND whose arguments are most faithful to the query. If both are "
    "equivalent, prefer A."
)


def build_pick_msg(user_query, cand_names, a_name, a_args, b_name, b_args):
    return (
        f'User said: "{user_query}"\n'
        f"Candidate functions: {json.dumps(cand_names)}\n\n"
        "Option A:\n"
        f"  name: {a_name!r}\n"
        f"  arguments: {json.dumps(a_args, separators=(',', ':'))}\n\n"
        "Option B:\n"
        f"  name: {b_name!r}\n"
        f"  arguments: {json.dumps(b_args, separators=(',', ':'))}\n\n"
        "Reply EXACTLY:\n"
        "PICK: A|B\n"
        "REASON: <one line>"
    )


PICK_RE = re.compile(r"PICK\s*:\s*([AB])", re.IGNORECASE)


def llama_pick(token, msg):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": msg},
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
            return m.group(1).upper() if m else None
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/iter23_h1_con_results.json")
    ap.add_argument("--out", default="/tmp/iter23_h1p3_con_results.json")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN missing")
        return

    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    print(f"[in] {args.inp}: {len(rows)} rows")

    # For each row, if base and H1 predictions identical → default to A.
    # Else, ask Llama.
    work = []
    for r in rows:
        same_name = r["base_pred_name"] == r["base_pred_name"]  # base always has a name
        # H1 used base's name (two-stage) — they should match always.
        # Args may differ.
        base_args_key = json.dumps(r["base_pred_args"], sort_keys=True)
        h1_args_key = json.dumps(r["h1_pred_args"], sort_keys=True)
        if base_args_key == h1_args_key:
            r["needs_pick"] = False
            r["llama_pick"] = "A"
            continue
        r["needs_pick"] = True
        work.append(r)
    print(f"[work] {len(work)} rows need pick "
          f"(others A==B → default A)")

    def task(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["llama_pick"] = "A"
            return r
        msg = build_pick_msg(
            user_query, cand_names,
            r["base_pred_name"], r["base_pred_args"],
            r["base_pred_name"], r["h1_pred_args"],  # same name (two-stage)
        )
        p = llama_pick(token, msg)
        r["llama_pick"] = p or "A"
        return r

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(task, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Compute H1.3_con per row
    for r in rows:
        p = r.get("llama_pick", "A")
        if p == "A":
            r["h1p3_name_ok"] = r["base_name_ok"]
            r["h1p3_args_ok"] = r["base_args_ok"]
            r["h1p3_exact_ok"] = r["base_exact_ok"]
        else:
            r["h1p3_name_ok"] = r["h1_name_ok"]
            r["h1p3_args_ok"] = r["h1_args_ok"]
            r["h1p3_exact_ok"] = r["h1_exact_ok"]
        r["oracle_exact_ok"] = r["base_exact_ok"] or r["h1_exact_ok"]

    sB = aggregate(rows, "base")
    sH1 = aggregate(rows, "h1")
    sH12 = aggregate(rows, "h12")
    sH1p3 = aggregate(rows, "h1p3")
    n = len(rows)
    oracle = sum(1 for r in rows if r["oracle_exact_ok"]) / n
    a_picks = sum(1 for r in rows if r.get("llama_pick") == "A")
    b_picks = sum(1 for r in rows if r.get("llama_pick") == "B")
    # Picker accuracy on disagreement rows
    dis = [r for r in rows if r["base_exact_ok"] != r["h1_exact_ok"]]
    correct = sum(
        1 for r in dis
        if (r["llama_pick"] == "A" and r["base_exact_ok"])
        or (r["llama_pick"] == "B" and r["h1_exact_ok"])
    )

    print(f"\n=== Iter 23 H1.3_con: Llama-70B picker over constrained outputs ===")
    print()
    print_summary("base_con (v6 constrained one-shot)", sB)
    print()
    print_summary("H1_con (two-stage)", sH1)
    print()
    print_summary("H1.2_con (clean-only fallback)", sH12)
    print(f"\n  oracle (max of {{base, H1}} per item): {oracle*100:.2f}%")
    print(f"  picker accuracy on disagreement ({len(dis)}): "
          f"{correct}/{len(dis)} = {correct/max(len(dis),1)*100:.1f}%")
    print(f"  picks: A={a_picks}/{n}  B={b_picks}/{n}")
    print()
    print_summary("H1.3_con (Llama-picked)", sH1p3)

    Path(args.out).write_text(json.dumps({
        "n": n,
        "summaries": {
            "base_con": sB, "h1_con": sH1, "h12_con": sH12,
            "h1p3_con": sH1p3, "oracle": oracle,
        },
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
