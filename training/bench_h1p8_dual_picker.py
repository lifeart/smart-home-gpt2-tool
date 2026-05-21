# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 27 — H1.8_con: dual-picker ensemble (Llama-70B + DeepSeek-V3.2).

H1.5_con (Llama picker, 3-way) = 71.0%. Iter 26 attempts to improve via
better prompts / shuffling / more candidates all plateaued near 70-71%.
The picker plateau suggests the Llama-70B picker itself has irreducible
bias, not just prompt issues.

Hypothesis: a *different model family* will make uncorrelated errors.
When two different pickers agree, the pick is very reliable. When they
disagree, the item is genuinely ambiguous and needs a tiebreaker.

Strategy:
  1. Run Llama-3.3-70B picker (already done, in /tmp/iter26_h1p5.json
     as `llama_pick3_v2`).
  2. Run DeepSeek-V3.2 picker with the same prompt.
  3. For each item:
     - Both agree → use that pick (high confidence).
     - Disagree → tiebreaker: prefer the one whose chosen candidate is
       part of the consensus group (if any 2 candidates agree on the
       same (name, args), that's a vote).
     - Still tied → use Llama's pick (default).

Reads /tmp/iter26_h1p5.json.
Writes /tmp/iter27_h1p8.json.
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
from bench_h1p5_picker_v2 import (
    PICKER_SYSTEM_V2, build_picker_msg_v2, REGISTRY_PATH,
)


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
PICKER_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


PICK_RE = re.compile(r"PICK\s*:\s*([A-Z])", re.IGNORECASE)


def deepseek_pick(token, msg, allowed):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": PICKER_MODEL,
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
    ap.add_argument("--in", dest="inp", default="/tmp/iter26_h1p5.json")
    ap.add_argument("--out", default="/tmp/iter27_h1p8.json")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        return
    registry = json.loads(REGISTRY_PATH.read_text())
    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    print(f"[in] {len(rows)} rows. Picker: {PICKER_MODEL}")

    # Step 1: DeepSeek pick (3-way over {base, H1, llama_direct})
    work = []
    for r in rows:
        a = (r["base_pred_name"], json.dumps(r["base_pred_args"], sort_keys=True))
        b = (r["base_pred_name"], json.dumps(r["h1_pred_args"], sort_keys=True))
        c_ = r.get("llama_direct")
        c = ((c_ or {}).get("name"), json.dumps((c_ or {}).get("arguments", {}), sort_keys=True))
        if len({a, b, c}) == 1:
            r["ds_pick"] = "A"
            continue
        work.append(r)
    print(f"[work] {len(work)} rows need pick")

    def pick(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["ds_pick"] = "A"
            return r
        a_call = {"name": r["base_pred_name"], "arguments": r["base_pred_args"]}
        b_call = {"name": r["base_pred_name"], "arguments": r["h1_pred_args"]}
        c_call = r.get("llama_direct")
        msg = build_picker_msg_v2(
            user_query, cand_names,
            [("A", a_call), ("B", b_call), ("C", c_call)],
            registry,
        )
        p = deepseek_pick(token, msg, {"A", "B", "C"})
        r["ds_pick"] = p or "A"
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

    # Score DeepSeek pick alone
    ds_correct = 0
    llama_correct = 0
    agree = 0
    ensemble_correct = 0
    for r in rows:
        score_map = {
            "A": r["base_exact_ok"],
            "B": r["h1_exact_ok"],
            "C": r["llama_direct_exact_ok"],
        }
        ds_p = r.get("ds_pick", "A")
        llama_p = r.get("llama_pick3_v2", "A")

        # Score individual
        r["ds_exact_ok"] = score_map[ds_p]
        r["llama_pick_exact_ok"] = score_map[llama_p]
        if r["ds_exact_ok"]:
            ds_correct += 1
        if r["llama_pick_exact_ok"]:
            llama_correct += 1
        if ds_p == llama_p:
            agree += 1

        # Ensemble: if pickers agree, use that. Else: tiebreak by "is there
        # consensus among the 3 candidates?" Find the 2-or-more candidate group
        # if any; pick from it.
        if ds_p == llama_p:
            ens = ds_p
        else:
            # Find consensus (any pair of base/H1/llama_direct agreeing)
            a_key = (r["base_pred_name"], json.dumps(r["base_pred_args"], sort_keys=True))
            b_key = (r["base_pred_name"], json.dumps(r["h1_pred_args"], sort_keys=True))
            c_ = r.get("llama_direct")
            c_key = ((c_ or {}).get("name"), json.dumps((c_ or {}).get("arguments", {}), sort_keys=True))
            keys_to_label = [(a_key, "A"), (b_key, "B"), (c_key, "C")]
            # Find which label of the pickers' two picks (ds_p, llama_p)
            # is part of a consensus group.
            consensus_labels = set()
            for i in range(3):
                for j in range(i+1, 3):
                    if keys_to_label[i][0] == keys_to_label[j][0]:
                        consensus_labels.add(keys_to_label[i][1])
                        consensus_labels.add(keys_to_label[j][1])
            if ds_p in consensus_labels and llama_p not in consensus_labels:
                ens = ds_p
            elif llama_p in consensus_labels and ds_p not in consensus_labels:
                ens = llama_p
            else:
                # Both or neither — default to Llama (the established picker)
                ens = llama_p
        r["ensemble_pick"] = ens
        r["h1p8_exact_ok"] = score_map[ens]
        r["h1p8_name_ok"] = score_map[ens]  # placeholder
        if r["h1p8_exact_ok"]:
            ensemble_correct += 1

    n = len(rows)
    h15_correct = sum(1 for r in rows if r["h1p5_exact_ok"])
    print(f"\n=== Iter 27 H1.8_con: Llama+DeepSeek dual-picker ensemble ===")
    print(f"  DeepSeek alone:  {ds_correct}/{n} = {ds_correct/n*100:.2f}%")
    print(f"  Llama alone:     {llama_correct}/{n} = {llama_correct/n*100:.2f}%")
    print(f"  Agree:           {agree}/{n} = {agree/n*100:.2f}%")
    print(f"  H1.5_con (Llama): {h15_correct}/{n} = {h15_correct/n*100:.2f}%")
    print(f"  H1.8_con ens:    {ensemble_correct}/{n} = {ensemble_correct/n*100:.2f}%")

    oracle3 = sum(1 for r in rows if r["oracle3_exact_ok"]) / n
    print(f"  oracle3:         {oracle3*100:.2f}%")

    # When they disagree, who's right more often?
    dis = [r for r in rows if r.get("ds_pick") != r.get("llama_pick3_v2")]
    if dis:
        ds_right = sum(1 for r in dis if r["ds_exact_ok"])
        llama_right = sum(1 for r in dis if r["llama_pick_exact_ok"])
        print(f"\n  On disagreement ({len(dis)}): "
              f"DeepSeek_right={ds_right}  Llama_right={llama_right}")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "ds_alone_acc": ds_correct / n,
        "llama_alone_acc": llama_correct / n,
        "ensemble_acc": ensemble_correct / n,
        "oracle3": oracle3,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
