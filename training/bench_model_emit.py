# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 28 — generic direct tool-call emitter for any HF-router model.

The picker plateaued at ~71% (oracle3 ~76%, oracle4 ~82%). Picker
iteration is exhausted. The remaining lever is *candidate quality*:
llama_direct alone is only 53.3%. A stronger direct emitter both raises
the oracle and makes the picker's job easier (one obviously-good option).

This script emits tool calls directly from an arbitrary router model and
scores it alone + as an added candidate to the existing pool.

Usage:
    python training/bench_model_emit.py \\
        --model deepseek-ai/DeepSeek-V4-Pro \\
        --in ../results/iter26_h1p6.json \\
        --tag dsv4pro
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
from bench_h1p4_llama_fallback import EMIT_SYSTEM, extract_system_schemas_block, build_emit_msg


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def model_emit(token: str, model: str, prompt: str) -> Optional[dict]:
    cand_names, user_query = parse_user_and_candidates(prompt)
    if not cand_names or not user_query:
        return None
    sys_block = extract_system_schemas_block(prompt)
    msg = build_emit_msg(user_query, sys_block)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EMIT_SYSTEM},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", default="../results/iter26_h1p6.json")
    ap.add_argument("--tag", required=True, help="Short id for this model's columns")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        return
    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    tag = args.tag
    print(f"[in] {len(rows)} rows. Emitter model: {args.model}  tag={tag}")

    t0 = time.time()

    def emit(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        r[f"{tag}_call"] = model_emit(token, args.model, prompt)
        return r

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(emit, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(rows)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Score this model alone
    for r in rows:
        c = r.get(f"{tag}_call")
        if not c:
            r[f"{tag}_name_ok"] = r[f"{tag}_args_ok"] = r[f"{tag}_exact_ok"] = False
            continue
        s = score(c.get("name"), c.get("arguments") or {}, r["gold"])
        r[f"{tag}_name_ok"] = s["name_ok"]
        r[f"{tag}_args_ok"] = s["args_ok"]
        r[f"{tag}_exact_ok"] = s["exact_ok"]

    n = len(rows)
    sM = aggregate(rows, tag)
    print(f"\n=== {args.model} direct-emit alone ===")
    print_summary(tag, sM)

    # Oracle deltas: existing pool is {base, h1, llama_direct, llama_args_only}
    def has(r, k):
        return r.get(f"{k}_exact_ok", False)

    oracle4 = sum(
        1 for r in rows
        if has(r, "base") or has(r, "h1") or has(r, "llama_direct") or has(r, "llama_args_only")
    ) / n
    oracle5 = sum(
        1 for r in rows
        if has(r, "base") or has(r, "h1") or has(r, "llama_direct")
        or has(r, "llama_args_only") or r.get(f"{tag}_exact_ok")
    ) / n
    print(f"\n  oracle4 (without {tag}):  {oracle4*100:.2f}%")
    print(f"  oracle5 (with {tag}):     {oracle5*100:.2f}%  "
          f"(Δ {(oracle5-oracle4)*100:+.2f} pp)")

    # How many NEW correct items does this model uniquely contribute?
    unique_new = sum(
        1 for r in rows
        if r.get(f"{tag}_exact_ok")
        and not (has(r, "base") or has(r, "h1") or has(r, "llama_direct") or has(r, "llama_args_only"))
    )
    print(f"  unique new correct items from {tag}: {unique_new}")

    out = args.out or args.inp.replace(".json", f"_{tag}.json")
    Path(out).write_text(json.dumps({
        "n": n, "model": args.model, "tag": tag,
        f"{tag}_summary": sM,
        "oracle4": oracle4, "oracle5": oracle5,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
