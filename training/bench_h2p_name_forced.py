# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
#   "requests>=2.31",
# ]
# ///
"""Iter 23 H2' — name-forced rerank.

H2 sanity (n=10) showed v5's greedy distribution collapses to a single name
even at T=0.7 top-k=20: 5/10 items produced identical candidates, the other
5 produced same-name args variants. So pure-sampling rerank can't recover
name errors. H2' forces every candidate name from the prompt registry by
prefilling `{"name":"<X>","arguments":` and letting v5 complete the args.

Per item:
  1. Parse candidate function-name list from the prompt's SYSTEM block.
  2. For each candidate name X, build prefill = prompt + `{"name":"X","arguments":`
     and greedy-decode until the JSON object closes. Parse args.
  3. Score each (name, args) tuple with the Llama-70B verifier from H2.
  4. Pick highest-scoring. Tie-break by "is this the greedy-baseline name?"

Reported:
  - baseline: v5 greedy one-shot exact (same as H2 baseline)
  - oracle:   any candidate name correct AND args correct → best possible
  - h2p:      Llama-picked candidate exact

Run:
    python training/bench_h2p_name_forced.py --n 10
    python training/bench_h2p_name_forced.py --n 300
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from bench_common import (
    aggregate,
    args_match,
    load_hf_token,
    load_test,
    parse_call,
    parse_gold,
    print_summary,
    score,
)
from bench_h2_rerank import (
    build_verifier_msg,
    hf_verify,
    parse_user_and_candidates,
)


BASE_MODEL = os.environ.get("BASE_MODEL", "lifeart/smart-home-gpt2-v5")


@torch.no_grad()
def generate_args_after_prefill(
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    prompt: str,
    forced_name: str,
    device,
    max_new: int = 64,
) -> Optional[dict]:
    """Force the model into `{"name":"X","arguments":` and decode args greedily.

    Returns parsed args dict, or None if completion didn't yield valid JSON.
    """
    prefill = f'{{"name":"{forced_name}","arguments":'
    full_prompt = prompt + prefill
    ids = tok.encode(full_prompt, add_special_tokens=False)
    if len(ids) > 950:
        # Trim from the left: keep the prefill suffix attached
        ids = ids[-950:]
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    # We're already inside the JSON object at depth 1 (the outer {).
    brace_depth = 1
    started = True
    for _ in range(max_new):
        if cur.shape[1] >= 1024:
            break
        out = model(cur)
        logits = out.logits[0, -1, :]
        nxt = int(logits.argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        tok_str = tok.decode([nxt])
        for c in tok_str:
            if c == "{":
                brace_depth += 1
            elif c == "}":
                brace_depth -= 1
        if started and brace_depth <= 0:
            break
    new_ids = cur[0, L:].tolist()
    completion = tok.decode(new_ids, skip_special_tokens=True)
    # Reconstruct the full JSON: prefill + completion
    full_json = prefill + completion
    # Trim anything after the matching close brace
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i, c in enumerate(full_json):
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        obj = json.loads(full_json[:end])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    a = obj.get("arguments")
    if isinstance(a, dict):
        return a
    return None


@torch.no_grad()
def generate_greedy_full(
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    prompt: str,
    device,
    max_new: int = 96,
) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    newline = tok.encode("\n", add_special_tokens=False)[0]
    brace_depth = 0
    started = False
    for _ in range(max_new):
        if cur.shape[1] >= 1024:
            break
        out = model(cur)
        logits = out.logits[0, -1, :]
        nxt = int(logits.argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        tok_str = tok.decode([nxt])
        for c in tok_str:
            if c == "{":
                brace_depth += 1
                started = True
            elif c == "}":
                brace_depth -= 1
        if started and brace_depth <= 0:
            break
        if nxt == newline and not started:
            break
    new_ids = cur[0, L:].tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=8,
                    help="Cap candidate names per item to limit cost.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="iter23_h2p_results.json")
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN missing")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    test = load_test()
    if args.n and args.n < len(test):
        test = test[: args.n]
    print(f"[test] {len(test)} items")

    print(f"[load] {BASE_MODEL}")
    tok = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device).eval()

    rows: list[dict] = []
    t0 = time.time()

    for i, s in enumerate(test):
        prompt = s["prompt"]
        gold = parse_gold(s["gold"])
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names:
            cand_names = []

        # Limit cost
        if len(cand_names) > args.max_candidates:
            cand_names = cand_names[: args.max_candidates]

        # 1. Greedy baseline (also the "natural" name pick for tie-breaking)
        base_text = generate_greedy_full(model, tok, prompt, device)
        base_call = parse_call(base_text)
        base_name = base_call["name"] if base_call else None
        sb = score(
            base_name,
            base_call["arguments"] if base_call else {},
            s["gold"],
        )

        # 2. Force each candidate name, decode args greedily
        forced: list[dict] = []
        for nm in cand_names:
            a = generate_args_after_prefill(model, tok, prompt, nm, device)
            if a is None:
                a = {}
            forced.append({
                "name": nm,
                "args": a,
                "is_greedy_name": (nm == base_name),
            })

        # If baseline picked a name NOT in the candidate list (rare), add it too
        if base_name and base_name not in [f["name"] for f in forced]:
            forced.insert(0, {
                "name": base_name,
                "args": base_call["arguments"] if base_call else {},
                "is_greedy_name": True,
            })

        # 3. Verify each with Llama-70B
        def verify_one(entry: dict) -> dict:
            msg = build_verifier_msg(user_query or "", cand_names, entry["name"], entry["args"])
            v = hf_verify(token, msg)
            entry["verdict"] = v
            if v is None:
                entry["score"] = 0
            else:
                n_ok, a_ok = v
                entry["score"] = (2 if n_ok else 0) + (1 if a_ok else 0)
            return entry

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            forced = list(pool.map(verify_one, forced))

        # 4. Pick: highest verifier score, tie-break = greedy name preferred
        ranked = sorted(
            forced,
            key=lambda e: (-e["score"], 0 if e["is_greedy_name"] else 1),
        )
        pick = ranked[0] if ranked else None

        # Oracle: any candidate (name + args) matches gold?
        oracle_hit = any(
            e["name"] == gold["name"] and args_match(e["args"], gold["arguments"])
            for e in forced
        )

        # H2' = picked
        sh = (
            score(pick["name"], pick["args"], s["gold"])
            if pick
            else {"name_ok": False, "args_ok": False, "exact_ok": False}
        )

        rows.append({
            "i": i,
            "domain": s.get("domain", "?"),
            "gold": s["gold"],
            "candidates": [
                {
                    "name": e["name"],
                    "args": e["args"],
                    "is_greedy_name": e["is_greedy_name"],
                    "verdict": e["verdict"],
                    "score": e["score"],
                }
                for e in forced
            ],
            "n_candidates": len(forced),
            "base_name_ok": sb["name_ok"],
            "base_args_ok": sb["args_ok"],
            "base_exact_ok": sb["exact_ok"],
            "oracle_exact_ok": oracle_hit,
            "h2p_name_ok": sh["name_ok"],
            "h2p_args_ok": sh["args_ok"],
            "h2p_exact_ok": sh["exact_ok"],
        })

        if (i + 1) % 5 == 0 or i + 1 == len(test):
            b = sum(1 for r in rows if r["base_exact_ok"])
            o = sum(1 for r in rows if r["oracle_exact_ok"])
            h = sum(1 for r in rows if r["h2p_exact_ok"])
            print(
                f"  [{i+1}/{len(test)}] "
                f"base={b/(i+1)*100:.1f}% oracle={o/(i+1)*100:.1f}% "
                f"H2'={h/(i+1)*100:.1f}% t={time.time()-t0:.0f}s",
                flush=True,
            )

    sB = aggregate(rows, "base")
    sO_exact = sum(1 for r in rows if r["oracle_exact_ok"]) / max(len(rows), 1)
    sH = aggregate(rows, "h2p")
    print(f"\n=== Iter 23 H2': name-forced rerank ({BASE_MODEL}) ===")
    print(f"  Test set: {len(test)} items, elapsed {time.time()-t0:.0f}s")
    print()
    print_summary("baseline (greedy)", sB)
    print(f"\n  oracle (any forced-name candidate matches): {sO_exact*100:.1f}%")
    print()
    print_summary("H2' (name-forced + Llama rerank)", sH)

    Path(args.out).write_text(json.dumps({
        "base_model": BASE_MODEL,
        "n": len(test),
        "elapsed_s": time.time() - t0,
        "baseline_summary": sB,
        "oracle_exact_acc": sO_exact,
        "h2p_summary": sH,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
