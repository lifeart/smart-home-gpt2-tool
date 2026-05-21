# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 23 H1 — two-stage decode (name-model + args-model).

Pipeline per item:
  1. Run NAME_MODEL on the original prompt, greedy, parse `name` (first JSON
     `"name":"X"` group).
  2. Build args-only prompt by injecting Granite's ARGS_HINT_TMPL with the
     predicted name (training/relabel_granite.py exact format).
  3. Run ARGS_MODEL on the args-only prompt, greedy, parse arguments.
  4. Combine into {name, arguments} and score exact-match vs gold using the
     same tolerant matcher as web/bench.js.

Baselines reported alongside H1 in the same run for clean A/B:
  - baseline_name: NAME_MODEL one-shot (full call, greedy)
  - baseline_args: ARGS_MODEL one-shot (full call, greedy)

H1 is the cross-model combination (name from NAME_MODEL, args from
ARGS_MODEL conditioned on the predicted name). If H1 > both baselines on
exact-match, two-stage is winning.

Default: NAME_MODEL=v6 (best in-domain name acc), ARGS_MODEL=v9 (only model
with explicit args_only Granite training, so the only one that should
reliably condition on the hint).

Run:
    python training/bench_h1_two_stage.py --n 30           # sanity
    python training/bench_h1_two_stage.py --n 300          # full
    NAME_MODEL=lifeart/smart-home-gpt2-v6 \\
    ARGS_MODEL=lifeart/smart-home-gpt2-v9 \\
        python training/bench_h1_two_stage.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from bench_common import (
    aggregate,
    build_args_only_prompt,
    load_test,
    parse_args_only,
    parse_call,
    print_summary,
    score,
)


NAME_MODEL = os.environ.get("NAME_MODEL", "lifeart/smart-home-gpt2-v6")
ARGS_MODEL = os.environ.get("ARGS_MODEL", "lifeart/smart-home-gpt2-v9")


@torch.no_grad()
def generate(
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
    close_brace = tok.encode("}", add_special_tokens=False)[0]
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
        # Track JSON object depth so we stop only at the matched close.
        for c in tok_str:
            if c == "{":
                brace_depth += 1
                started = True
            elif c == "}":
                brace_depth -= 1
        if started and brace_depth <= 0:
            break
        if nxt == newline and not started:
            # Premature newline before any JSON started — stop early.
            break
    new_ids = cur[0, L:].tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def load_model(repo: str, device):
    print(f"[load] {repo}")
    tok = GPT2TokenizerFast.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = full sh_test.json")
    ap.add_argument("--out", default="iter23_h1_results.json")
    ap.add_argument(
        "--fallback-domains",
        default="",
        help="Comma-separated domains to route through NAME_MODEL's args "
        "(skip stage 2). Use to dodge ARGS_MODEL's known per-domain regressions "
        "(e.g. v9 on clean).",
    )
    args = ap.parse_args()
    fallback = {d.strip() for d in args.fallback_domains.split(",") if d.strip()}
    if fallback:
        print(f"[gate] fallback domains (use NAME_MODEL args): {sorted(fallback)}")

    # Device: cuda > mps > cpu
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

    name_tok, name_model = load_model(NAME_MODEL, device)
    if ARGS_MODEL == NAME_MODEL:
        args_tok, args_model = name_tok, name_model
    else:
        args_tok, args_model = load_model(ARGS_MODEL, device)

    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(test):
        row: dict = {"i": i, "domain": s.get("domain", "?"), "gold": s["gold"]}

        # Baseline A: NAME_MODEL one-shot full call
        out_a = generate(name_model, name_tok, s["prompt"], device)
        call_a = parse_call(out_a)
        sa = score(
            call_a["name"] if call_a else None,
            call_a["arguments"] if call_a else {},
            s["gold"],
        )
        row["baseA_text"] = out_a[:200]
        row["baseA_pred_name"] = call_a["name"] if call_a else None
        row["baseA_pred_args"] = call_a["arguments"] if call_a else {}
        row["baseA_name_ok"] = sa["name_ok"]
        row["baseA_args_ok"] = sa["args_ok"]
        row["baseA_exact_ok"] = sa["exact_ok"]

        # Baseline B: ARGS_MODEL one-shot full call (only if different model)
        if ARGS_MODEL == NAME_MODEL:
            row["baseB_text"] = row["baseA_text"]
            row["baseB_pred_name"] = row["baseA_pred_name"]
            row["baseB_pred_args"] = row["baseA_pred_args"]
            row["baseB_name_ok"] = row["baseA_name_ok"]
            row["baseB_args_ok"] = row["baseA_args_ok"]
            row["baseB_exact_ok"] = row["baseA_exact_ok"]
        else:
            out_b = generate(args_model, args_tok, s["prompt"], device)
            call_b = parse_call(out_b)
            sb = score(
                call_b["name"] if call_b else None,
                call_b["arguments"] if call_b else {},
                s["gold"],
            )
            row["baseB_text"] = out_b[:200]
            row["baseB_pred_name"] = call_b["name"] if call_b else None
            row["baseB_pred_args"] = call_b["arguments"] if call_b else {}
            row["baseB_name_ok"] = sb["name_ok"]
            row["baseB_args_ok"] = sb["args_ok"]
            row["baseB_exact_ok"] = sb["exact_ok"]

        # H1: name from NAME_MODEL (stage A's name), args from ARGS_MODEL
        # conditioned on that name via Granite ARGS_HINT_TMPL.
        pred_name = call_a["name"] if call_a else None
        item_domain = s.get("domain", "?")
        if pred_name and item_domain in fallback:
            # Domain gated to NAME_MODEL's args (e.g. clean → v6).
            out_h1 = "(gated: used NAME_MODEL args)"
            pred_args = call_a["arguments"] if call_a else {}
        elif pred_name:
            args_prompt = build_args_only_prompt(s["prompt"], pred_name)
            out_h1 = generate(args_model, args_tok, args_prompt, device)
            pred_args = parse_args_only(out_h1)
            if pred_args is None:
                # Tolerate: maybe the model emitted a full call anyway.
                call_h1 = parse_call(out_h1)
                pred_args = call_h1["arguments"] if call_h1 else {}
        else:
            out_h1 = ""
            pred_args = {}
        sh1 = score(pred_name, pred_args, s["gold"])
        row["h1_text"] = out_h1[:200]
        row["h1_pred_name"] = pred_name
        row["h1_pred_args"] = pred_args
        row["h1_name_ok"] = sh1["name_ok"]
        row["h1_args_ok"] = sh1["args_ok"]
        row["h1_exact_ok"] = sh1["exact_ok"]

        rows.append(row)

        if (i + 1) % 10 == 0 or i + 1 == len(test):
            ec = sum(1 for r in rows if r["h1_exact_ok"])
            ba = sum(1 for r in rows if r["baseA_exact_ok"])
            bb = sum(1 for r in rows if r["baseB_exact_ok"])
            print(
                f"  [{i+1}/{len(test)}] "
                f"baseA={ba/(i+1)*100:.1f}% baseB={bb/(i+1)*100:.1f}% "
                f"H1={ec/(i+1)*100:.1f}% "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    print(f"\n=== Iter 23 H1: {NAME_MODEL} → {ARGS_MODEL} ===")
    print(f"  Test set: {len(test)} items, elapsed {time.time()-t0:.0f}s")
    sA = aggregate(rows, "baseA")
    sB = aggregate(rows, "baseB")
    sH = aggregate(rows, "h1")
    print()
    print_summary(f"baseline A ({NAME_MODEL.split('/')[-1]} one-shot)", sA)
    print()
    print_summary(f"baseline B ({ARGS_MODEL.split('/')[-1]} one-shot)", sB)
    print()
    print_summary(f"H1 two-stage ({NAME_MODEL.split('/')[-1]} name + {ARGS_MODEL.split('/')[-1]} args)", sH)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "name_model": NAME_MODEL,
        "args_model": ARGS_MODEL,
        "n": len(test),
        "elapsed_s": time.time() - t0,
        "baseline_A_summary": sA,
        "baseline_B_summary": sB,
        "h1_summary": sH,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
