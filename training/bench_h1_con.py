# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
# ]
# ///
"""Iter 23 — H1.2 under the (Python-ported) constrained decoder.

Mirrors the browser's `con` mode but in Python so we can A/B without porting
to the browser. The decoder is `training/grammar.py` (faithful port of
`web/grammar.js`).

Pipeline per item:
  baseline_con      : v6 constrained one-shot (full call).
  H1_con            : v6 constrained stage 1 (full call → parse name);
                       v9 with ARGS_HINT_TMPL stage 2 (unconstrained — v9
                       args_only is already JSON-clean).
  H1.2_con          : H1_con, except `clean` domain routes to baseline_con's
                       args (skips v9 stage 2). Domain detected from prompt
                       candidates via `*_vacuum` token signature so it works
                       at inference time without metadata.

Run:
    python training/bench_h1_con.py --n 30           # sanity
    python training/bench_h1_con.py --n 300          # full gate test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from bench_common import (
    aggregate,
    args_match,
    build_args_only_prompt,
    load_test,
    parse_args_only,
    parse_call,
    print_summary,
    score,
)
from grammar import (
    build_schema_constraint,
    constrained_generate,
    extract_candidate_names,
    extract_prompt_schemas,
)


NAME_MODEL = os.environ.get("NAME_MODEL", "lifeart/smart-home-gpt2-v6")
ARGS_MODEL = os.environ.get("ARGS_MODEL", "lifeart/smart-home-gpt2-v9")
REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "tool_registry.json"
)


@torch.no_grad()
def greedy_unconstrained(
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    prompt: str,
    device,
    max_new: int = 96,
) -> str:
    """Plain greedy with brace-balanced early stop (used for stage 2)."""
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


def detect_clean_domain(candidate_names: list[str]) -> bool:
    """Detect SH-clean (vacuum) domain via candidate-list signature.

    Works at inference time without the test-set `domain` metadata.
    """
    return any("vacuum" in n for n in candidate_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", default="iter23_h1_con_results.json")
    ap.add_argument(
        "--no-wide-names",
        action="store_true",
        help="Disable wide_names (Iter 9.2). Default ON.",
    )
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")
    wide = not args.no_wide_names
    print(f"[cfg] wide_names={wide} typed_args=True top_k={args.top_k}")

    test = load_test()
    if args.n and args.n < len(test):
        test = test[: args.n]
    print(f"[test] {len(test)} items")

    print(f"[reg] {REGISTRY_PATH}")
    registry = json.loads(REGISTRY_PATH.read_text())

    print(f"[load] {NAME_MODEL}")
    name_tok = GPT2TokenizerFast.from_pretrained(NAME_MODEL)
    name_tok.pad_token = name_tok.eos_token
    name_model = GPT2LMHeadModel.from_pretrained(NAME_MODEL).to(device).eval()
    if ARGS_MODEL == NAME_MODEL:
        args_tok, args_model = name_tok, name_model
    else:
        print(f"[load] {ARGS_MODEL}")
        args_tok = GPT2TokenizerFast.from_pretrained(ARGS_MODEL)
        args_tok.pad_token = args_tok.eos_token
        args_model = GPT2LMHeadModel.from_pretrained(ARGS_MODEL).to(device).eval()

    rows: list[dict] = []
    t0 = time.time()

    for i, s in enumerate(test):
        prompt = s["prompt"]
        domain_label = s.get("domain", "?")
        cand_names = extract_candidate_names(prompt)
        prompt_schemas = extract_prompt_schemas(prompt)
        c = build_schema_constraint(
            cand_names, registry,
            prompt_schemas=prompt_schemas,
            typed_args=True,
            wide_names=wide,
        )
        is_clean = detect_clean_domain(cand_names)

        # --- baseline_con: v6 constrained one-shot ---
        text_b = constrained_generate(
            name_model, name_tok, prompt, c, device, top_k=args.top_k,
        )
        call_b = parse_call(text_b)
        sb = score(
            call_b["name"] if call_b else None,
            call_b["arguments"] if call_b else {},
            s["gold"],
        )

        # --- H1_con: v6 constrained stage 1 → v9 unconstrained stage 2 ---
        pred_name = call_b["name"] if call_b else None
        if pred_name:
            args_prompt = build_args_only_prompt(prompt, pred_name)
            text_h1 = greedy_unconstrained(args_model, args_tok, args_prompt, device)
            pred_args = parse_args_only(text_h1)
            if pred_args is None:
                call2 = parse_call(text_h1)
                pred_args = call2["arguments"] if call2 else {}
        else:
            text_h1 = ""
            pred_args = {}
        sh1 = score(pred_name, pred_args, s["gold"])

        # --- H1.2_con: clean → use baseline_con args, else → H1_con ---
        if is_clean and pred_name:
            h12_args = call_b["arguments"] if call_b else {}
        else:
            h12_args = pred_args
        sh12 = score(pred_name, h12_args, s["gold"])

        rows.append({
            "i": i, "domain": domain_label, "gold": s["gold"],
            "is_clean_detected": is_clean,
            "base_pred_name": pred_name,
            "base_pred_args": call_b["arguments"] if call_b else {},
            "base_name_ok": sb["name_ok"], "base_args_ok": sb["args_ok"],
            "base_exact_ok": sb["exact_ok"],
            "h1_pred_args": pred_args,
            "h1_name_ok": sh1["name_ok"], "h1_args_ok": sh1["args_ok"],
            "h1_exact_ok": sh1["exact_ok"],
            "h12_pred_args": h12_args,
            "h12_name_ok": sh12["name_ok"], "h12_args_ok": sh12["args_ok"],
            "h12_exact_ok": sh12["exact_ok"],
        })

        if (i + 1) % 10 == 0 or i + 1 == len(test):
            b = sum(1 for r in rows if r["base_exact_ok"])
            h = sum(1 for r in rows if r["h1_exact_ok"])
            g = sum(1 for r in rows if r["h12_exact_ok"])
            print(
                f"  [{i+1}/{len(test)}] base_con={b/(i+1)*100:.1f}% "
                f"H1_con={h/(i+1)*100:.1f}% H1.2_con={g/(i+1)*100:.1f}% "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    sB = aggregate(rows, "base")
    sH1 = aggregate(rows, "h1")
    sH12 = aggregate(rows, "h12")
    print(f"\n=== Iter 23 H1.2 with constrained decoder ===")
    print(f"  Test: {len(test)} items, elapsed {time.time()-t0:.0f}s, device={device}")
    print()
    print_summary(f"baseline_con (v6 constrained one-shot)", sB)
    print()
    print_summary(f"H1_con (v6-con name + v9 args)", sH1)
    print()
    print_summary(f"H1.2_con (clean→base, else→H1)", sH12)

    # Detection precision/recall
    label_clean = sum(1 for r in rows if r["domain"] == "clean")
    detected_clean = sum(1 for r in rows if r["is_clean_detected"])
    correct_detected = sum(
        1 for r in rows
        if r["is_clean_detected"] and r["domain"] == "clean"
    )
    if label_clean:
        print(f"\n  clean detection: precision="
              f"{correct_detected/max(detected_clean,1)*100:.0f}%  "
              f"recall={correct_detected/label_clean*100:.0f}%  "
              f"({correct_detected}/{detected_clean} det, "
              f"{correct_detected}/{label_clean} actual)")

    Path(args.out).write_text(json.dumps({
        "name_model": NAME_MODEL, "args_model": ARGS_MODEL,
        "n": len(test), "elapsed_s": time.time() - t0,
        "top_k": args.top_k, "wide_names": wide,
        "baseline_con_summary": sB,
        "h1_con_summary": sH1,
        "h12_con_summary": sH12,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
