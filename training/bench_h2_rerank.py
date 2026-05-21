# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
#   "requests>=2.31",
# ]
# ///
"""Iter 23 H2 — N-best generation + Llama-3.3-70B reranker.

Per item:
  1. Generate N candidates from BASE_MODEL: 1 greedy + (N-1) temperature
     samples with top-k. Dedupe by canonical JSON.
  2. Each unique candidate is scored by Llama-3.3-70B via HF Inference
     Providers router (Groq backend, free tier) using a verifier prompt
     adapted from refine_labels.py. Score = 2*name_ok + args_ok.
  3. Pick highest-scoring candidate (ties → first). Score winner vs gold.

Reported metrics per run:
  - baseline:   greedy candidate exact-match (= what BASE_MODEL emits alone)
  - oracle:     best-of-N candidate (upper bound — what rerank COULD reach
                if the verifier were perfect)
  - h2_rerank:  Llama-picked candidate exact-match (the real H2 number)
  - verifier_acc: fraction of items where rerank-pick matches oracle

Run:
    python training/bench_h2_rerank.py --n 30           # sanity
    python training/bench_h2_rerank.py --n 300          # full
    BASE_MODEL=lifeart/smart-home-gpt2-v5 \\
        python training/bench_h2_rerank.py --n 30
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

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


BASE_MODEL = os.environ.get("BASE_MODEL", "lifeart/smart-home-gpt2-v5")
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
VERIFIER_MODEL = os.environ.get(
    "VERIFIER_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
)


# ---------------- prompt parser (mirrors refine_labels.py) ----------------

def parse_user_and_candidates(prompt: str) -> tuple[Optional[list[str]], Optional[str]]:
    a_idx = prompt.rfind("\n\n\nASSISTANT:")
    if a_idx < 0:
        return None, None
    u_idx = prompt.rfind("\n\n\nUSER:", 0, a_idx)
    if u_idx < 0:
        return None, None
    cand_start = prompt.find("Use them if required -\n")
    if cand_start < 0:
        return None, None
    cand_start += len("Use them if required -\n")
    cand_block = prompt[cand_start:u_idx].strip()
    user_query = prompt[u_idx + len("\n\n\nUSER:"):a_idx].strip()
    names: list[str] = []
    try:
        parsed = json.loads(cand_block)
        if isinstance(parsed, list):
            for c in parsed:
                if isinstance(c, str):
                    names.append(c)
                elif isinstance(c, dict) and "name" in c:
                    names.append(str(c["name"]))
    except Exception:
        names = re.findall(
            r'"([a-zA-Z_][a-zA-Z0-9_]*)"', cand_block
        )
    return (names or None), (user_query or None)


# ---------------- verifier ----------------

VERIFIER_SYSTEM = (
    "You validate model-emitted tool calls. Reply only in the exact required "
    "format. Be strict: only mark NAME_OK=yes if the function name correctly "
    "addresses the user's intent given the candidates. Only mark ARGS_OK=yes "
    "if every argument value is justified by the user query (canonical "
    "defaults for unmentioned slots are acceptable)."
)


def build_verifier_msg(
    user_query: str,
    cand_names: list[str],
    pred_name: Optional[str],
    pred_args: dict,
) -> str:
    args_str = json.dumps(pred_args, separators=(",", ":"))
    return (
        "You are scoring a model-predicted tool call.\n\n"
        f'User said: "{user_query}"\n'
        f"Candidate functions: {json.dumps(cand_names)}\n"
        f'Model picked function: {pred_name!r}\n'
        f"Model's arguments JSON: {args_str}\n\n"
        "Tasks:\n"
        "1. Is the chosen function name correct? (yes/no)\n"
        "2. Are the arguments correctly extracted from the query? (yes/no)\n\n"
        "Notes:\n"
        "- If the user query doesn't justify ANY listed function (off-topic), "
        "NAME_OK should be 'no' for any non-decline pick.\n"
        "- Canonical defaults for unmentioned slots (e.g. 'turn on lights' "
        "→ room='living_room') are acceptable; don't fail ARGS_OK for those.\n"
        "- Only fail ARGS_OK on contradictions: digits, named entities, "
        "or enum values that mismatch what the user said.\n\n"
        "Reply in this EXACT format (no markdown, no commentary):\n"
        "NAME_OK: yes|no\n"
        "ARGS_OK: yes|no"
    )


VERDICT_RE = re.compile(
    r"NAME_OK\s*:\s*(\S+).*?ARGS_OK\s*:\s*(\S+)", re.DOTALL | re.IGNORECASE
)


def parse_verdict(text: str) -> Optional[tuple[bool, bool]]:
    if not text:
        return None
    m = VERDICT_RE.search(text)
    if not m:
        return None
    n = m.group(1).strip().lower()
    a = m.group(2).strip().lower()
    return (n.startswith("y"), a.startswith("y"))


def hf_verify(
    token: str,
    user_msg: str,
    *,
    model: str = VERIFIER_MODEL,
    timeout: int = 60,
) -> Optional[tuple[bool, bool]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 60,
    }
    for attempt in range(3):
        try:
            r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 200:
            try:
                txt = r.json()["choices"][0]["message"]["content"]
            except Exception:
                return None
            return parse_verdict(txt)
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


# ---------------- generation ----------------

@torch.no_grad()
def generate(
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    prompt: str,
    device,
    *,
    max_new: int = 96,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
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
        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        else:
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


def candidate_key(call: Optional[dict]) -> str:
    if not call:
        return "__NONE__"
    return json.dumps(
        {"n": call.get("name"), "a": call.get("arguments") or {}},
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------- driver ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = full sh_test.json")
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel verifier calls per item")
    ap.add_argument("--out", default="iter23_h2_results.json")
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN not found (env or ~/.cache/huggingface/token)")
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

        # 1. Generate N candidates: 1 greedy + (N-1) sampled with different seeds
        cand_texts: list[str] = []
        cand_texts.append(generate(model, tok, prompt, device, temperature=0.0))
        for k in range(1, args.num_candidates):
            cand_texts.append(generate(
                model, tok, prompt, device,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=1000 * i + k,
            ))

        # 2. Parse, dedupe
        parsed = [parse_call(t) for t in cand_texts]
        unique: dict[str, dict] = {}
        for idx, (txt, call) in enumerate(zip(cand_texts, parsed)):
            k = candidate_key(call)
            if k not in unique:
                unique[k] = {
                    "key": k,
                    "text": txt,
                    "call": call,
                    "is_greedy": (idx == 0),
                }
        unique_list = list(unique.values())

        # 3. Verify each unique candidate with Llama (parallel within item)
        def verify_one(entry: dict) -> dict:
            call = entry["call"]
            if call is None or call.get("name") is None or cand_names is None:
                entry["verdict"] = None
                entry["score"] = -1
                return entry
            msg = build_verifier_msg(
                user_query or "",
                cand_names,
                call["name"],
                call.get("arguments") or {},
            )
            v = hf_verify(token, msg)
            entry["verdict"] = v
            if v is None:
                entry["score"] = 0
            else:
                n_ok, a_ok = v
                entry["score"] = (2 if n_ok else 0) + (1 if a_ok else 0)
            return entry

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            unique_list = list(pool.map(verify_one, unique_list))

        # 4. Pick highest-scoring (ties → first occurrence in original order,
        # which is greedy-first by construction).
        ranked = sorted(
            unique_list,
            key=lambda e: (-e["score"], 0 if e["is_greedy"] else 1),
        )
        pick = ranked[0] if ranked else None

        # Baseline = greedy candidate
        greedy_entry = next((e for e in unique_list if e["is_greedy"]), unique_list[0])
        base_call = greedy_entry["call"]
        sb = score(
            base_call["name"] if base_call else None,
            base_call["arguments"] if base_call else {},
            s["gold"],
        )

        # Oracle = is gold among the N candidates?
        oracle_hit = False
        for entry in unique_list:
            c = entry["call"]
            if not c:
                continue
            if c.get("name") == gold["name"] and args_match(c.get("arguments") or {}, gold["arguments"]):
                oracle_hit = True
                break

        # H2 = rerank pick
        pick_call = pick["call"] if pick else None
        sh = score(
            pick_call["name"] if pick_call else None,
            pick_call["arguments"] if pick_call else {},
            s["gold"],
        )

        rows.append({
            "i": i,
            "domain": s.get("domain", "?"),
            "gold": s["gold"],
            "n_unique": len(unique_list),
            "candidates": [
                {
                    "key": e["key"],
                    "is_greedy": e["is_greedy"],
                    "name": (e["call"] or {}).get("name") if e["call"] else None,
                    "args": (e["call"] or {}).get("arguments") if e["call"] else None,
                    "verdict": e["verdict"],
                    "score": e["score"],
                }
                for e in unique_list
            ],
            "base_name_ok": sb["name_ok"],
            "base_args_ok": sb["args_ok"],
            "base_exact_ok": sb["exact_ok"],
            "oracle_exact_ok": oracle_hit,
            "h2_name_ok": sh["name_ok"],
            "h2_args_ok": sh["args_ok"],
            "h2_exact_ok": sh["exact_ok"],
        })

        if (i + 1) % 5 == 0 or i + 1 == len(test):
            b = sum(1 for r in rows if r["base_exact_ok"])
            o = sum(1 for r in rows if r["oracle_exact_ok"])
            h = sum(1 for r in rows if r["h2_exact_ok"])
            print(
                f"  [{i+1}/{len(test)}] "
                f"base={b/(i+1)*100:.1f}% "
                f"oracle={o/(i+1)*100:.1f}% "
                f"H2={h/(i+1)*100:.1f}% "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    print(f"\n=== Iter 23 H2: rerank {BASE_MODEL} w/ {VERIFIER_MODEL} ===")
    print(f"  Test set: {len(test)} items, elapsed {time.time()-t0:.0f}s")
    sB = aggregate(rows, "base")
    sO_exact = sum(1 for r in rows if r["oracle_exact_ok"]) / len(rows)
    sH = aggregate(rows, "h2")
    print()
    print_summary("baseline (greedy)", sB)
    print(f"\n  oracle best-of-N exact: {sO_exact*100:.1f}%")
    print()
    print_summary("H2 (Llama-70B rerank)", sH)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "base_model": BASE_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "n": len(test),
        "elapsed_s": time.time() - t0,
        "baseline_summary": sB,
        "oracle_exact_acc": sO_exact,
        "h2_summary": sH,
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
