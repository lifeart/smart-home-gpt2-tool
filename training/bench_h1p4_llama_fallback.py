# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 25 H1.4_con — Llama-70B direct-emission fallback.

H1.3_con cleared the 60% gate at 61.3% but `misc` domain (n=72, 24% of
test set) is still the limiter at 45.8% exact, with oracle at 54.2% —
8.4 pp of slack the picker hasn't captured. Below the oracle, 33 misc
items have NEITHER base_con NOR H1_con correct; the picker can't pick
its way out of that.

H1.4_con extends the rerank pool by adding a third candidate: a direct
emission from Llama-3.3-70B given the user query + the prompt's
candidate function schemas. Llama is then asked to pick the best of
three. The fallback path: when neither GPT-2 path is right, Llama's
direct emission is a viable third option.

Reads:
  /tmp/iter23_h1_con_results.json  (base_con + H1_con predictions, n=300)

Writes:
  /tmp/iter23_h1p4_results.json

Run:
    python training/bench_h1p4_llama_fallback.py
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


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


# ---------------- Llama direct-emission ----------------

EMIT_SYSTEM = (
    "You are a smart-home tool-call generator. Given the user query, the "
    "list of candidate functions and their schemas, output ONE JSON object "
    "of the form {\"name\":\"<fn>\",\"arguments\":{...}}. The function name "
    "must be from the candidate list. The arguments must follow the schema "
    "of the chosen function. Never include explanation or markdown. Just "
    "output the JSON."
)


def extract_system_schemas_block(prompt: str) -> str:
    """Return the SYSTEM/function-list region of the prompt (for emit context)."""
    m = re.search(r"SYSTEM:.*?(?=\n\n\nUSER:)", prompt, re.DOTALL)
    return m.group(0) if m else ""


def build_emit_msg(user_query: str, sys_block: str) -> str:
    return (
        f"{sys_block}\n\n"
        f'User query: "{user_query}"\n\n'
        "Output the tool call as a single JSON object on one line. "
        "No prose, no markdown."
    )


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def llama_emit(token: str, prompt: str) -> Optional[dict]:
    cand_names, user_query = parse_user_and_candidates(prompt)
    if not cand_names or not user_query:
        return None
    sys_block = extract_system_schemas_block(prompt)
    msg = build_emit_msg(user_query, sys_block)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": EMIT_SYSTEM},
            {"role": "user", "content": msg},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
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
            return parse_call(txt or "")
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


# ---------------- 3-way picker ----------------

PICK3_SYSTEM = (
    "Pick the best of three tool-call options for the same user query. "
    "Reply in EXACT format. If multiple are equally correct, prefer A then "
    "B over C."
)


def build_pick3_msg(user_query, cand_names, a, b, c):
    def fmt(label, call):
        if not call:
            return f"Option {label}: (no valid output)"
        n = call.get("name")
        ar = call.get("arguments") or {}
        return (
            f"Option {label}:\n"
            f"  name: {n!r}\n"
            f"  arguments: {json.dumps(ar, separators=(',', ':'))}"
        )
    return (
        f'User said: "{user_query}"\n'
        f"Candidate functions: {json.dumps(cand_names)}\n\n"
        f"{fmt('A', a)}\n\n"
        f"{fmt('B', b)}\n\n"
        f"{fmt('C', c)}\n\n"
        "Reply EXACTLY:\n"
        "PICK: A|B|C\n"
        "REASON: <one line>"
    )


PICK3_RE = re.compile(r"PICK\s*:\s*([ABC])", re.IGNORECASE)


def llama_pick3(token, msg):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PICK3_SYSTEM},
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
            m = PICK3_RE.search(txt or "")
            return m.group(1).upper() if m else None
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/iter23_h1_con_results.json")
    ap.add_argument("--out", default="/tmp/iter23_h1p4_results.json")
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

    # Step 1: Llama direct emit for every item (parallel).
    t0 = time.time()

    def emit(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        call = llama_emit(token, prompt)
        r["llama_direct"] = call
        return r

    print(f"\n[1/2] Llama direct-emit on {len(rows)} items @ concurrency={args.concurrency}")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(emit, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(rows)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Score llama_direct alone
    llama_correct = 0
    for r in rows:
        c = r.get("llama_direct")
        if not c:
            r["llama_direct_name_ok"] = False
            r["llama_direct_args_ok"] = False
            r["llama_direct_exact_ok"] = False
            continue
        s = score(c.get("name"), c.get("arguments") or {}, r["gold"])
        r["llama_direct_name_ok"] = s["name_ok"]
        r["llama_direct_args_ok"] = s["args_ok"]
        r["llama_direct_exact_ok"] = s["exact_ok"]
        if s["exact_ok"]:
            llama_correct += 1
    print(f"\n  Llama-direct exact alone: {llama_correct/len(rows)*100:.2f}% "
          f"({llama_correct}/{len(rows)})")

    # Step 2: 3-way Llama pick for items where the three options differ.
    print(f"\n[2/2] Llama 3-way pick")
    work = []
    for r in rows:
        a = (r["base_pred_name"], json.dumps(r["base_pred_args"], sort_keys=True))
        b = (r["base_pred_name"], json.dumps(r["h1_pred_args"], sort_keys=True))
        c_ = r.get("llama_direct")
        c = ((c_ or {}).get("name"), json.dumps((c_ or {}).get("arguments", {}), sort_keys=True))
        # Build distinct-option set
        opts = {a: "A"}
        if b not in opts:
            opts[b] = "B"
        if c not in opts:
            opts[c] = "C"
        if len(opts) == 1:
            r["needs_pick"] = False
            r["llama_pick3"] = "A"
            continue
        r["needs_pick"] = True
        work.append(r)
    print(f"  {len(work)} rows need pick (others all-same → default A)")

    def pick(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["llama_pick3"] = "A"
            return r
        a_call = {"name": r["base_pred_name"], "arguments": r["base_pred_args"]}
        b_call = {"name": r["base_pred_name"], "arguments": r["h1_pred_args"]}
        c_call = r.get("llama_direct")
        msg = build_pick3_msg(user_query, cand_names, a_call, b_call, c_call)
        p = llama_pick3(token, msg)
        r["llama_pick3"] = p or "A"
        return r

    t1 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(pick, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t1:.0f}s", flush=True)
    print(f"  done in {time.time()-t1:.0f}s")

    # Compose H1.4 per row
    for r in rows:
        p = r.get("llama_pick3", "A")
        if p == "A":
            r["h1p4_name_ok"] = r["base_name_ok"]
            r["h1p4_args_ok"] = r["base_args_ok"]
            r["h1p4_exact_ok"] = r["base_exact_ok"]
        elif p == "B":
            r["h1p4_name_ok"] = r["h1_name_ok"]
            r["h1p4_args_ok"] = r["h1_args_ok"]
            r["h1p4_exact_ok"] = r["h1_exact_ok"]
        else:  # C
            r["h1p4_name_ok"] = r["llama_direct_name_ok"]
            r["h1p4_args_ok"] = r["llama_direct_args_ok"]
            r["h1p4_exact_ok"] = r["llama_direct_exact_ok"]
        r["oracle3_exact_ok"] = (
            r["base_exact_ok"] or r["h1_exact_ok"] or r["llama_direct_exact_ok"]
        )

    n = len(rows)
    sB = aggregate(rows, "base")
    sH1 = aggregate(rows, "h1")
    sH12 = aggregate(rows, "h12")
    sLD = aggregate(rows, "llama_direct")
    sH4 = aggregate(rows, "h1p4")
    oracle3 = sum(1 for r in rows if r["oracle3_exact_ok"]) / n

    print(f"\n=== Iter 25 H1.4_con: 3-way rerank with Llama direct fallback ===")
    print()
    print_summary("base_con", sB)
    print()
    print_summary("H1_con", sH1)
    print()
    print_summary("H1.2_con (clean-gate)", sH12)
    print()
    print_summary("llama_direct alone", sLD)
    print(f"\n  oracle (best of {{base, H1, llama_direct}} per item): {oracle3*100:.2f}%")
    a_c = sum(1 for r in rows if r.get("llama_pick3") == "A")
    b_c = sum(1 for r in rows if r.get("llama_pick3") == "B")
    c_c = sum(1 for r in rows if r.get("llama_pick3") == "C")
    print(f"  picks: A={a_c}  B={b_c}  C={c_c}")
    print()
    print_summary("H1.4_con (3-way Llama pick)", sH4)

    # Misc-specific
    misc = [r for r in rows if r["domain"] == "misc"]
    if misc:
        m_b = sum(1 for r in misc if r["base_exact_ok"])
        m_h1 = sum(1 for r in misc if r["h1_exact_ok"])
        m_ld = sum(1 for r in misc if r["llama_direct_exact_ok"])
        m_or = sum(1 for r in misc if r["oracle3_exact_ok"])
        m_h4 = sum(1 for r in misc if r["h1p4_exact_ok"])
        print(f"\n  misc (n={len(misc)}): base={m_b}  H1={m_h1}  "
              f"llama_direct={m_ld}  oracle3={m_or}  H1.4={m_h4}")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "summaries": {
            "base_con": sB, "h1_con": sH1, "h12_con": sH12,
            "llama_direct": sLD, "h1p4_con": sH4, "oracle3": oracle3,
        },
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
