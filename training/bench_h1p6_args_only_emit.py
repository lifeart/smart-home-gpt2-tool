# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Iter 26.2 — H1.6_con: add Llama-70B args-only emission as 4th candidate.

Iter 25 H1.4_con: 3-way pick {base, H1, llama_direct} → 70.3% exact.
Iter 26.1 H1.5_con: improved picker prompt → 71.0% (oracle3 = 77.0%).
The 6pp gap to oracle is hard to close with picker alone; the real lever
is expanding the candidate pool to raise the oracle ceiling.

This iteration adds a 4th candidate: **llama_args_only**. Given the
function name picked by stage 1 (v6 / H1 — they share the name), ask
Llama-70B to emit just the arguments for that function, with the full
schema in-prompt. Hypothesis:
  - Name accuracy is constant (uses H1's picked name).
  - Args accuracy may be better than llama_direct because Llama is
    constrained to the right schema and can focus on extracting values.
  - Different error mode than v9-args (Llama has stronger world knowledge
    and reasoning, weaker in-domain training).

Then 4-way Llama picks among {base, H1, llama_direct, llama_args_only}.

Reads:
  /tmp/iter26_h1p5.json (already has v5 picker results + llama_direct +
  base + H1 predictions for all 300 items)

Writes:
  /tmp/iter26_h1p6.json
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
from bench_h1p5_picker_v2 import (
    PICKER_SYSTEM_V2, _registry_schema_str, build_picker_msg_v2,
    llama_pick_v2, REGISTRY_PATH,
)


ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


ARGS_EMIT_SYSTEM = (
    "You extract arguments for a smart-home function call. Given a user "
    "query, the function name, and the function's declared schema, output "
    "ONE JSON object: {\"arguments\":{...}}. Rules:\n"
    "- Include ONLY arguments mentioned by the user OR that the schema "
    "marks as required.\n"
    "- Do not invent arguments the user did not mention.\n"
    "- Match enum values exactly when the user's wording maps to one.\n"
    "- Use the same value formatting the schema implies (underscores, "
    "case).\n"
    "- Output only the JSON object — no prose, no markdown."
)


def build_args_emit_msg(user_query: str, fn_name: str, schema: str) -> str:
    return (
        f'User query: "{user_query}"\n'
        f"Function name (already chosen for you): {fn_name}\n"
        f"Function schema:\n  {schema}\n\n"
        f"Output only the args:\n{{\"arguments\":{{...}}}}"
    )


def llama_emit_args(token: str, user_query: str, fn_name: str, registry: dict) -> Optional[dict]:
    schema = _registry_schema_str(registry, fn_name)
    msg = build_args_emit_msg(user_query, fn_name, schema)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ARGS_EMIT_SYSTEM},
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
            # The model may emit either {"arguments":{...}} or just {...};
            # try both shapes.
            m = re.search(r"\{[\s\S]*\}", txt or "")
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None
            if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict):
                return obj["arguments"]
            if isinstance(obj, dict):
                return obj
            return None
        if r.status_code in (429, 503):
            time.sleep(2 + 2 * attempt)
            continue
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/iter26_h1p5.json")
    ap.add_argument("--out", default="/tmp/iter26_h1p6.json")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN missing")
        return
    registry = json.loads(REGISTRY_PATH.read_text())
    print(f"[reg] {len(registry)} functions")

    src = json.loads(Path(args.inp).read_text())
    rows = src["rows"]
    test = load_test()
    print(f"[in] {len(rows)} rows")

    # Step 1: emit args-only Llama for every item with a name pick
    t0 = time.time()

    def emit(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        fn_name = r.get("base_pred_name")  # shared by base and H1 (two-stage)
        if not fn_name or not user_query:
            r["llama_args_only"] = None
            return r
        args_dict = llama_emit_args(token, user_query, fn_name, registry)
        if args_dict is None:
            r["llama_args_only"] = None
        else:
            r["llama_args_only"] = {"name": fn_name, "arguments": args_dict}
        return r

    print(f"\n[1/2] Llama args-only emit on {len(rows)} items")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(emit, r) for r in rows]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(rows)}] t={time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    # Score llama_args_only alone
    lao_correct = 0
    for r in rows:
        c = r.get("llama_args_only")
        if not c:
            r["llama_args_only_name_ok"] = False
            r["llama_args_only_args_ok"] = False
            r["llama_args_only_exact_ok"] = False
            continue
        s = score(c.get("name"), c.get("arguments") or {}, r["gold"])
        r["llama_args_only_name_ok"] = s["name_ok"]
        r["llama_args_only_args_ok"] = s["args_ok"]
        r["llama_args_only_exact_ok"] = s["exact_ok"]
        if s["exact_ok"]:
            lao_correct += 1
    print(f"\n  Llama args-only alone: {lao_correct/len(rows)*100:.2f}% exact "
          f"({lao_correct}/{len(rows)})")

    # Oracle of 4: {base, H1, llama_direct, llama_args_only}
    oracle4 = 0
    for r in rows:
        r["oracle4_exact_ok"] = (
            r["base_exact_ok"] or r["h1_exact_ok"]
            or r["llama_direct_exact_ok"] or r["llama_args_only_exact_ok"]
        )
        if r["oracle4_exact_ok"]:
            oracle4 += 1
    print(f"  oracle4 (best of 4): {oracle4/len(rows)*100:.2f}%")

    # Step 2: 4-way Llama pick
    print(f"\n[2/2] 4-way Llama pick (CoT+schema)")

    work = []
    for r in rows:
        a = (r["base_pred_name"], json.dumps(r["base_pred_args"], sort_keys=True))
        b = (r["base_pred_name"], json.dumps(r["h1_pred_args"], sort_keys=True))
        c_ = r.get("llama_direct")
        c = ((c_ or {}).get("name"), json.dumps((c_ or {}).get("arguments", {}), sort_keys=True))
        d_ = r.get("llama_args_only")
        d = ((d_ or {}).get("name"), json.dumps((d_ or {}).get("arguments", {}), sort_keys=True))
        if len({a, b, c, d}) == 1:
            r["llama_pick4"] = "A"
            continue
        work.append(r)
    print(f"  {len(work)} rows need pick")

    def pick4(r):
        i = r["i"]
        prompt = test[i]["prompt"]
        cand_names, user_query = parse_user_and_candidates(prompt)
        if not cand_names or not user_query:
            r["llama_pick4"] = "A"
            return r
        a_call = {"name": r["base_pred_name"], "arguments": r["base_pred_args"]}
        b_call = {"name": r["base_pred_name"], "arguments": r["h1_pred_args"]}
        c_call = r.get("llama_direct")
        d_call = r.get("llama_args_only")
        msg = build_picker_msg_v2(
            user_query, cand_names,
            [("A", a_call), ("B", b_call), ("C", c_call), ("D", d_call)],
            registry,
        )
        p = llama_pick_v2(token, msg, {"A", "B", "C", "D"})
        r["llama_pick4"] = p or "A"
        return r

    t1 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(pick4, r) for r in work]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 30 == 0:
                print(f"  [{done}/{len(work)}] t={time.time()-t1:.0f}s", flush=True)
    print(f"  done in {time.time()-t1:.0f}s")

    # Compose H1.6 per row
    for r in rows:
        p = r.get("llama_pick4", "A")
        m = {
            "A": ("base_name_ok", "base_args_ok", "base_exact_ok"),
            "B": ("h1_name_ok", "h1_args_ok", "h1_exact_ok"),
            "C": ("llama_direct_name_ok", "llama_direct_args_ok", "llama_direct_exact_ok"),
            "D": ("llama_args_only_name_ok", "llama_args_only_args_ok", "llama_args_only_exact_ok"),
        }[p]
        r["h1p6_name_ok"] = r[m[0]]
        r["h1p6_args_ok"] = r[m[1]]
        r["h1p6_exact_ok"] = r[m[2]]

    sH4 = aggregate(rows, "h1p4")
    sH5 = aggregate(rows, "h1p5")
    sLAO = aggregate(rows, "llama_args_only")
    sH6 = aggregate(rows, "h1p6")
    n = len(rows)
    oracle3 = sum(1 for r in rows if r["oracle3_exact_ok"]) / n
    oracle4_acc = oracle4 / n
    pa = sum(1 for r in rows if r.get("llama_pick4") == "A")
    pb = sum(1 for r in rows if r.get("llama_pick4") == "B")
    pc = sum(1 for r in rows if r.get("llama_pick4") == "C")
    pd = sum(1 for r in rows if r.get("llama_pick4") == "D")

    print(f"\n=== Iter 26.2 H1.6_con: 4-way pick with args-only Llama ===")
    print()
    print_summary("H1.4_con (3-way old picker)", sH4)
    print()
    print_summary("H1.5_con (3-way CoT picker)", sH5)
    print()
    print_summary("llama_args_only alone", sLAO)
    print()
    print_summary("H1.6_con (4-way CoT pick)", sH6)
    print(f"\n  oracle3:        {oracle3*100:.2f}%")
    print(f"  oracle4:        {oracle4_acc*100:.2f}%  (Δ vs oracle3: {(oracle4_acc-oracle3)*100:+.2f} pp)")
    print(f"  picks (4-way): A={pa}  B={pb}  C={pc}  D={pd}")

    # Per-domain misc
    misc = [r for r in rows if r["domain"] == "misc"]
    if misc:
        m_h6 = sum(1 for r in misc if r["h1p6_exact_ok"])
        m_or4 = sum(1 for r in misc if r["oracle4_exact_ok"])
        print(f"\n  misc (n={len(misc)}): H1.6={m_h6}  oracle4={m_or4}")

    Path(args.out).write_text(json.dumps({
        "n": n,
        "summaries": {
            "h1p4_con": sH4, "h1p5_con": sH5,
            "llama_args_only": sLAO,
            "h1p6_con": sH6,
            "oracle3": oracle3, "oracle4": oracle4_acc,
        },
        "rows": rows,
    }, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
