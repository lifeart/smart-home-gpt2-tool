"""Refine v6 SFT labels via Llama-3.3-70B-Instruct verifier (HF router, FREE Groq backend).

Iter 17 — verify each training row's (name, args) is correct given the user query and
candidate function list. Mark KEEP / FIX / DROP. For FIX, capture corrected args.

Usage:
    # Pilot (200 items)
    python training/refine_labels.py --in data/sh_train_v6.json --out data/sh_train_v6r_pilot.json --target 200 --concurrency 8 --seed 42
    # Full
    python training/refine_labels.py --in data/sh_train_v6.json --out data/sh_train_v6r.json --concurrency 12

Reads HF_TOKEN from env or ~/.cache/huggingface/token.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_FALLBACKS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]


# ---------------- prompt parser ----------------


def parse_row(prompt: str) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """Return (candidate_names, user_query, raw_candidates_block)."""
    a_idx = prompt.rfind("\n\n\nASSISTANT:")
    if a_idx < 0:
        return None, None, None
    u_idx = prompt.rfind("\n\n\nUSER:", 0, a_idx)
    if u_idx < 0:
        return None, None, None
    cand_start = prompt.find("Use them if required -\n")
    if cand_start < 0:
        return None, None, None
    cand_start += len("Use them if required -\n")
    cand_block = prompt[cand_start:u_idx].strip()
    user_query = prompt[u_idx + len("\n\n\nUSER:"):a_idx].strip()
    names: List[str] = []
    try:
        parsed = json.loads(cand_block)
        if isinstance(parsed, list):
            for c in parsed:
                if isinstance(c, str):
                    names.append(c)
                elif isinstance(c, dict) and "name" in c:
                    names.append(str(c["name"]))
    except Exception:
        pass
    if not names:
        # Fallback: regex
        names = re.findall(
            r'"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"', cand_block
        )
    return (names or None), (user_query or None), cand_block


# ---------------- verifier prompt ----------------


VERIFIER_SYSTEM = (
    "You validate annotated tool-call training items. Reply only in the required "
    "structured format. Be strict but fair: only mark FIX if the annotator's "
    "args are demonstrably wrong, and DROP if the query truly doesn't match any "
    "candidate. Most items should be KEEP."
)


def build_verifier_prompt(
    user_query: str,
    candidate_names: List[str],
    gold_name: str,
    gold_args: Any,
) -> str:
    cand_str = json.dumps(candidate_names)
    gold_args_str = json.dumps(gold_args, separators=(",", ":"))
    return (
        "You are validating a tool-call training item.\n\n"
        f"User said: \"{user_query}\"\n"
        f"Candidate functions: {cand_str}\n"
        f"Annotator picked: {gold_name}\n"
        f"Annotator's arguments JSON: {gold_args_str}\n\n"
        "Tasks:\n"
        "1. Is the function name correct given the query and candidates? (yes/no)\n"
        "2. Are the arguments correctly extracted from the query? (yes/no, with reason if no)\n"
        "3. If arguments are wrong but the name is right, output the corrected arguments JSON.\n"
        "4. If the query doesn't justify ANY listed function (off-topic), output DROP.\n\n"
        "Notes:\n"
        "- IRRELEVANCE TRAINING: when gold name is 'none' and arguments are {}, the annotator "
        "is intentionally teaching the model to refuse off-topic queries. If the user query is "
        "indeed unrelated to ALL listed candidates, mark NAME_OK=yes, ARGS_OK=yes, ACTION=KEEP. "
        "Only mark ACTION=DROP for 'none' rows if some candidate actually fits the query.\n"
        "- Don't penalize argument schemas just because a key isn't mentioned literally — common "
        "sense inference (e.g. 'the front door' -> door='front') is fine.\n"
        "- Arguments may be empty {} if the function takes no parameters.\n"
        "- If the user query is somewhat plausible for the picked function (even if imperfect), "
        "prefer KEEP over DROP. Only DROP when the query truly cannot justify any candidate.\n"
        "- CRITICAL: Do NOT propose a FIX unless the user query explicitly contradicts an annotator "
        "argument value. If the query is ambiguous or doesn't mention a slot value, KEEP the "
        "annotator's value (it's likely a reasonable canonical default). Examples requiring FIX: "
        "'set to 23 degrees' but annotator wrote 20 (digit contradiction); 'lock front door' but "
        "annotator wrote 'back' (entity contradiction). Examples NOT requiring FIX: 'turn lights on' "
        "with no room specified (annotator's room='living_room' is a fine default).\n"
        "- CRITICAL: When proposing FIX, keep the SAME argument keys as the annotator. Do not add "
        "new keys, rename keys, or remove keys — only change VALUES of existing keys.\n\n"
        "Reply in this EXACT format (no markdown fences, no extra commentary):\n"
        "NAME_OK: yes|no\n"
        "ARGS_OK: yes|no\n"
        "ACTION: KEEP|FIX|DROP\n"
        "FIX_JSON: <json or empty>\n"
        "REASON: <one-line>"
    )


# ---------------- response parser ----------------


VERIFIER_RE = re.compile(
    r"NAME_OK:\s*(\S+).*?ARGS_OK:\s*(\S+).*?ACTION:\s*(\S+).*?FIX_JSON:\s*(.*?)\s*(?:\n|$)REASON:\s*(.*?)$",
    re.DOTALL | re.IGNORECASE,
)


def parse_verifier_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Strip code fences if any
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1]
        t = t.rstrip("` \n")
    # Pull lines
    name_ok = args_ok = action = fix_json = reason = None
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("name_ok:"):
            name_ok = s.split(":", 1)[1].strip().lower()
        elif s.lower().startswith("args_ok:"):
            args_ok = s.split(":", 1)[1].strip().lower()
        elif s.lower().startswith("action:"):
            action = s.split(":", 1)[1].strip().upper()
        elif s.lower().startswith("fix_json:"):
            fix_json = s.split(":", 1)[1].strip()
        elif s.lower().startswith("reason:"):
            reason = s.split(":", 1)[1].strip()
    if action is None:
        return None
    # Normalize
    if action not in {"KEEP", "FIX", "DROP"}:
        # Accept partial matches
        ax = action.upper()
        for cand in ("KEEP", "FIX", "DROP"):
            if cand in ax:
                action = cand
                break
        else:
            return None
    fix_obj: Optional[Dict[str, Any]] = None
    if action == "FIX" and fix_json:
        fj = fix_json.strip()
        if fj.lower() not in ("", "empty", "null", "none"):
            # Extract leading {...} blob
            start = fj.find("{")
            if start >= 0:
                depth = 0
                in_str = False
                esc = False
                end = -1
                for i in range(start, len(fj)):
                    ch = fj[i]
                    if esc:
                        esc = False
                        continue
                    if ch == "\\" and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > 0:
                    try:
                        fix_obj = json.loads(fj[start:end])
                    except Exception:
                        fix_obj = None
    return {
        "name_ok": name_ok == "yes",
        "args_ok": args_ok == "yes",
        "action": action,
        "fix_args": fix_obj,
        "reason": (reason or "").strip(),
    }


# ---------------- HF inference ----------------


def hf_chat(
    token: str,
    user_msg: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 400,
    timeout: int = 90,
) -> Optional[str]:
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
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException:
        return None
    if r.status_code != 200:
        # Surface for caller log
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def verify_one(
    token: str,
    row: Dict[str, Any],
    models: List[str],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Return enriched record with verification verdict + original row."""
    gold = json.loads(row["gold"])
    gold_name = gold.get("name", "")
    gold_args = gold.get("arguments", {})
    cand_names, user_query, _ = parse_row(row["prompt"])
    if cand_names is None or user_query is None:
        return {"row": row, "verdict": None, "skip_reason": "parse_failed"}
    user_msg = build_verifier_prompt(user_query, cand_names, gold_name, gold_args)
    last_text = None
    for attempt in range(max_retries):
        for model in models:
            text = hf_chat(token, user_msg, model=model, temperature=0.0)
            if text is not None:
                last_text = text
                verdict = parse_verifier_response(text)
                if verdict is not None:
                    return {
                        "row": row,
                        "verdict": verdict,
                        "user_query": user_query,
                        "cand_names": cand_names,
                        "gold_name": gold_name,
                        "gold_args": gold_args,
                        "raw": text,
                    }
        time.sleep(0.5 + attempt)
    return {
        "row": row,
        "verdict": None,
        "skip_reason": "parse_or_http_failed",
        "raw": last_text,
        "user_query": user_query,
        "cand_names": cand_names,
        "gold_name": gold_name,
        "gold_args": gold_args,
    }


# ---------------- driver ----------------


def load_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        p = Path("~/.cache/huggingface/token").expanduser()
        if p.exists():
            token = p.read_text().strip()
    if not token:
        print("ERROR: HF_TOKEN not set and ~/.cache/huggingface/token missing")
        sys.exit(2)
    return token


def heartbeat_loop(stop_flag: List[bool], counters: Dict[str, int], total: int):
    last = time.time()
    while not stop_flag[0]:
        now = time.time()
        if now - last >= 30:
            print(
                f"[hb] processed={counters['done']}/{total} "
                f"KEEP={counters['keep']} FIX={counters['fix']} "
                f"DROP={counters['drop']} SKIP={counters['skip']} "
                f"elapsed={int(now - counters['start_time'])}s",
                flush=True,
            )
            last = now
        time.sleep(5)


def apply_verdict(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (refined_row | None, applied_action).

    Conservative policy (Iter 17, after pilot review):
    - DROP: only if gold name is not in the candidate list (genuine error). Otherwise KEEP.
      Rationale: rows where gold_name isn't in the candidate prompt are unrecoverable training
      noise (model literally cannot emit a name absent from prompt).
    - FIX: disabled by default. Llama-3.3-70B's arg corrections are noisy (~30% useful per pilot
      review). Pass --enable-fix to apply value-only fixes (same-keys requirement).
    - KEEP: pass through unchanged.
    """
    row = record["row"]
    v = record.get("verdict")
    if v is None:
        return dict(row), "skip_keep"
    action = v["action"]
    cand_names = record.get("cand_names") or []
    gold_name = record.get("gold_name", "")
    if action == "DROP":
        # Only drop if gold name truly missing from candidates (unrecoverable)
        if gold_name and gold_name != "none" and gold_name not in cand_names:
            return None, "drop_applied"
        # Otherwise treat as keep (Llama may be over-aggressive on irrelevance rows)
        return dict(row), "drop_softened_to_keep"
    if action == "KEEP":
        return dict(row), "keep"
    # FIX
    fix_args = v.get("fix_args")
    old_args = record.get("gold_args") or {}
    if not isinstance(fix_args, dict) or not isinstance(old_args, dict):
        return dict(row), "fix_no_json"
    if set(fix_args.keys()) != set(old_args.keys()):
        # Key restructure — likely hallucination, keep original
        return dict(row), "fix_keys_diverge"
    if not _FIX_ENABLED:
        return dict(row), "fix_disabled"
    # Apply value-only fix
    gold_old = json.loads(row["gold"])
    gold_new = {"name": gold_old.get("name", ""), "arguments": fix_args}
    new_row = dict(row)
    new_row["gold"] = json.dumps(gold_new, separators=(",", ":"))
    return new_row, "fix_applied"


_FIX_ENABLED: bool = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=0,
                    help="Pilot mode: limit to N items. 0 = full dataset.")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--sample-fixes", type=int, default=10,
                    help="How many FIX samples to print at end")
    ap.add_argument("--start-index", type=int, default=0,
                    help="For resume: skip first N items (full mode)")
    ap.add_argument("--enable-fix", action="store_true",
                    help="Apply value-only arg fixes (default: drop-only mode)")
    args = ap.parse_args()
    global _FIX_ENABLED
    _FIX_ENABLED = bool(args.enable_fix)

    token = load_token()
    models = [args.model] if args.model else MODEL_FALLBACKS

    src = json.loads(Path(args.inp).read_text())
    print(f"[data] loaded {len(src)} rows from {args.inp}", flush=True)
    rng = random.Random(args.seed)

    if args.target > 0 and args.target < len(src):
        # Pilot: stratified-ish — random sample
        items = rng.sample(src, args.target)
        print(f"[plan] pilot mode: {len(items)} items", flush=True)
    else:
        items = src[args.start_index:]
        if args.start_index:
            print(f"[plan] resume from index {args.start_index}: {len(items)} items", flush=True)
        else:
            print(f"[plan] full mode: {len(items)} items", flush=True)

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    counters = {
        "done": 0,
        "keep": 0,
        "fix": 0,
        "drop": 0,
        "skip": 0,
        "start_time": time.time(),
    }
    lock = threading.Lock()
    stop_flag = [False]
    hb_thread = threading.Thread(
        target=heartbeat_loop, args=(stop_flag, counters, len(items)), daemon=True
    )
    hb_thread.start()

    def worker(idx: int, row: Dict[str, Any]):
        rec = verify_one(token, row, models)
        with lock:
            counters["done"] += 1
            v = rec.get("verdict")
            if v is None:
                counters["skip"] += 1
            elif v["action"] == "KEEP":
                counters["keep"] += 1
            elif v["action"] == "FIX":
                counters["fix"] += 1
            elif v["action"] == "DROP":
                counters["drop"] += 1
        results[idx] = rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, i, row) for i, row in enumerate(items)]
        for _ in concurrent.futures.as_completed(futs):
            pass

    stop_flag[0] = True
    elapsed = time.time() - counters["start_time"]
    print(
        f"\n[done] processed={counters['done']}/{len(items)} "
        f"KEEP={counters['keep']} FIX={counters['fix']} "
        f"DROP={counters['drop']} SKIP={counters['skip']} "
        f"elapsed={int(elapsed)}s",
        flush=True,
    )

    # Build refined dataset
    refined: List[Dict[str, Any]] = []
    fix_samples: List[Dict[str, Any]] = []
    drop_samples: List[Dict[str, Any]] = []
    applied_actions: Dict[str, int] = {}
    for rec in results:
        if rec is None:
            continue
        new_row, applied = apply_verdict(rec)
        applied_actions[applied] = applied_actions.get(applied, 0) + 1
        if new_row is not None:
            refined.append(new_row)
        else:
            if len(drop_samples) < 20:
                drop_samples.append({
                    "user_query": rec.get("user_query"),
                    "cand_names": rec.get("cand_names"),
                    "gold_name": rec.get("gold_name"),
                    "gold_args": rec.get("gold_args"),
                    "reason": rec.get("verdict", {}).get("reason") if rec.get("verdict") else None,
                })
        v = rec.get("verdict")
        if v and v["action"] == "FIX" and len(fix_samples) < 40:
            fix_samples.append({
                "user_query": rec.get("user_query"),
                "cand_names": rec.get("cand_names"),
                "gold_name": rec.get("gold_name"),
                "old_args": rec.get("gold_args"),
                "new_args": v.get("fix_args"),
                "reason": v.get("reason"),
                "applied": applied,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2))

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "input": args.inp,
        "input_count": len(src),
        "processed": len(items),
        "kept_after_refine": len(refined),
        "buckets": {
            "KEEP": counters["keep"],
            "FIX": counters["fix"],
            "DROP": counters["drop"],
            "SKIP": counters["skip"],
        },
        "percentages": {
            "KEEP": round(counters["keep"] / max(1, len(items)) * 100, 2),
            "FIX": round(counters["fix"] / max(1, len(items)) * 100, 2),
            "DROP": round(counters["drop"] / max(1, len(items)) * 100, 2),
            "SKIP": round(counters["skip"] / max(1, len(items)) * 100, 2),
        },
        "elapsed_sec": int(elapsed),
        "model": models[0],
        "concurrency": args.concurrency,
        "seed": args.seed,
        "applied_actions": applied_actions,
        "fix_samples": fix_samples[: args.sample_fixes],
        "drop_samples": drop_samples[:10],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n[wrote] refined={out_path} ({len(refined)} rows)", flush=True)
    print(f"[wrote] meta={meta_path}", flush=True)
    print(
        f"[pct] KEEP={meta['percentages']['KEEP']}% "
        f"FIX={meta['percentages']['FIX']}% "
        f"DROP={meta['percentages']['DROP']}% "
        f"SKIP={meta['percentages']['SKIP']}%",
        flush=True,
    )

    # Print sample FIX items
    print(f"\n--- {min(args.sample_fixes, len(fix_samples))} sample FIX items ---", flush=True)
    for i, fs in enumerate(fix_samples[: args.sample_fixes]):
        print(f"\n[FIX {i+1}] {fs['user_query']}")
        print(f"  cands : {fs['cand_names']}")
        print(f"  name  : {fs['gold_name']}")
        print(f"  old   : {fs['old_args']}")
        print(f"  new   : {fs['new_args']}")
        print(f"  reason: {fs['reason']}")

    if drop_samples:
        print(f"\n--- {min(5, len(drop_samples))} sample DROP items ---", flush=True)
        for i, ds in enumerate(drop_samples[:5]):
            print(f"\n[DROP {i+1}] {ds['user_query']}")
            print(f"  cands : {ds['cand_names']}")
            print(f"  name  : {ds['gold_name']}")
            print(f"  reason: {ds['reason']}")


if __name__ == "__main__":
    main()
