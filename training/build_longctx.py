# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 36 — synthesize long-context (1024-4096 token) data for the ctx-4096
model.

The ctx-4096 problem (PLAN.md Iter 35): `sh_train_v9.json` tops out at
~3542 tokens and `sh_test.json` is entirely <=739 tokens, so the upper half
of a 4096 window is barely trained and never measured.

This builder closes both gaps with NO model API — purely by padding the
rich-schema function list of a prompt with extra *distractor* schemas drawn
from the dataset's own 7.4k-schema pool. The user request and gold answer
are untouched; only the SYSTEM function list grows. That is exactly the
real long-context task ("find the right tool among many").

Two ways a prompt becomes long:
  * PAD     — the prompt already lists object schemas: append distractors.
  * REBUILD — the prompt's schema list is string-only or was truncated
              mid-array; if the gold function's schema exists in the pool,
              assemble a fresh prompt (gold schema + distractors + the
              original user request).

Outputs (../data/):
  sh_train_v14_long.json — 78k originals + N synthetic long rows, lengths
                           spread uniformly across [MIN_TOK, MAX_TOK].
  sh_test_long.json      — every rebuildable sh_test row replicated into
                           length buckets (short / 1500 / 2500 / 3500),
                           tagged `bucket`, `n_tok`, and `exact_ok_valid`
                           (True only when the gold function's *original*
                           schema is preserved, so exact-match is fair).

Deterministic (seed 42). Run locally — tokenizer only, no GPU:
    python training/build_longctx.py
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42

N_SYNTH_TRAIN = int(os.environ.get("N_SYNTH_TRAIN", "22000"))
MIN_TOK, MAX_TOK = 1100, 3850          # training synth length window
TRAIN_TOL, TRAIN_CAP = 200, 4000
TEST_BUCKETS = {"1500": 1500, "2500": 2500, "3500": 3500}
TEST_TOL = 150
# Distractors must be modest so length control is fine-grained: a handful of
# pool schemas are 1000-2100 tokens (giant enum lists) and would make every
# add/remove a coarse jump. The gold function's own schema is never capped.
MAX_DISTRACTOR = 350

_DEC = json.JSONDecoder(strict=False)
SYS_HEAD = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n"
)


# --------------------------------------------------------------------------
# prompt <-> (head, schema-array, tail)
# --------------------------------------------------------------------------

def split_prompt(prompt: str) -> tuple[str, list, str] | None:
    """(head, schema_array, tail) or None if the array is unparseable."""
    i = prompt.find("[")
    if i == -1:
        return None
    try:
        arr, end = _DEC.raw_decode(prompt, i)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(arr, list):
        return None
    return prompt[:i], arr, prompt[end:]


def render(head: str, arr: list, tail: str) -> str:
    return head + json.dumps(arr, indent=2) + tail


def is_all_named_dict(arr: list) -> bool:
    return len(arr) > 0 and all(
        isinstance(e, dict) and "name" in e for e in arr
    )


def extract_user(prompt: str) -> str | None:
    m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", prompt, re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def make_prompt(arr: list, user_text: str) -> str:
    """Assemble a canonical rich-schema prompt from scratch."""
    return (
        SYS_HEAD
        + json.dumps(arr, indent=2)
        + f"\n\n\nUSER: {user_text}\n\n\nASSISTANT: <functioncall> "
    )


# --------------------------------------------------------------------------
# distractor pool
# --------------------------------------------------------------------------

def build_pool(rows: list[dict], tok: GPT2TokenizerFast) -> dict[str, tuple]:
    """name -> (schema, in_array_token_cost). The cost re-indents the
    standalone dump by +2 spaces to match an indent=2 array element, plus
    the `,\\n` separator — accurate to ~1 token."""
    pool: dict[str, tuple] = {}
    for r in rows:
        sp = split_prompt(r["prompt"])
        if sp is None:
            continue
        for e in sp[1]:
            if isinstance(e, dict) and "name" in e and e["name"] not in pool:
                body = json.dumps(e, indent=2)
                indented = "\n".join("  " + ln for ln in body.split("\n"))
                cost = len(tok.encode(indented + ",\n", add_special_tokens=False))
                pool[e["name"]] = (e, cost)
    return pool


# --------------------------------------------------------------------------
# fill an array with distractors until it tokenizes to ~target
# --------------------------------------------------------------------------

def fill(
    head: str,
    base_arr: list,
    tail: str,
    distractors: list[tuple],
    tok: GPT2TokenizerFast,
    target: int,
    tol: int,
    rng: random.Random,
) -> tuple[list, int]:
    """Append distractor schemas to `base_arr` until render() tokenizes to
    `target` (+/- tol). Walks a shuffled candidate list and SKIPS any schema
    that would overshoot `target + tol` — since costs run 27..MAX_DISTRACTOR
    there is always a small enough one to close the final gap, so the result
    lands inside the band with no need to add-then-pop. Returns
    (shuffled_array, true_n_tok)."""
    have = {e["name"] for e in base_arr if isinstance(e, dict) and "name" in e}
    cands = [(n, s, c) for (n, s, c) in distractors if n not in have]
    rng.shuffle(cands)

    arr = list(base_arr)
    est = len(tok.encode(render(head, arr, tail), add_special_tokens=False))
    for _name, schema, cost in cands:
        if est >= target:
            break
        if est + cost > target + tol:
            continue  # would overshoot — keep scanning for a smaller one
        arr.append(schema)
        est += cost
    rng.shuffle(arr)

    n_tok = len(tok.encode(render(head, arr, tail), add_special_tokens=False))
    return arr, n_tok


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)
    print("[tok] loading gpt2 tokenizer")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    train = json.loads((DATA / "sh_train_v9.json").read_text())
    test = json.loads((DATA / "sh_test.json").read_text())
    print(f"[data] v9 train={len(train)}  sh_test={len(test)}")

    print("[pool] collecting distractor schemas")
    pool = build_pool(train, tok)
    distractors = [
        (n, s, c) for n, (s, c) in pool.items() if c <= MAX_DISTRACTOR
    ]
    costs = sorted(c for _, _, c in distractors)
    print(
        f"[pool] {len(pool)} schemas total, {len(distractors)} usable as "
        f"distractors (cost <= {MAX_DISTRACTOR})  "
        f"cost min/med/max = {costs[0]}/{costs[len(costs)//2]}/{costs[-1]}"
    )

    # ---- training mix --------------------------------------------------
    sources = [
        r for r in train
        if (sp := split_prompt(r["prompt"])) is not None
        and is_all_named_dict(sp[1])
        and (r.get("gold_name") in {e["name"] for e in sp[1]}
             or r.get("gold_name") == "none")
    ]
    print(f"[train] {len(sources)}/{len(train)} rows usable as long-ctx sources")

    synth: list[dict] = []
    while len(synth) < N_SYNTH_TRAIN:
        src = rng.choice(sources)
        head, arr, tail = split_prompt(src["prompt"])
        target = rng.randint(MIN_TOK, MAX_TOK)
        new_arr, n_tok = fill(
            head, arr, tail, distractors, tok, target, TRAIN_TOL, rng,
        )
        if n_tok <= 1024 or n_tok > TRAIN_CAP + 100:
            continue
        row = dict(src)
        row["prompt"] = render(head, new_arr, tail)
        row["n_tok"] = n_tok
        row["synth_long"] = True
        synth.append(row)
        if len(synth) % 4000 == 0:
            print(f"  [train] {len(synth)}/{N_SYNTH_TRAIN} synthetic rows")

    mix = train + synth
    rng.shuffle(mix)
    (DATA / "sh_train_v14_long.json").write_text(json.dumps(mix))
    sl = sorted(s["n_tok"] for s in synth)
    n = len(sl)
    print(
        f"[train] wrote sh_train_v14_long.json: {len(mix)} rows "
        f"({len(train)} orig + {len(synth)} synth)"
    )
    print(
        f"[train] synth n_tok p10/p50/p90/max = "
        f"{sl[n//10]}/{sl[n//2]}/{sl[9*n//10]}/{sl[-1]}  "
        f">2048={sum(1 for t in sl if t>2048)}  >3072={sum(1 for t in sl if t>3072)}"
    )

    # ---- long test set -------------------------------------------------
    # short bucket: every original row, untouched (baseline).
    # long buckets: PAD if the row already lists the gold's object schema
    #   (exact-match stays fair); otherwise REBUILD from the pool schema
    #   (name-match fair, exact-match flagged invalid).
    long_test: list[dict] = []
    n_pad = n_rebuild = n_skip = 0
    for r in test:
        base_tok = len(tok.encode(r["prompt"], add_special_tokens=False))
        short = dict(r)
        short.update(bucket="short", n_tok=base_tok, exact_ok_valid=True)
        long_test.append(short)

        gold_name = r.get("gold_name")
        sp = split_prompt(r["prompt"])
        user_text = extract_user(r["prompt"])

        if sp is not None and is_all_named_dict(sp[1]) and \
                gold_name in {e["name"] for e in sp[1]}:
            mode, head, base_arr, tail, exact_valid = (
                "pad", sp[0], sp[1], sp[2], True
            )
        elif gold_name in pool and user_text is not None:
            head = SYS_HEAD
            base_arr = [pool[gold_name][0]]
            tail = f"\n\n\nUSER: {user_text}\n\n\nASSISTANT: <functioncall> "
            mode, exact_valid = "rebuild", False
        else:
            n_skip += 1
            continue
        n_pad += mode == "pad"
        n_rebuild += mode == "rebuild"

        for label, target in TEST_BUCKETS.items():
            new_arr, n_tok = fill(
                head, base_arr, tail, distractors, tok,
                target, TEST_TOL, rng,
            )
            row = dict(r)
            row["prompt"] = render(head, new_arr, tail)
            row.update(
                bucket=label, n_tok=n_tok, n_schemas=len(new_arr),
                exact_ok_valid=exact_valid,
            )
            long_test.append(row)

    (DATA / "sh_test_long.json").write_text(json.dumps(long_test, indent=1))
    by_bucket: dict[str, list[int]] = {}
    for r in long_test:
        by_bucket.setdefault(r["bucket"], []).append(r["n_tok"])
    print(
        f"[test] wrote sh_test_long.json: {len(long_test)} rows  "
        f"(pad={n_pad} rebuild={n_rebuild} skip={n_skip})"
    )
    for b in ["short", *TEST_BUCKETS]:
        v = sorted(by_bucket.get(b, []))
        if v:
            print(
                f"  bucket {b:<6} n={len(v):<4} "
                f"n_tok min/med/max = {v[0]}/{v[len(v)//2]}/{v[-1]}"
            )


if __name__ == "__main__":
    main()
