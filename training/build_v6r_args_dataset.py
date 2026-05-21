# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 24 — build args_only training data from v6r (SH only).

v9's Granite curriculum gave us a working args-only stage 2 in H1, but v9
was trained on v6r + HA + Nemotron, which diluted the SH-`clean` schema
prior. H1.2 used a domain gate to recover.

Hypothesis: a *clean v6r-args* model — fine-tune v6 on Granite args_only
data drawn from v6r SH-only — should match or beat v9 on args without the
clean regression, eliminating the H1.2 gate.

This script:
  1. Pulls `sh_train_v6r.json` (19,084 rows, Llama-verified SH data).
  2. Drops rows where the gold function name is missing/`none` (irrelevance
     rows — no args to extract).
  3. Emits ONE args_only variant per remaining row, using Granite's
     `ARGS_HINT_TMPL` (matches `training/relabel_granite.py` byte-exactly).
  4. Pushes `sh_train_v6r_args.json` to lifeart/smart-home-sft-v2.

Run:
    python training/build_v6r_args_dataset.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")
IN_FILE = "sh_train_v6r.json"
OUT_FILE = "sh_train_v6r_args.json"

ARGS_HINT_TMPL = (
    "Note: The function name will be: {name}. Output the arguments only.\n\n\n"
)


def parse_gold(gold_str: str) -> tuple[str | None, dict]:
    try:
        obj = json.loads(gold_str)
    except Exception:
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def make_args_only(row: dict) -> dict | None:
    name, args = parse_gold(row["gold"])
    if name is None:
        return None
    # Skip irrelevance rows (they teach "no function fits" — no args to learn).
    if name.lower() in ("none", "null"):
        return None
    prompt = row["prompt"]
    marker = "\n\n\nASSISTANT:"
    hint = ARGS_HINT_TMPL.format(name=name)
    if marker in prompt:
        head, tail = prompt.split(marker, 1)
        prompt_args = head + "\n" + hint + "ASSISTANT:" + tail
    else:
        prompt_args = prompt + " " + hint
    return {
        "prompt": prompt_args,
        "gold": json.dumps({"arguments": args}, ensure_ascii=False),
        "gold_name": name,
        "domain": row.get("domain", "?"),
        "task": "args_only",
    }


def main() -> None:
    print(f"[in] downloading {DATA_REPO}/{IN_FILE}")
    p = hf_hub_download(DATA_REPO, IN_FILE, repo_type="dataset")
    rows = json.loads(Path(p).read_text())
    print(f"[in] {len(rows)} v6r rows")

    out: list[dict] = []
    skipped_none = 0
    skipped_parse = 0
    for r in rows:
        v = make_args_only(r)
        if v is None:
            name, _ = parse_gold(r["gold"])
            if name is None:
                skipped_parse += 1
            else:
                skipped_none += 1
            continue
        out.append(v)
    print(
        f"[out] {len(out)} args_only rows "
        f"(skipped: parse={skipped_parse}, irrelevance={skipped_none})"
    )

    # Domain distribution
    doms = Counter(r["domain"] for r in out)
    print("[stats] domain distribution:")
    for d, n in sorted(doms.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {d:<22} {n}")

    # Shuffle for training
    import random
    random.seed(42)
    random.shuffle(out)

    out_path = Path("data") / OUT_FILE if Path("data").exists() else Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[wrote] {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # Sample
    print("\n=== sample row ===")
    print("PROMPT-TAIL:", out[0]["prompt"][-300:])
    print("GOLD:", out[0]["gold"])

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=OUT_FILE,
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter24: v6r-args dataset ({len(out)} args_only rows "
                    f"from {len(rows)} v6r SH base)"
                ),
            )
            print(f"[push] -> {DATA_REPO}/{OUT_FILE}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
