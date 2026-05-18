# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 15.2 — build sh_train_v6.json = sh_train_v5.json + 9 fresh_bench JSONs.

Inputs:
  - data/sh_train_v5.json (20010 items)
  - training/external_data/fresh_*.json (9 files, 690 items)

Outputs:
  - data/sh_train_external.json (~590 items, after 100-item holdout)
  - data/sh_test_external.json (100 items, stratified across 9 categories)
  - data/sh_train_v6.json (~20600 = v5 + external_train)
  - push to lifeart/smart-home-sft-v2 (v6 + holdout)

Transform rules per file type:
  - bio/culture/industrial/materials/nichetech/opus (single fn dict):
      prompt = SYSTEM with [function dict] + USER + ASSISTANT
      gold   = {"name": gold_name, "arguments": gold_args}
      domain = external_<file>
  - multiple (list of fn dicts, gold is one):
      prompt = SYSTEM with full list + USER + ASSISTANT
      gold   = single call
      domain = external_multiple
  - parallel (list of fn dicts, gold_calls is multi-call):
      prompt = SYSTEM with full list + USER + ASSISTANT
      gold   = FIRST call from gold_calls (we model single-call only)
      domain = external_parallel
  - irrelevance (list of fn dicts, gold_name="none"):
      prompt = SYSTEM with full list + USER + ASSISTANT
      gold   = {"name":"none","arguments":{}}
      domain = external_irrelevance
"""

import json
import os
import random
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
V5_PATH = ROOT / "data" / "sh_train_v5.json"
EXT_DIR = ROOT / "training" / "external_data"
OUT_TRAIN = ROOT / "data" / "sh_train_v6.json"
OUT_EXT_TRAIN = ROOT / "data" / "sh_train_external.json"
OUT_EXT_TEST = ROOT / "data" / "sh_test_external.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")

SINGLE_FN_FILES = ["bio", "culture", "industrial", "materials", "nichetech", "opus"]
LIST_FN_FILES = ["multiple", "parallel", "irrelevance"]


def build_prompt(fn_spec, user_query: str) -> str:
    """Build SYSTEM+USER+ASSISTANT prompt with given function spec (dict OR list)."""
    spec_str = json.dumps(fn_spec, indent=2)
    return (
        "SYSTEM: You are a helpful assistant with access to the following functions. "
        "Use them if required -\n"
        f"{spec_str}\n\n\n"
        f"USER: {user_query}\n\n\n"
        "ASSISTANT: <functioncall> "
    )


def transform_item(item: dict, category: str) -> dict | None:
    user = item.get("prompt", "").strip()
    if not user:
        return None
    fn = item.get("function")

    if category == "parallel":
        gold_calls = item.get("gold_calls") or []
        if not gold_calls:
            return None
        first = gold_calls[0]
        gold_name = first.get("name", "")
        gold_args = first.get("arguments", {}) or {}
    elif category == "irrelevance":
        gold_name = "none"
        gold_args = {}
    else:
        gold_name = item.get("gold_name", "")
        gold_args = item.get("gold_args", {}) or {}
        if not gold_name:
            return None

    prompt = build_prompt(fn, user)
    gold = json.dumps({"name": gold_name, "arguments": gold_args}, separators=(",", ":"))
    return {
        "prompt": prompt,
        "gold": gold,
        "gold_name": gold_name,
        "domain": f"external_{category}",
    }


def main() -> None:
    random.seed(42)
    v5 = json.loads(V5_PATH.read_text())
    print(f"[in] v5={len(v5)}")

    # Build per-category lists
    per_cat: dict[str, list[dict]] = {}
    for cat in SINGLE_FN_FILES + LIST_FN_FILES:
        path = EXT_DIR / f"fresh_{cat}.json"
        raw = json.loads(path.read_text())
        rows: list[dict] = []
        for item in raw:
            r = transform_item(item, cat)
            if r is not None:
                rows.append(r)
        random.shuffle(rows)
        per_cat[cat] = rows
        print(f"[load] {cat}: {len(rows)} (from {len(raw)} raw)")

    # Stratified holdout: 100 items / 9 cats ≈ 11 per cat (opus has 50 raw → take 11 too)
    target_per_cat = 100 // 9  # 11
    remainder = 100 - target_per_cat * 9  # 1 extra
    cats_sorted = sorted(per_cat.keys())
    held: list[dict] = []
    train_ext: list[dict] = []
    for i, cat in enumerate(cats_sorted):
        take = target_per_cat + (1 if i < remainder else 0)
        rows = per_cat[cat]
        held.extend(rows[:take])
        train_ext.extend(rows[take:])
    print(f"[split] external train={len(train_ext)} holdout={len(held)}")

    # Verify holdout balance
    from collections import Counter
    held_dom = Counter(r["domain"] for r in held)
    train_dom = Counter(r["domain"] for r in train_ext)
    print(f"[holdout domains] {dict(held_dom)}")
    print(f"[train external domains] {dict(train_dom)}")

    # Write external splits
    OUT_EXT_TRAIN.write_text(json.dumps(train_ext, indent=2))
    OUT_EXT_TEST.write_text(json.dumps(held, indent=2))
    print(f"[wrote] {OUT_EXT_TRAIN} ({OUT_EXT_TRAIN.stat().st_size/1e6:.2f} MB)")
    print(f"[wrote] {OUT_EXT_TEST} ({OUT_EXT_TEST.stat().st_size/1e6:.2f} MB)")

    # v6 = v5 + external train
    v6 = list(v5) + train_ext
    random.shuffle(v6)
    OUT_TRAIN.write_text(json.dumps(v6, indent=2))
    print(f"[wrote] {OUT_TRAIN} ({OUT_TRAIN.stat().st_size/1e6:.1f} MB) total={len(v6)}")

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_TRAIN),
                path_in_repo="sh_train_v6.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    "Iter 15.2 — v6 = v5 + 590 fresh_bench items (universal tool-call mix)"
                ),
            )
            print(f"[push] uploaded sh_train_v6.json")
            api.upload_file(
                path_or_fileobj=str(OUT_EXT_TEST),
                path_in_repo="sh_test_external.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    "Iter 15.2 — external holdout (100 items, stratified across 9 categories)"
                ),
            )
            print(f"[push] uploaded sh_test_external.json")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
