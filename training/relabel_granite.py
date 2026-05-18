# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 20.1.e — Granite multi-task curriculum relabel.

Recipe (arXiv 2407.00121): each row is reformatted into THREE variants that explicitly
isolate the sub-tasks. Trained together, the model learns name-prediction,
arg-extraction, and full-call as separate skills, lifting args-acc on small LMs.

Variants per input row:
  - full       (weight ~0.50): prompt unchanged, gold = {"name":"X","arguments":{...}}  (original)
  - name_only  (weight ~0.30): prompt unchanged, gold = {"name":"X"}
  - args_only  (weight ~0.20): prompt augmented with
                    "\nNote: The function name will be: X. Output the arguments only."
                immediately before the ASSISTANT marker,
                gold = {"arguments":{...}}

The output dataset is NOT actually weighted at the SFT step (Trainer can't easily do
per-example weights for CE LM loss). Instead, we *sample* each variant with the target
probability so that the resulting row counts reflect the curriculum mix. For ~29k inputs:
  - n_full      = round(0.50 * 3 * 29k / 3) ≈ 14500 full
  - n_name_only = round(0.30 * 3 * 29k / 3) ≈ 8700  name-only
  - n_args_only = round(0.20 * 3 * 29k / 3) ≈ 5800  args-only
Sum ≈ 29000, but the PLAN spec emits "3 variants per input" = ~87k. We implement the
3×-emit interpretation (full/name/args) which is what the Granite paper describes —
weights are an emission-multiplier, not a per-sample loss weight.

We emit ALL THREE variants for every input row, so final size = 3 * N.

For a 29k base, that's ~87k rows. Hyperparams (2 epochs) make total training tokens
~6× v6r-only. We absorb this by keeping seq_len 1024 unchanged.

Input:  data/sh_train_v9_base.json  (concat of v6r + HA + Nemotron)
Output: data/sh_train_v9.json       (3×N rows after relabel)
"""

import json
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "sh_train_v9_base.json"
OUT_PATH = ROOT / "data" / "sh_train_v9.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")

ARGS_HINT_TMPL = (
    "Note: The function name will be: {name}. Output the arguments only.\n\n\n"
)


def parse_gold(gold_str: str) -> tuple[str | None, dict | None]:
    """Parse the gold JSON; tolerate trailing junk."""
    try:
        obj = json.loads(gold_str)
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(obj, dict):
        return None, None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str):
        return None, None
    if not isinstance(args, dict):
        args = {}
    return name, args


def make_variants(row: dict) -> list[dict]:
    name, args = parse_gold(row["gold"])
    if name is None:
        # Gold isn't parseable as {"name":..., "arguments":...} — keep as full only.
        return [{**row, "task": "full"}]

    domain = row.get("domain", "?")

    full = {
        "prompt": row["prompt"],
        "gold": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
        "gold_name": name,
        "domain": domain,
        "task": "full",
    }

    name_only = {
        "prompt": row["prompt"],
        "gold": json.dumps({"name": name}, ensure_ascii=False),
        "gold_name": name,
        "domain": domain,
        "task": "name_only",
    }

    # args_only: inject hint immediately before the ASSISTANT marker
    # Prompt template is:  "...USER: <q>\n\n\nASSISTANT: <functioncall> "
    # Insert the hint between "USER: <q>" and "\n\n\nASSISTANT:".
    prompt = row["prompt"]
    if "\n\n\nASSISTANT:" in prompt:
        head, tail = prompt.split("\n\n\nASSISTANT:", 1)
        prompt_args = head + "\n" + ARGS_HINT_TMPL.format(name=name) + "ASSISTANT:" + tail
    else:
        prompt_args = prompt + " " + ARGS_HINT_TMPL.format(name=name)

    args_only = {
        "prompt": prompt_args,
        "gold": json.dumps({"arguments": args}, ensure_ascii=False),
        "gold_name": name,
        "domain": domain,
        "task": "args_only",
    }

    return [full, name_only, args_only]


def main() -> None:
    if IN_PATH.exists():
        print(f"[in] using local {IN_PATH}")
        rows = json.loads(IN_PATH.read_text())
    else:
        print(f"[in] fetching {DATA_REPO}/sh_train_v9_base.json")
        p = hf_hub_download(
            DATA_REPO, "sh_train_v9_base.json", repo_type="dataset"
        )
        with open(p) as f:
            rows = json.load(f)
    print(f"[in] {len(rows)} base rows")

    out: list[dict] = []
    for r in rows:
        out.extend(make_variants(r))
    print(f"[out] {len(out)} relabeled rows (×{len(out)/max(len(rows),1):.2f})")

    task_counts = Counter(r.get("task", "?") for r in out)
    print(f"[stats] task counts: {dict(task_counts)}")

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(out)

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # Print 1 sample per variant
    print("\n=== samples ===")
    seen_tasks: set[str] = set()
    for r in out:
        if r.get("task") in seen_tasks:
            continue
        seen_tasks.add(r["task"])
        print(f"\n--- task={r['task']} domain={r['domain']} ---")
        print("PROMPT-TAIL:", r["prompt"][-300:])
        print("GOLD:", r["gold"][:200])
        if len(seen_tasks) >= 3:
            break

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_PATH),
                path_in_repo="sh_train_v9.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter20.1: v9 Granite-relabel ({len(out)} rows = "
                    f"3 × {len(rows)} base)"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
