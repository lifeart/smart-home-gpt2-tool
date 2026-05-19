# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 21.1 — build sh_train_v10.json (selective Granite curriculum).

Refinement of Iter 20 (v9):
  v9 applied 3-way Granite relabel globally — diluted external data's already-rich
  prior and hurt `clean` domain (-34 pp). v10 applies the curriculum ONLY to native
  SH rows (those whose gold_name is in tool_registry.json); HA/Nemotron/ToolACE/xLAM
  rows stay as single full variants.

Sources:
  - lifeart/smart-home-sft-v2 / sh_train_v6r.json (19084 refined rows)
  - lifeart/smart-home-sft-v2 / sh_train_v9_base.json (HA+Nemotron 6979 net additions)
  - data/sh_train_v7_additions_r.json (5282 ToolACE/xLAM/Glaive refined)

Split:
  - SH rows = gold_name in tool_registry.json (123 native SH funcs) — ~12672
  - non-SH rows from v6r (xlam_sh, irrelevant, external_*, iot_aux, etc.) — ~6412

Output:
  SH × 3 (full/name_only/args_only) + non-SH × 1 + HA × 1 + Nemotron × 1 + v7add × 1

Output:
  data/sh_train_v10.json
  Push to lifeart/smart-home-sft-v2/sh_train_v10.json
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "sh_train_v10.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")

ARGS_HINT_TMPL = (
    "Note: The function name will be: {name}. Output the arguments only.\n\n\n"
)


def load_local_or_hf(local_name: str, hf_name: str) -> list[dict]:
    local = ROOT / "data" / local_name
    if local.exists():
        print(f"[load] using local {local}")
        with local.open() as f:
            return json.load(f)
    print(f"[load] fetching {DATA_REPO}/{hf_name}")
    p = hf_hub_download(DATA_REPO, hf_name, repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def parse_gold(gold_str: str) -> tuple[str | None, dict | None]:
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


def make_granite_variants(row: dict) -> list[dict]:
    """3-way relabel: full / name_only / args_only.

    Same recipe as training/relabel_granite.py (Iter 20). We re-implement here so
    v10 is one self-contained script.
    """
    name, args = parse_gold(row["gold"])
    if name is None:
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

    prompt = row["prompt"]
    if "\n\n\nASSISTANT:" in prompt:
        head, tail = prompt.split("\n\n\nASSISTANT:", 1)
        prompt_args = (
            head + "\n" + ARGS_HINT_TMPL.format(name=name) + "ASSISTANT:" + tail
        )
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


def make_full(row: dict) -> dict:
    """Pass-through with task='full' tag."""
    return {**row, "task": "full"}


def main() -> None:
    random.seed(42)

    # Load registry to identify native SH rows
    reg_path = ROOT / "data" / "tool_registry.json"
    with reg_path.open() as f:
        registry = json.load(f)
    sh_names: set[str] = set(registry.keys())
    print(f"[reg] {len(sh_names)} native SH function names")

    # Load sources
    v6r = load_local_or_hf("sh_train_v6r.json", "sh_train_v6r.json")
    v9_base = load_local_or_hf("sh_train_v9_base.json", "sh_train_v9_base.json")
    v7_add = load_local_or_hf(
        "sh_train_v7_additions_r.json", "sh_train_v7_additions_r.json"
    )

    print(f"[in] v6r={len(v6r)} v9_base={len(v9_base)} v7_add={len(v7_add)}")

    # ---- Split v6r into SH vs non-SH ----
    sh_rows = [r for r in v6r if r.get("gold_name", "") in sh_names]
    non_sh_rows = [r for r in v6r if r.get("gold_name", "") not in sh_names]
    print(
        f"[split] v6r SH={len(sh_rows)} non-SH={len(non_sh_rows)}"
        f" (total={len(sh_rows)+len(non_sh_rows)})"
    )

    if len(sh_rows) < 5000:
        raise SystemExit(
            f"[error] SH count {len(sh_rows)} too low — filter likely broken"
        )

    # ---- HA + Nemotron from v9_base ----
    ha_rows = [r for r in v9_base if r.get("domain") == "external_ha"]
    nem_rows = [r for r in v9_base if r.get("domain") == "external_nemotron"]
    print(f"[split] HA={len(ha_rows)} Nemotron={len(nem_rows)}")

    # ---- Apply Granite 3-way to SH only ----
    sh_variants: list[dict] = []
    for r in sh_rows:
        sh_variants.extend(make_granite_variants(r))
    print(f"[granite] {len(sh_rows)} SH -> {len(sh_variants)} variants")

    task_counts = Counter(r.get("task", "?") for r in sh_variants)
    print(f"[granite] task counts: {dict(task_counts)}")

    # ---- Non-SH stays full only ----
    non_sh_full = [make_full(r) for r in non_sh_rows]
    ha_full = [make_full(r) for r in ha_rows]
    nem_full = [make_full(r) for r in nem_rows]
    v7_full = [make_full(r) for r in v7_add]

    # ---- Concat ----
    merged = sh_variants + non_sh_full + ha_full + nem_full + v7_full
    random.shuffle(merged)
    print(f"\n[out] total v10 = {len(merged)}")

    # Stats
    domain_counts = Counter(r.get("domain", "?") for r in merged)
    print("\n=== v10 by domain (top 20) ===")
    for k, v in sorted(domain_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:<28} {v}")

    overall_tasks = Counter(r.get("task", "?") for r in merged)
    print(f"\n=== v10 task mix: {dict(overall_tasks)} ===")
    print(f"\n=== TOTAL {len(merged)} ===")

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # Sample one per task
    print("\n=== samples ===")
    seen_tasks: set[str] = set()
    for r in merged:
        t = r.get("task", "?")
        if t in seen_tasks:
            continue
        seen_tasks.add(t)
        print(f"\n--- task={t} domain={r.get('domain')} ---")
        print("PROMPT-TAIL:", r["prompt"][-260:])
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
                path_in_repo="sh_train_v10.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter21.1: v10 selective Granite ({len(merged)} rows) — "
                    f"SH×3({len(sh_rows)}->{len(sh_variants)}) + "
                    f"non-SH×1({len(non_sh_full)}) + "
                    f"HA×1({len(ha_full)}) + Nemotron×1({len(nem_full)}) + "
                    f"v7add×1({len(v7_full)})"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
