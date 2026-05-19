# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 22.2 — build sh_train_v11.json (PURE data scaling, no Granite).

Hypothesis: Iter 20 (v9) and Iter 21 (v10) both used Granite relabel and both
regressed in-domain. Iter 22 isolates the question: does *data scaling alone*
(HA + Nemotron added to a strong SH base) help, without any Granite curriculum?

Recipe (simplest):
  v6 base (sh_train_v6.json, ~20600 rows, the v6 train input)
  + HA rows from sh_train_v9_base.json (HA already adapted+deduped vs v6r in Iter 20.1)
  + Nemotron rows from sh_train_v9_base.json
  No 3-way relabel. Every row stays single-variant.

Dedup pass against v6 base (key = (gold_name, query[:200].lower())).

Output:
  data/sh_train_v11.json
  Push to lifeart/smart-home-sft-v2/sh_train_v11.json
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "sh_train_v11.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")


def extract_user_query(prompt: str) -> str:
    try:
        head = prompt.split("USER: ", 1)[1]
        return head.split("\n\n\nASSISTANT:")[0].strip()
    except Exception:  # noqa: BLE001
        return prompt[:120]


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


def main() -> None:
    random.seed(42)

    v6 = load_local_or_hf("sh_train_v6.json", "sh_train_v6.json")
    v9_base = load_local_or_hf("sh_train_v9_base.json", "sh_train_v9_base.json")
    print(f"[in] v6={len(v6)} v9_base={len(v9_base)}")

    ha_rows = [r for r in v9_base if r.get("domain") == "external_ha"]
    nem_rows = [r for r in v9_base if r.get("domain") == "external_nemotron"]
    print(f"[in] HA={len(ha_rows)} Nemotron={len(nem_rows)} (from v9_base)")

    # Dedup HA + Nemotron against v6 by (gold_name, user-query-lowercased[:200])
    # (v9_base already deduped vs v6r; v6 is a superset of refined-base in spirit
    #  but the actual prompts differ — refine kept domain, dropped unrecoverables.
    #  Re-dedup defensively against v6 itself.)
    seen: set[tuple[str, str]] = set()
    for r in v6:
        q = extract_user_query(r["prompt"]).lower()[:200]
        seen.add((r.get("gold_name", ""), q))
    print(f"[dedup] {len(seen)} keys from v6")

    def dedupe(rows: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for r in rows:
            q = extract_user_query(r["prompt"]).lower()[:200]
            key = (r.get("gold_name", ""), q)
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
        return kept

    ha_kept = dedupe(ha_rows)
    nem_kept = dedupe(nem_rows)
    print(f"[dedup] HA {len(ha_rows)}->{len(ha_kept)}  Nemotron {len(nem_rows)}->{len(nem_kept)}")

    # Strip task tag if present (we want single-variant full only)
    def strip_task(r: dict) -> dict:
        if "task" in r:
            r = {k: v for k, v in r.items() if k != "task"}
        return r

    merged = (
        [strip_task(r) for r in v6]
        + [strip_task(r) for r in ha_kept]
        + [strip_task(r) for r in nem_kept]
    )
    random.shuffle(merged)
    print(f"\n[out] total v11 = {len(merged)}")

    domain_counts = Counter(r.get("domain", "?") for r in merged)
    print("\n=== v11 by domain (top 20) ===")
    for k, v in sorted(domain_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:<28} {v}")
    print(f"\n=== TOTAL {len(merged)} ===")

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # 1 sample per source
    print("\n=== samples ===")
    seen_domains: set[str] = set()
    for r in merged:
        d = r.get("domain", "?")
        bucket = (
            "external_ha"
            if d == "external_ha"
            else "external_nemotron"
            if d == "external_nemotron"
            else "sh_base"
        )
        if bucket in seen_domains:
            continue
        seen_domains.add(bucket)
        print(f"\n--- bucket={bucket} domain={d} ---")
        print("PROMPT-TAIL:", r["prompt"][-220:])
        print("GOLD:", r["gold"][:200])
        if len(seen_domains) >= 3:
            break

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_PATH),
                path_in_repo="sh_train_v11.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    f"iter22.2: v11 pure-data-scaling ({len(merged)} rows) — "
                    f"v6({len(v6)}) + HA×1({len(ha_kept)}) + Nemotron×1({len(nem_kept)}), "
                    f"NO Granite"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
