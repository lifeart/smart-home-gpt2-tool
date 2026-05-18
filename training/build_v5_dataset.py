# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 13.4 — build sh_train_v5.json = sh_train_v4.json + HF-Inference-synthetic.

Inputs:
  - data/sh_train_v4.json  (18532 rows, v4 base)
  - data/sh_train_synthetic.json  (HF-Inference generated, target ~3000)

Output:
  - data/sh_train_v5.json (~21.5k rows, four fields: prompt/gold/gold_name/domain)
  - push to lifeart/smart-home-sft-v2 as sh_train_v5.json

Dedup is by (gold_name, user_query) exact (user_query extracted from prompt
between "USER: " and "\\n\\n\\nASSISTANT:"). Synthetic rows that collide with
v4 are dropped to avoid trivial bias.
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
V4_PATH = ROOT / "data" / "sh_train_v4.json"
SYN_PATH = ROOT / "data" / "sh_train_synthetic.json"
OUT_PATH = ROOT / "data" / "sh_train_v5.json"
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")


def extract_user_query(prompt: str) -> str:
    try:
        head = prompt.split("USER: ", 1)[1]
        return head.split("\n\n\nASSISTANT:")[0].strip()
    except Exception:
        return prompt[:120]


def main() -> None:
    v4 = json.loads(V4_PATH.read_text())
    syn = json.loads(SYN_PATH.read_text())
    print(f"[in] v4={len(v4)} synthetic={len(syn)}")

    seen: set[tuple[str, str]] = set()
    for r in v4:
        key = (r.get("gold_name") or "", extract_user_query(r.get("prompt", "")))
        seen.add(key)

    kept_syn: list[dict] = []
    dropped_dup = 0
    for r in syn:
        key = (r.get("gold_name") or "", extract_user_query(r.get("prompt", "")))
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        kept_syn.append({
            "prompt": r["prompt"],
            "gold": r["gold"],
            "gold_name": r["gold_name"],
            "domain": r["domain"],
        })
    print(f"[dedup] synthetic kept={len(kept_syn)} dropped_dup={dropped_dup}")

    out = list(v4) + kept_syn
    print(f"[out] total={len(out)}")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    if os.environ.get("HF_TOKEN") or (
        Path("~/.cache/huggingface/token").expanduser().exists()
    ):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_PATH),
                path_in_repo="sh_train_v5.json",
                repo_id=DATA_REPO,
                repo_type="dataset",
                commit_message=(
                    "Iter 13.4 — v5 dataset = v4 + HF-Inference synthetic (Llama-3.3-70B)"
                ),
            )
            print(f"[push] uploaded to https://huggingface.co/datasets/{DATA_REPO}")
        except Exception as e:  # noqa: BLE001
            print(f"[push] failed: {e}")
    else:
        print("[push] skipped (no HF_TOKEN found)")


if __name__ == "__main__":
    main()
