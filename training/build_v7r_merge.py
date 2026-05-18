"""Build sh_train_v7r.json = sh_train_v6r.json + refined(v7 - v6).

Iter 17.3 — when both v7 dataset and v6r refined dataset exist, take the rows that are
in v7 but not v6 (the ~5500 Iter-16 additions: ToolACE/xLAM_v7/Glaive), run them through
the Llama-70B refiner, and concatenate with v6r.

Usage:
    python training/build_v7r_merge.py \
        --v6 data/sh_train_v6.json \
        --v7 data/sh_train_v7.json \
        --v6r data/sh_train_v6r.json \
        --out data/sh_train_v7r.json \
        --refined-additions data/sh_train_v7_additions_r.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def row_key(r: dict) -> str:
    return f"{r.get('prompt','')}\x1f{r.get('gold','')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", required=True)
    ap.add_argument("--v7", required=True)
    ap.add_argument("--v6r", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refined-additions", required=True,
                    help="Where to write the refined v7-v6 additions")
    ap.add_argument("--concurrency", type=int, default=15)
    args = ap.parse_args()

    v6 = json.loads(Path(args.v6).read_text())
    v7 = json.loads(Path(args.v7).read_text())
    v6r = json.loads(Path(args.v6r).read_text())

    v6_keys = {row_key(r) for r in v6}
    additions = [r for r in v7 if row_key(r) not in v6_keys]
    print(f"[diff] v7={len(v7)} v6={len(v6)} additions={len(additions)}")

    # Write additions to a temp input file for refine_labels.py
    add_path = Path(args.refined_additions).with_suffix(".raw.json")
    add_path.write_text(json.dumps(additions))
    print(f"[wrote] additions raw: {add_path}")

    # Invoke refine_labels.py on additions
    here = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(here / "refine_labels.py"),
        "--in", str(add_path),
        "--out", args.refined_additions,
        "--concurrency", str(args.concurrency),
        "--seed", "43",
    ]
    print("[run]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"refine_labels.py failed rc={rc}")

    refined_adds = json.loads(Path(args.refined_additions).read_text())
    print(f"[refined] additions kept={len(refined_adds)}/{len(additions)}")

    merged = list(v6r) + refined_adds
    Path(args.out).write_text(json.dumps(merged))
    print(f"[merge] v7r total: {len(merged)} rows -> {args.out}")


if __name__ == "__main__":
    main()
