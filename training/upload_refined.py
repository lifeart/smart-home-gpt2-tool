"""Upload refined dataset JSON to lifeart/smart-home-sft-v2.

Usage:
    python training/upload_refined.py --file data/sh_train_v6r.json --remote sh_train_v6r.json --msg "iter17.2: ..."
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DATA_REPO = "lifeart/smart-home-sft-v2"


def load_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        p = Path("~/.cache/huggingface/token").expanduser()
        if p.exists():
            token = p.read_text().strip()
    return token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--msg", required=True)
    args = ap.parse_args()

    token = load_token()
    if not token:
        raise SystemExit("HF_TOKEN missing")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=args.remote,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=args.msg,
    )
    print(f"[push] {args.file} -> {DATA_REPO}/{args.remote}")


if __name__ == "__main__":
    main()
