# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "datasets>=2.20",
#   "accelerate>=0.34",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 40 — v15: failure-driven targeted augmentation (accuracy idea B3).

Continues the fine-tune from v9 (NOT base GPT-2) for ONE epoch on
`sh_train_v15.json` = v9's full 78k curriculum + ~2.6k targeted rows
(oversampled 3x) that teach implication-style inference the synthesis
error analysis flagged — "it's freezing" -> heat, "kill the lights" ->
off, "deploy the awning" -> extend.

Additive by design: the targeted rows are ADDED on top of v9's full mix,
never replacing it — Iter 24's v6r-args replaced SH data and regressed
cross-domain. Architecture is unchanged (stock GPT-2 124M, 1024 ctx).

Dataset: lifeart/smart-home-sft-v2 (sh_train_v15.json)
Base:    lifeart/smart-home-gpt2-v9
Target:  lifeart/smart-home-gpt2-v15

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 2h \\
        --detach training/train_hf_v15.py
"""
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "lifeart/smart-home-gpt2-v9")
TARGET_REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2-v15")
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = os.environ.get("DATA_FILE", "sh_train_v15.json")
PAD = 1024
OUT_DIR = Path("out/final")


def download_dataset() -> list[dict]:
    print(f"[data] fetching {DATA_REPO}/{DATA_FILE}")
    p = hf_hub_download(repo_id=DATA_REPO, filename=DATA_FILE, repo_type="dataset")
    with open(p) as f:
        pairs = json.load(f)
    print(f"[data] {len(pairs)} pairs")
    return pairs


def build_dataset(pairs: list[dict], tok: GPT2TokenizerFast) -> Dataset:
    eos = tok.eos_token_id
    cols = sorted({k for ex in pairs[:2000] for k in ex.keys()})

    def encode(ex):
        p = tok.encode(ex["prompt"], add_special_tokens=False)
        g = tok.encode(ex["gold"], add_special_tokens=False)[:80]
        max_p = PAD - len(g) - 1
        p = p[-max_p:]
        seq = p + g + [eos]
        labels = [-100] * len(p) + g + [eos]
        attn = [1] * len(seq)
        pad_n = PAD - len(seq)
        if pad_n > 0:
            seq += [eos] * pad_n
            labels += [-100] * pad_n
            attn += [0] * pad_n
        return {"input_ids": seq, "labels": labels, "attention_mask": attn}

    return Dataset.from_list(pairs).map(encode, remove_columns=cols)


def main() -> None:
    t0 = time.time()
    pairs = download_dataset()

    print(f"[model] loading {BASE_MODEL}")
    tok = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

    ds = build_dataset(pairs, tok)
    print(f"[data] tokenized {len(ds)} samples, seq_len={PAD}")

    cuda = torch.cuda.is_available()
    bf16_ok = cuda and torch.cuda.is_bf16_supported()
    print(f"[device] cuda={cuda} bf16={bf16_ok}")
    if cuda:
        print(
            f"[device] {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
        )

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        # one continued-finetune epoch from v9 (not base GPT-2)
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=100,
        save_strategy="no",
        bf16=bf16_ok,
        fp16=cuda and not bf16_ok,
        report_to=[],
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    print(f"[save] wrote {OUT_DIR}")

    api = HfApi()
    api.create_repo(TARGET_REPO, private=False, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=TARGET_REPO,
        commit_message=(
            "Iter 40: GPT-2 124M v15 — continued fine-tune from v9, 1 epoch on "
            "sh_train_v15 (v9 78k + 2.6k failure-driven targeted rows x3). "
            "L40s bf16, batch 8, lr 1e-5, seq 1024."
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")
    print(f"[time] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
