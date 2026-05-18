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
"""SFT GPT-2 124M v6 — v5 base + 590 fresh_bench items (universal tool-call).

Same hyperparams as v5: 1 epoch, lr 1e-5, bs 2, grad-accum 2, seq_len 1024.

Dataset: lifeart/smart-home-sft-v2 (sh_train_v6.json, ~20600 rows)
Base:    openai-community/gpt2
Target:  lifeart/smart-home-gpt2-v6

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 3h \\
        training/train_hf_v6.py
"""
import json
import os
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

BASE_MODEL = "openai-community/gpt2"
TARGET_REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2-v6")
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = "sh_train_v6.json"
PAD = 1024
OUT_DIR = Path("out/final")


def download_dataset() -> list[dict]:
    print(f"[data] fetching {DATA_REPO}/{DATA_FILE}")
    p = hf_hub_download(
        repo_id=DATA_REPO, filename=DATA_FILE, repo_type="dataset"
    )
    with open(p) as f:
        pairs = json.load(f)
    print(f"[data] {len(pairs)} pairs")
    return pairs


def build_dataset(pairs: list[dict], tok: GPT2TokenizerFast) -> Dataset:
    eos = tok.eos_token_id

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

    return Dataset.from_list(pairs).map(
        encode, remove_columns=list(pairs[0].keys())
    )


def main() -> None:
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

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_steps=20,
        logging_steps=50,
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
    api.create_repo(
        TARGET_REPO, private=False, exist_ok=True, repo_type="model"
    )
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=TARGET_REPO,
        commit_message=(
            "SFT GPT-2 124M v6 — v5 base + 590 fresh_bench universal tool-call items"
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    main()
