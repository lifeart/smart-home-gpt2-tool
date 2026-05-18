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
"""SFT GPT-2 124M v8 — refined v6/v7 labels via Llama-3.3-70B verifier (Iter 17).

Same hyperparams as v7 (L40s bf16 batch 8 epochs 2). Inputs come from sh_train_v8.json
(refined dataset). The dataset is v6r (~19000 rows after Llama-validated drops) or v7r
(v6r + refined Iter 16 additions) — whichever was uploaded.

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 2h \\
        training/train_hf_v8.py
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
TARGET_REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2-v8")
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = os.environ.get("DATA_FILE", "sh_train_v8.json")
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
    if cuda:
        print(
            f"[device] {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
        )

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        warmup_steps=50,
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
            "SFT GPT-2 124M v8 — Llama-70B label-refined dataset, "
            "L40s bf16 batch 8 epochs 2"
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    main()
