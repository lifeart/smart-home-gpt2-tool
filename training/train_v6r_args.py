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
"""Iter 24 — fine-tune v6 → v6r-args (Granite args_only on SH-only data).

Starting from `lifeart/smart-home-gpt2-v6` (best in-domain single model),
fine-tune on `sh_train_v6r_args.json` (16,515 args_only rows from v6r SH
data only — no HA/Nemotron dilution).

Hypothesis: gives an args-stage-2 model that matches/exceeds v9 on most
domains AND fixes v9's `clean` regression, eliminating the H1.2 clean-gate
and potentially closing the H1.2 → H1.3 gap (gate-cleared cleanly without
runtime Llama dependency).

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 1h \\
        training/train_v6r_args.py
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


BASE_MODEL = os.environ.get("BASE_MODEL", "lifeart/smart-home-gpt2-v6")
TARGET_REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2-v6r-args")
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = os.environ.get("DATA_FILE", "sh_train_v6r_args.json")
PAD = 1024
LR = float(os.environ.get("LR", "5e-6"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
BATCH = int(os.environ.get("BATCH", "8"))
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

    return Dataset.from_list(pairs).map(encode, remove_columns=list(pairs[0].keys()))


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

    print(f"[cfg] LR={LR} EPOCHS={EPOCHS} BATCH={BATCH}")

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_steps=30,
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
    api.create_repo(TARGET_REPO, private=False, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=TARGET_REPO,
        commit_message=(
            f"Iter 24: v6r-args — fine-tune {BASE_MODEL} on {DATA_FILE} "
            f"(L40s bf16 batch {BATCH} lr {LR} epochs {EPOCHS})"
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    main()
