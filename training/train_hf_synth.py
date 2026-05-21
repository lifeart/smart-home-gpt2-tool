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
"""B4 stage 2 — train the synthesis-aware GPT-2 (`smart-home-gpt2-synth`).

Continues the fine-tune from v9 on `sh_train_synth.json` — each row's
prompt carries the user query + function schemas + two candidate calls
(v6 + v9 greedy) as labelled evidence; the target is the TRUE gold call.

The model learns to do what the Llama synthesizer does in the API
pipeline — pick the right function, merge the best arguments, fix wrong
values — but trained on ground truth and small enough to run in-browser.
It becomes the 3rd stage of the v6 -> v9 -> synth browser cascade.

The candidates block sits just before the ASSISTANT marker, so left-side
truncation (long schemas, 1024 ctx) always preserves the evidence — the
synth model can correct from candidates + query even when the schema head
is clipped. v6/v9 candidate generators are 1024-ctx too: train == infer.

Dataset: lifeart/smart-home-sft-v2 (sh_train_synth.json, 2500 rows)
Base:    lifeart/smart-home-gpt2-v9
Target:  lifeart/smart-home-gpt2-synth

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 1h \\
        --detach training/train_hf_synth.py
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
TARGET_REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2-synth")
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = os.environ.get("DATA_FILE", "sh_train_synth.json")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
PAD = 1024
OUT_DIR = Path("out/final")


def build_dataset(pairs: list[dict], tok: GPT2TokenizerFast) -> Dataset:
    eos = tok.eos_token_id
    cols = sorted({k for ex in pairs[:2000] for k in ex.keys()})

    def encode(ex):
        p = tok.encode(ex["prompt"], add_special_tokens=False)
        g = tok.encode(ex["gold"], add_special_tokens=False)[:80]
        max_p = PAD - len(g) - 1
        p = p[-max_p:]  # keep the tail — candidates block + query survive
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
    print(f"[data] fetching {DATA_REPO}/{DATA_FILE}")
    p = hf_hub_download(repo_id=DATA_REPO, filename=DATA_FILE, repo_type="dataset")
    pairs = json.loads(Path(p).read_text())
    print(f"[data] {len(pairs)} synth-distillation pairs")

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
        # new task (CANDIDATES block) on a small set -> slightly hotter lr
        # and a few epochs; init-from-v9 retains general tool-calling.
        learning_rate=3e-5,
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
            f"B4: GPT-2 124M synth — continued fine-tune from v9, {EPOCHS:g} "
            "epochs on sh_train_synth (2500 candidate-distillation rows). "
            "L40s bf16, batch 8, lr 3e-5, seq 1024."
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")
    print(f"[time] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
