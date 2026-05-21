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
"""Iter 35 — extend smart-home GPT-2 context window 1024 -> 4096, v9 data.

Stock `openai-community/gpt2` uses learned absolute position embeddings:
a 1024x768 `transformer.wpe` table. Position >=1024 has no embedding, so
1024 is a hard wall. Iter 34 lifted that wall to 2048 (v12-ctx2048); this
script lifts it to 4096 AND retrains on the stronger v9 dataset.

  1. Load GPT2LMHeadModel (124M, 12 layers, 768 hidden).
  2. `extend_wpe(model, 4096)` — linear-interpolate the 1024x768 `wpe`
     table to 4096x768 and install it as a new nn.Embedding(4096, 768).
     `config.n_positions = config.n_ctx = 4096`.
  3. One short SFT epoch at seq_len 4096 so the model adapts to the
     interpolated positions.

This stays bit-for-bit the GPT-2 architecture (just a bigger wpe table),
so `training/export_onnx.py` exports it unchanged and transformers.js
runs it.

Difference vs v12 (Iter 34):
- v12 trained on `sh_train_v11.json` (27.6k rows) and scored 79.3%
  name-accuracy — below v9's 81.0%. The gap is the *data*, not the
  context method. v13 trains on `sh_train_v9.json` (78k-row Granite
  curriculum) so it reaches v9-level accuracy with a 4096 window.
- 4096-ctx attention is heavy. Memory budget: per_device_train_batch_size
  =1, gradient_accumulation_steps=8 (effective batch 8, same as v12),
  gradient_checkpointing=True. Targets an L40s (48 GB).

Dataset: lifeart/smart-home-sft-v2 (sh_train_v9.json) — 78k Granite rows.
Base:    openai-community/gpt2
Target:  lifeart/smart-home-gpt2-v13-ctx4096

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 4h \\
        --detach training/train_hf_v13_ctx4096.py
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "openai-community/gpt2"
TARGET_REPO = os.environ.get(
    "TARGET_REPO", "lifeart/smart-home-gpt2-v13-ctx4096"
)
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = "sh_train_v9.json"
NEW_CTX = 4096
PAD = 4096
OUT_DIR = Path("out/final")


def extend_wpe(model: GPT2LMHeadModel, new_len: int = NEW_CTX) -> None:
    """Resize the learned position-embedding table from its current length
    to `new_len` rows via linear interpolation of the existing rows.

    `transformer.wpe.weight` is (old_len, 768). F.interpolate expects
    (batch, channels, length), so we transpose to (1, 768, old_len),
    interpolate the length axis to (1, 768, new_len), then transpose back
    to (new_len, 768). A fresh nn.Embedding(new_len, 768) is installed and
    config.n_positions / config.n_ctx are bumped.
    """
    old = model.transformer.wpe.weight.data  # (old_len, 768)
    old_len, hidden = old.shape
    print(f"[wpe] extending {old_len} -> {new_len} (hidden={hidden})")
    if new_len == old_len:
        print("[wpe] already at target length, nothing to do")
        return
    # (old_len, 768) -> (1, 768, old_len)
    src = old.t().unsqueeze(0).float()
    # (1, 768, new_len)
    interp = F.interpolate(
        src, size=new_len, mode="linear", align_corners=True
    )
    # (1, 768, new_len) -> (new_len, 768)
    new_weight = interp.squeeze(0).t().contiguous()
    assert new_weight.shape == (new_len, hidden), new_weight.shape

    new_wpe = nn.Embedding(new_len, hidden)
    new_wpe.weight.data.copy_(new_weight.to(old.dtype))
    model.transformer.wpe = new_wpe

    model.config.n_positions = new_len
    model.config.n_ctx = new_len
    print(
        f"[wpe] installed nn.Embedding({new_len}, {hidden}); "
        f"config.n_positions={model.config.n_positions} "
        f"config.n_ctx={model.config.n_ctx}"
    )


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
    t0 = time.time()
    pairs = download_dataset()

    print(f"[model] loading {BASE_MODEL}")
    tok = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

    # --- context-window extension -------------------------------------
    extend_wpe(model, new_len=NEW_CTX)
    # sanity: forward a >1024-token batch before training. Use seq 3500 to
    # exercise positions well past both the old 1024 wall and the v12 2048
    # wall, confirming the interpolated 4096-row wpe is wired correctly.
    with torch.no_grad():
        probe = torch.zeros((1, 3500), dtype=torch.long)
        _ = model(probe)
    print("[wpe] >2048-token forward probe OK (seq_len=3500)")
    # ------------------------------------------------------------------

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

    # 4096-ctx attention is O(seq^2) memory. To fit an L40s (48 GB):
    #   batch 1 x grad-accum 8 (effective batch 8, same as v12)
    #   gradient_checkpointing=True (recompute activations in backward).
    # gradient_checkpointing needs use_cache=False on GPT-2.
    model.config.use_cache = False
    if cuda:
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
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

    # restore use_cache for inference/export
    model.config.use_cache = True

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
            "Iter 35: GPT-2 124M v13 — context window extended 1024->4096 "
            "via linear-interpolated wpe, then 1 SFT epoch on sh_train_v9 "
            "(78k Granite curriculum). L40s bf16, batch 1 x grad-accum 8, "
            "gradient checkpointing, lr 1e-5, seq 4096."
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")
    print(f"[time] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
