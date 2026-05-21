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
"""Iter 36 — ctx-4096 v14: fix the stretch tax + train real long context.

Iter 35's v13-ctx4096 lost 3.7 pp of short-prompt accuracy (81.0 -> 77.3)
because `extend_wpe` linearly interpolated the *whole* 1024-row position
table to 4096 rows — so even a 300-token prompt landed on 4x-compressed,
distorted `wpe` rows. And it trained from base GPT-2 for 1 epoch, throwing
away v9's 2-epoch fine-tune.

v14 fixes both:

  1. INIT FROM v9, not base GPT-2. Keep every fine-tuned weight v9 learned.

  2. BLOCK-PRESERVING wpe extension. The new 4096-row table is:
       rows    0-1023  = v9's exact, already-tuned wpe  (bit-for-bit)
       rows 1024-4095  = v9's 1024 rows linearly interpolated to 3072
     A prompt <=1024 tokens therefore sees v9's *native* position
     embeddings — zero distortion, so the short-prompt tax is gone. Only
     the rarely-used tail carries any stretch.

  3. FREEZE the 0-1023 wpe block during SFT (a gradient hook zeros its
     grad). The short-range position table stays identical to v9 no matter
     what; only the new tail rows and the transformer body adapt.

  4. Train on `sh_train_v14_long.json` — v9's 78k originals plus ~22k
     synthetic long rows (built by `build_longctx.py`, schema-padded to
     span 1024-4000 tokens uniformly). v13 never saw real content above
     ~3542 tokens; v14 does, at every position.

Still bit-for-bit the GPT-2 architecture (just a 4096-row wpe), so
`export_onnx.py` and transformers.js consume it unchanged.

Dataset: lifeart/smart-home-sft-v2 (sh_train_v14_long.json)
Base:    lifeart/smart-home-gpt2-v9
Target:  lifeart/smart-home-gpt2-v14-ctx4096

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 6h \\
        --detach training/train_hf_v14_ctx4096.py
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

BASE_MODEL = os.environ.get("BASE_MODEL", "lifeart/smart-home-gpt2-v9")
TARGET_REPO = os.environ.get(
    "TARGET_REPO", "lifeart/smart-home-gpt2-v14-ctx4096"
)
DATA_REPO = "lifeart/smart-home-sft-v2"
DATA_FILE = os.environ.get("DATA_FILE", "sh_train_v14_long.json")
NEW_CTX = 4096
KEEP = 1024     # rows 0..KEEP-1 are preserved bit-for-bit from v9
MAXLEN = 4096   # hard sequence cap; rows are NOT padded to it (see collate)
OUT_DIR = Path("out/final")


def extend_wpe_block(model: GPT2LMHeadModel, new_len: int = NEW_CTX) -> None:
    """Block-preserving position-table extension.

    Unlike Iter 35's whole-table interpolation, the first `KEEP` rows are
    copied verbatim from the source model, so prompts <=KEEP tokens see the
    exact native embeddings. Only the `new_len - KEEP` tail rows are
    synthesized — by linearly interpolating the source's `KEEP` rows up to
    the tail length, giving the tail unique, smoothly varying embeddings.
    """
    old = model.transformer.wpe.weight.data  # (old_len, 768)
    old_len, hidden = old.shape
    print(f"[wpe] block-extend {old_len} -> {new_len} (keep 0..{KEEP-1} exact)")
    if old_len != KEEP:
        raise ValueError(f"expected a {KEEP}-row source wpe, got {old_len}")
    if new_len == old_len:
        print("[wpe] already at target length, nothing to do")
        return

    tail_len = new_len - KEEP
    # (KEEP, 768) -> (1, 768, KEEP) -> interpolate -> (1, 768, tail_len)
    src = old.t().unsqueeze(0).float()
    tail = F.interpolate(
        src, size=tail_len, mode="linear", align_corners=True
    ).squeeze(0).t().contiguous()  # (tail_len, 768)

    new_weight = torch.empty((new_len, hidden), dtype=old.dtype)
    new_weight[:KEEP] = old                       # exact native prefix
    new_weight[KEEP:] = tail.to(old.dtype)        # interpolated tail
    assert torch.equal(new_weight[:KEEP], old), "prefix not preserved!"

    new_wpe = nn.Embedding(new_len, hidden)
    new_wpe.weight.data.copy_(new_weight)
    model.transformer.wpe = new_wpe
    model.config.n_positions = new_len
    model.config.n_ctx = new_len
    print(
        f"[wpe] installed nn.Embedding({new_len}, {hidden}); "
        f"n_positions={model.config.n_positions}"
    )


def freeze_wpe_prefix(model: GPT2LMHeadModel, keep: int = KEEP) -> None:
    """Register a backward hook that zeros the gradient of wpe rows 0..keep-1
    so the preserved short-range position table never moves during SFT."""
    w = model.transformer.wpe.weight
    mask = torch.ones_like(w)
    mask[:keep] = 0.0
    w.register_hook(lambda grad: grad * mask.to(grad.device))
    print(f"[wpe] froze gradient on rows 0..{keep-1} ({keep} rows locked)")


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
    """Encode each row to its NATURAL length (capped at MAXLEN). Rows are not
    padded here — `collate` pads each batch to its own longest row. With
    per_device_train_batch_size=1 that means zero padding waste: a 400-token
    row does a 400-token forward, not a 4096-token one."""
    eos = tok.eos_token_id
    cols = sorted({k for ex in pairs[:2000] for k in ex.keys()})

    def encode(ex):
        p = tok.encode(ex["prompt"], add_special_tokens=False)
        g = tok.encode(ex["gold"], add_special_tokens=False)[:80]
        max_p = MAXLEN - len(g) - 1
        p = p[-max_p:]
        seq = p + g + [eos]
        labels = [-100] * len(p) + g + [eos]
        attn = [1] * len(seq)
        return {"input_ids": seq, "labels": labels, "attention_mask": attn}

    return Dataset.from_list(pairs).map(encode, remove_columns=cols)


def make_collate(eos: int):
    """Right-pad a batch to its longest row (input_ids=eos, labels=-100,
    attention_mask=0). Correct for any batch size; a no-op when batch=1."""

    def collate(features: list[dict]) -> dict:
        m = max(len(f["input_ids"]) for f in features)
        ids, lab, att = [], [], []
        for f in features:
            n = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [eos] * n)
            lab.append(f["labels"] + [-100] * n)
            att.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }

    return collate


def main() -> None:
    t0 = time.time()
    pairs = download_dataset()

    print(f"[model] loading {BASE_MODEL}")
    tok = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    print(f"[model] source wpe = {tuple(model.transformer.wpe.weight.shape)}")

    # --- context-window extension -------------------------------------
    extend_wpe_block(model, new_len=NEW_CTX)
    freeze_wpe_prefix(model, keep=KEEP)
    with torch.no_grad():
        probe = torch.zeros((1, 3900), dtype=torch.long)
        _ = model(probe)
    print("[wpe] >2048-token forward probe OK (seq_len=3900)")
    # ------------------------------------------------------------------

    ds = build_dataset(pairs, tok)
    lens = [len(x) for x in ds["input_ids"]]
    lens.sort()
    print(
        f"[data] tokenized {len(ds)} samples — seq_len p50/p90/p99/max = "
        f"{lens[len(lens)//2]}/{lens[int(len(lens)*0.9)]}/"
        f"{lens[int(len(lens)*0.99)]}/{lens[-1]} (dynamic-padded per batch)"
    )

    cuda = torch.cuda.is_available()
    bf16_ok = cuda and torch.cuda.is_bf16_supported()
    print(f"[device] cuda={cuda} bf16={bf16_ok}")
    if cuda:
        print(
            f"[device] {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
        )

    # 4096-ctx attention is O(seq^2). Fit an L40s (48 GB) with batch 1 x
    # grad-accum 8 + gradient checkpointing (needs use_cache=False).
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
        # weight_decay=0: AdamW's decoupled decay would shrink the frozen
        # wpe prefix even with its gradient hooked to zero. 0.0 keeps the
        # preserved rows 0..1023 bit-exact; for a 1-epoch adaptation of an
        # already-converged checkpoint, decay changes nothing measurable.
        weight_decay=0.0,
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
        data_collator=make_collate(tok.eos_token_id),
    )
    trainer.train()

    # sanity: the preserved prefix must be byte-identical to v9's wpe
    src_wpe = GPT2LMHeadModel.from_pretrained(
        BASE_MODEL
    ).transformer.wpe.weight.data
    trained_prefix = model.transformer.wpe.weight.data[:KEEP].cpu()
    if torch.equal(trained_prefix, src_wpe):
        print(f"[wpe] OK — rows 0..{KEEP-1} byte-identical to v9 after training")
    else:
        drift = (trained_prefix - src_wpe).abs().max().item()
        print(f"[wpe] WARNING — prefix drifted (max abs {drift:.2e})")

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
            "Iter 36: GPT-2 124M v14-ctx4096 — block-preserving wpe "
            "(rows 0-1023 = v9 verbatim, frozen; 1024-4095 interpolated), "
            "init from v9, 1 SFT epoch on sh_train_v14_long (78k orig + "
            "22k synthetic long rows). L40s bf16, batch 1 x grad-accum 8, "
            "gradient checkpointing, lr 1e-5, seq 4096."
        ),
    )
    print(f"[push] uploaded to https://huggingface.co/{TARGET_REPO}")
    print(f"[time] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
