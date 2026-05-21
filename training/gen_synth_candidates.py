# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""B4 stage 1 — generate v6 + v9 candidate calls over the synth source rows.

Replaces the constrained-decoding candidate run (bench_h1_con_cloud.py),
which hung on degenerate rows and was slow / unmonitorable. This does plain
KV-cached greedy `model.generate` — fast (~0.3 s/row/model), robust, and it
cannot stall on a constraint. The candidates are scaffolding/evidence for
the synth model; greedy decode is exactly what the browser cascade runs.

For each source row emits {prompt, gold, gold_name, domain, v6_call,
v9_call}. Stage 2 turns these into the distillation dataset.

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 1h \\
        --detach training/gen_synth_candidates.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA_REPO = "lifeart/smart-home-sft-v2"
SRC_FILE = os.environ.get("SRC_FILE", "sh_synth_src.json")
OUT_FILE = os.environ.get("OUT_FILE", "b4_synth_candidates.json")
MODELS = {"v6": "lifeart/smart-home-gpt2-v6", "v9": "lifeart/smart-home-gpt2-v9"}


def first_json(text: str):
    """First balanced {...} object, parsed; None if unparseable."""
    s = text.find("{")
    if s < 0:
        return None
    depth = 0
    in_str = esc = False
    for i in range(s, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s:i + 1])
                except Exception:
                    return None
    return None


@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new: int = 64) -> str:
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids
    cap = model.config.n_positions
    if ids.shape[1] > cap - max_new:
        ids = ids[:, -(cap - max_new):]
    ids = ids.to(device)
    attn = torch.ones_like(ids)
    out = model.generate(
        input_ids=ids, attention_mask=attn, max_new_tokens=max_new,
        do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    p = hf_hub_download(DATA_REPO, SRC_FILE, repo_type="dataset")
    rows = json.loads(Path(p).read_text())
    print(f"[data] {len(rows)} source rows", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    cands: dict[str, list] = {}
    for tag, repo in MODELS.items():
        print(f"[load] {repo}", flush=True)
        tok = GPT2TokenizerFast.from_pretrained(repo)
        tok.pad_token = tok.eos_token
        model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()
        out_calls = []
        t0 = time.time()
        for i, r in enumerate(rows):
            out_calls.append(first_json(generate(model, tok, r["prompt"], device)))
            if (i + 1) % 250 == 0:
                print(f"  [{tag} {i+1}/{len(rows)}] t={time.time()-t0:.0f}s",
                      flush=True)
        cands[tag] = out_calls
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)

    out = [
        {
            "prompt": r["prompt"], "gold": r["gold"],
            "gold_name": r.get("gold_name"), "domain": r.get("domain"),
            "v6_call": cands["v6"][i], "v9_call": cands["v9"][i],
        }
        for i, r in enumerate(rows)
    ]
    Path(OUT_FILE).write_text(json.dumps(out))
    n_v6 = sum(1 for c in cands["v6"] if c)
    n_v9 = sum(1 for c in cands["v9"] if c)
    print(f"[save] {OUT_FILE}: {len(out)} rows "
          f"(v6 parsed {n_v6}, v9 parsed {n_v9})", flush=True)

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            HfApi().upload_file(
                path_or_fileobj=OUT_FILE, path_in_repo=OUT_FILE,
                repo_id=DATA_REPO, repo_type="dataset",
                commit_message="B4: v6+v9 greedy candidate calls",
            )
            print(f"[push] -> {DATA_REPO}/{OUT_FILE}", flush=True)
        except Exception as e:
            print(f"[push] failed: {e}", flush=True)


if __name__ == "__main__":
    main()
