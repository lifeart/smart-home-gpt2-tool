# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""B4 stage 1 — generate the two candidate calls per source row.

The synth model needs DIVERSE, complementary evidence. The first cut used
v6 and v9 each as bare greedy full-call generators — but v9 is an
*arguments specialist*: in the H1.2_con cascade it is always prompted with
`build_args_only_prompt` (the function name hinted, "output the arguments
only"). Run bare, v9 scored only 12% — useless as a candidate. Corrected:

  candidate A  = v6 greedy full call            (the strong full-call model)
  candidate B  = v6's name + v9 args-only        (the H1 cascade output)

Both via plain KV-cached greedy `model.generate` — fast, robust, no
constrained decoder to hang. For each source row emits {prompt, gold,
gold_name, domain, v6_call, v9_call}; stage 2 turns these into the
distillation set.

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
V6 = "lifeart/smart-home-gpt2-v6"
V9 = "lifeart/smart-home-gpt2-v9"

# matches bench_h1_con_cloud.py — how the H1.2_con cascade invokes v9
ARGS_HINT_TMPL = "Note: The function name will be: {name}. Output the arguments only.\n\n\n"
MARKER = "\n\n\nASSISTANT:"


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


def build_args_only_prompt(prompt: str, name: str) -> str:
    """Re-frame the prompt for the v9 args specialist (name hinted)."""
    hint = ARGS_HINT_TMPL.format(name=name)
    if MARKER in prompt:
        head, tail = prompt.split(MARKER, 1)
        return head + "\n" + hint + "ASSISTANT:" + tail
    return prompt + " " + hint


def args_from(obj):
    """Pull the arguments dict out of a v9 args-only output."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("arguments"), dict):
        return obj["arguments"]
    if "arguments" not in obj and "name" not in obj:
        return obj  # bare args dict
    return None


@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new: int = 64) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids
    cap = model.config.n_positions
    if ids.shape[1] > cap - max_new:
        ids = ids[:, -(cap - max_new):]
    ids = ids.to(device)
    out = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load(repo, device):
    tok = GPT2TokenizerFast.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()
    return tok, model


def main() -> None:
    p = hf_hub_download(DATA_REPO, SRC_FILE, repo_type="dataset")
    rows = json.loads(Path(p).read_text())
    print(f"[data] {len(rows)} source rows", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # --- candidate A: v6 greedy full call -----------------------------
    print(f"[load] {V6}", flush=True)
    tok, model = load(V6, device)
    v6_calls = []
    t0 = time.time()
    for i, r in enumerate(rows):
        v6_calls.append(first_json(generate(model, tok, r["prompt"], device)))
        if (i + 1) % 250 == 0:
            print(f"  [v6 {i+1}/{len(rows)}] t={time.time()-t0:.0f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[v6] done in {time.time()-t0:.0f}s", flush=True)

    # --- candidate B: v6's name + v9 args-only ------------------------
    print(f"[load] {V9}", flush=True)
    tok, model = load(V9, device)
    v9_calls = []
    t0 = time.time()
    for i, r in enumerate(rows):
        name = v6_calls[i].get("name") if isinstance(v6_calls[i], dict) else None
        if isinstance(name, str) and name:
            ap = build_args_only_prompt(r["prompt"], name)
            args = args_from(first_json(generate(model, tok, ap, device)))
            v9_calls.append({"name": name, "arguments": args}
                            if args is not None else None)
        else:
            v9_calls.append(None)
        if (i + 1) % 250 == 0:
            print(f"  [v9 {i+1}/{len(rows)}] t={time.time()-t0:.0f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[v9] done in {time.time()-t0:.0f}s", flush=True)

    out = [
        {
            "prompt": r["prompt"], "gold": r["gold"],
            "gold_name": r.get("gold_name"), "domain": r.get("domain"),
            "v6_call": v6_calls[i], "v9_call": v9_calls[i],
        }
        for i, r in enumerate(rows)
    ]
    Path(OUT_FILE).write_text(json.dumps(out))
    n_v6 = sum(1 for c in v6_calls if c)
    n_v9 = sum(1 for c in v9_calls if c)
    print(f"[save] {OUT_FILE}: {len(out)} rows "
          f"(v6 parsed {n_v6}, v9 parsed {n_v9})", flush=True)

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            HfApi().upload_file(
                path_or_fileobj=OUT_FILE, path_in_repo=OUT_FILE,
                repo_id=DATA_REPO, repo_type="dataset",
                commit_message="B4: v6 full-call + v6-name/v9-args candidates",
            )
            print(f"[push] -> {DATA_REPO}/{OUT_FILE}", flush=True)
        except Exception as e:
            print(f"[push] failed: {e}", flush=True)


if __name__ == "__main__":
    main()
