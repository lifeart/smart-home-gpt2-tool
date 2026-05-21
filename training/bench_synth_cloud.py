# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""B4 stage 3a — does the synth model actually add value?

Over sh_test.json (held-out), measures raw exact-match (parsed-dict
equality) for, under ONE consistent metric:
  - v6 greedy            (candidate A)
  - v9 greedy            (candidate B)
  - oracle(v6, v9)       best-of-candidates — synth's copy-only ceiling
  - synth                v6 -> v9 -> synth full cascade

If synth does not beat v9 alone, B4 does not pan out and there is no
point ONNX-exporting / wiring the browser cascade — report the honest
negative and stop. If it beats v9 (and approaches/【beats oracle), the
browser integration is worth it.

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 1h \\
        --detach training/bench_synth_cloud.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA_REPO = "lifeart/smart-home-sft-v2"
TEST_FILE = os.environ.get("TEST_FILE", "sh_test.json")
V6 = "lifeart/smart-home-gpt2-v6"
V9 = "lifeart/smart-home-gpt2-v9"
SYNTH = os.environ.get("SYNTH_REPO", "lifeart/smart-home-gpt2-synth")
ASSIST_MARKER = "\n\n\nASSISTANT:"


def first_json(text: str):
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


def compact(call) -> str:
    if not isinstance(call, dict):
        return "(no valid call)"
    return json.dumps(call, separators=(",", ":"))


def make_synth_prompt(prompt: str, v6_call, v9_call) -> str:
    """Identical to build_synth_distill.make_synth_prompt."""
    idx = prompt.find(ASSIST_MARKER)
    if idx == -1:
        return prompt
    head, tail = prompt[:idx], prompt[idx:]
    block = (
        "\n\nProposed tool calls (evidence from other systems — pick the "
        "right function, merge the best arguments, and fix wrong values):\n"
        f"- candidate A: {compact(v6_call)}\n"
        f"- candidate B: {compact(v9_call)}"
    )
    return head + block + tail


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
    t0 = time.time()
    p = hf_hub_download(DATA_REPO, TEST_FILE, repo_type="dataset")
    rows = json.loads(Path(p).read_text())
    print(f"[data] {len(rows)} test rows", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    def gold_of(r):
        g = r["gold"]
        return json.loads(g) if isinstance(g, str) else g

    # --- v6 + v9 candidates -------------------------------------------
    cand = {}
    for tag, repo in (("v6", V6), ("v9", V9)):
        print(f"[load] {repo}", flush=True)
        tok, model = load(repo, device)
        calls = []
        for i, r in enumerate(rows):
            calls.append(first_json(generate(model, tok, r["prompt"], device)))
            if (i + 1) % 100 == 0:
                print(f"  [{tag} {i+1}/{len(rows)}]", flush=True)
        cand[tag] = calls
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # --- synth on (prompt + candidates) -------------------------------
    print(f"[load] {SYNTH}", flush=True)
    tok, model = load(SYNTH, device)
    synth_calls = []
    for i, r in enumerate(rows):
        sp = make_synth_prompt(r["prompt"], cand["v6"][i], cand["v9"][i])
        synth_calls.append(first_json(generate(model, tok, sp, device)))
        if (i + 1) % 100 == 0:
            print(f"  [synth {i+1}/{len(rows)}]", flush=True)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # --- score: raw parsed-dict exact-match ---------------------------
    n = len(rows)
    hit = {"v6": 0, "v9": 0, "oracle": 0, "synth": 0}
    synth_fixed = synth_broke = 0
    for i, r in enumerate(rows):
        g = gold_of(r)
        v6_ok = cand["v6"][i] == g
        v9_ok = cand["v9"][i] == g
        s_ok = synth_calls[i] == g
        hit["v6"] += v6_ok
        hit["v9"] += v9_ok
        hit["oracle"] += (v6_ok or v9_ok)
        hit["synth"] += s_ok
        if s_ok and not (v6_ok or v9_ok):
            synth_fixed += 1            # synth produced gold no candidate had
        if (v6_ok or v9_ok) and not s_ok:
            synth_broke += 1            # synth lost a call a candidate had right

    print("\n=== B4 synth bench (raw parsed-dict exact-match, "
          f"n={n}, {TEST_FILE}) ===", flush=True)
    for k in ("v6", "v9", "oracle", "synth"):
        print(f"  {k:8s} {hit[k]:4d}/{n} = {hit[k]/n*100:5.1f}%", flush=True)
    print(f"\n  synth vs v9      : {(hit['synth']-hit['v9'])/n*100:+.1f} pp", flush=True)
    print(f"  synth vs oracle  : {(hit['synth']-hit['oracle'])/n*100:+.1f} pp", flush=True)
    print(f"  synth fixed (no candidate had it): {synth_fixed}", flush=True)
    print(f"  synth broke (lost a correct candidate): {synth_broke}", flush=True)
    print(f"\n[time] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
