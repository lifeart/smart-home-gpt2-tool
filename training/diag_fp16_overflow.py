# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 37 diagnostic — is the fp16-on-WebGPU failure caused by activation
overflow, and if so, where?

The browser runs v14-ctx4096's fp16 ONNX on WebGPU (genuine fp16 compute)
and emits all-spaces garbage; the fp32 ONNX is fine. Hypothesis: GPT-2's
activations exceed fp16's max finite value (65504) somewhere in the forward
pass, so fp16 saturates to +-Inf -> NaN -> degenerate output.

This loads v14 in fp32 PyTorch, hooks every module, runs short and long
prompts, and reports the max-abs activation per module — flagging any that
exceed the fp16 ceiling. That pinpoints which op-types a corrected fp16
export must keep in fp32.

Local, single model, no GPU needed. Run:
    python training/diag_fp16_overflow.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO = "lifeart/smart-home-gpt2-v14-ctx4096"
FP16_MAX = 65504.0
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print(f"[load] {REPO}")
    tok = GPT2TokenizerFast.from_pretrained(REPO)
    model = GPT2LMHeadModel.from_pretrained(REPO).eval()

    # max abs activation seen per module, across all runs
    peak: dict[str, float] = {}

    def mk_hook(name: str):
        def hook(_m, _inp, out):
            tensors = []
            if isinstance(out, torch.Tensor):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [t for t in out if isinstance(t, torch.Tensor)]
            for t in tensors:
                if t.numel():
                    m = t.detach().abs().max().item()
                    if m > peak.get(name, 0.0):
                        peak[name] = m
        return hook

    for name, mod in model.named_modules():
        if name:  # skip the root ""
            mod.register_forward_hook(mk_hook(name))

    # representative inputs: a short SH prompt and a genuine long one
    prompts: list[tuple[str, str]] = []
    test = json.loads((ROOT / "data" / "sh_test.json").read_text())
    prompts.append(("short", test[0]["prompt"]))
    long_test = json.loads((ROOT / "data" / "sh_test_long.json").read_text())
    longrow = next((r for r in long_test if r.get("bucket") == "3500"), None)
    if longrow:
        prompts.append(("long-3500", longrow["prompt"]))

    for tag, p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)
        print(f"[run] {tag}: {ids.shape[1]} tokens")

    # ---- report -------------------------------------------------------
    print(f"\n[fp16 ceiling] {FP16_MAX}")
    items = sorted(peak.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== top-20 modules by peak abs activation ===")
    for name, v in items[:20]:
        flag = "  <-- OVER fp16 max" if v > FP16_MAX else ""
        print(f"  {v:14.1f}  {name}{flag}")

    over = [(n, v) for n, v in items if v > FP16_MAX]
    print(f"\n=== modules exceeding fp16 max ({len(over)}) ===")
    # group by op-type suffix (e.g. 'mlp.c_fc', 'ln_1') to see the pattern
    by_kind: dict[str, list[float]] = {}
    for n, v in over:
        kind = ".".join(p for p in n.split(".") if not p.isdigit()) or n
        by_kind.setdefault(kind, []).append(v)
    for kind, vs in sorted(by_kind.items(), key=lambda kv: max(kv[1]), reverse=True):
        print(f"  {kind:<28} n={len(vs):<3} peak={max(vs):.1f}")

    glob = items[0]
    print(f"\n[verdict] global peak = {glob[1]:.1f} at '{glob[0]}'")
    if glob[1] > FP16_MAX:
        print("OVERFLOW CONFIRMED — fp16 compute saturates here; a corrected")
        print("fp16 export must keep the above op-types in fp32.")
    else:
        print("No activation exceeds the fp16 ceiling — the WebGPU failure is")
        print("NOT plain overflow; investigate the fp16 ONNX graph instead.")


if __name__ == "__main__":
    main()
