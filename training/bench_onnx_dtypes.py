# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "optimum[onnxruntime-gpu]>=1.23",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
#   "torch>=2.4",
# ]
# ///
"""Bench the three ONNX weight precisions of the shipped model.

Compares fp32 / fp16 / q8 ONNX exports of `lifeart/smart-home-gpt2-v9` on
the held-out test set (sh_test.json, n=300) — name / args / exact-match —
so the demo's Dtype dropdown can be backed by real accuracy numbers.

Accuracy is execution-provider-independent (a correct run gives the same
tokens whatever the backend), so each dtype uses the provider that runs it
cleanly:
  - fp32 → CPU
  - q8   → CPU   (dynamic-quantized; CPU EP runs it natively)
  - fp16 → CUDA  (CPU EP lacks fp16 kernels; needs a GPU)

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 1h \\
        training/bench_onnx_dtypes.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2TokenizerFast

MODEL_REPO = os.environ.get("MODEL_REPO", "lifeart/smart-home-gpt2-v9")
DATA_REPO = os.environ.get("DATA_REPO", "lifeart/smart-home-sft-v2")
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = full test set

ONNX_SUBFOLDER = "onnx"
DTYPES = [
    ("fp32", "model.onnx", "CPUExecutionProvider"),
    ("q8", "model_quantized.onnx", "CPUExecutionProvider"),
    ("fp16", "model_fp16.onnx", "CUDAExecutionProvider"),
]


# ---------------- scoring (mirrors web/bench.js argsMatch) ----------------

def _norm(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t and re.fullmatch(r"-?\d+(\.\d+)?", t):
            return float(t) if "." in t else int(t)
        return t.lower()
    return v


def args_match(pred: Any, gold: Any) -> bool:
    a = pred if isinstance(pred, dict) else {}
    b = gold if isinstance(gold, dict) else {}
    if sorted(a.keys()) != sorted(b.keys()):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(bv, list):
            if not isinstance(av, list) or len(av) != len(bv):
                return False
            try:
                if sorted(_norm(x) for x in av) != sorted(_norm(x) for x in bv):
                    return False
            except TypeError:
                if json.dumps(av, sort_keys=True) != json.dumps(bv, sort_keys=True):
                    return False
        elif isinstance(bv, dict):
            if not isinstance(av, dict) or not args_match(av, bv):
                return False
        else:
            an, bn = _norm(av), _norm(bv)
            if an is None and bn is None:
                continue
            if an is None or bn is None:
                return False
            if isinstance(an, (int, float)) and isinstance(bn, (int, float)):
                if an != bn:
                    return False
            elif str(an) != str(bn):
                return False
    return True


def parse_call(text: str) -> Optional[dict]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
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
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    return None
                if not isinstance(obj, dict):
                    return None
                return {
                    "name": obj.get("name") if isinstance(obj.get("name"), str) else None,
                    "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
                }
    return None


def score(pred_name, pred_args, gold_str):
    try:
        gold = json.loads(gold_str)
    except Exception:
        gold = {}
    g_name = gold.get("name") if isinstance(gold.get("name"), str) else None
    g_args = gold.get("arguments") if isinstance(gold.get("arguments"), dict) else {}
    name_ok = pred_name == g_name and g_name is not None
    a_ok = args_match(pred_args, g_args)
    return name_ok, a_ok, name_ok and a_ok


# ---------------- data ----------------

def fetch(filename: str) -> Any:
    p = hf_hub_download(DATA_REPO, filename, repo_type="dataset")
    return json.loads(Path(p).read_text())


# ---------------- bench one dtype ----------------

@torch.no_grad()
def bench_dtype(tag, file_name, provider, test, tok):
    from optimum.onnxruntime import ORTModelForCausalLM
    from onnxruntime import SessionOptions, GraphOptimizationLevel

    # Disable graph optimization — ORT's extended fusions (SimplifiedLayerNorm)
    # crash on the fp16 GPT-2 export. Optimization level does not change
    # accuracy (a correct run yields the same tokens), so disabling it for
    # all three dtypes keeps the comparison consistent.
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL

    print(f"\n[{tag}] loading {file_name} on {provider}", flush=True)
    t_load = time.time()

    def _load(prov):
        return ORTModelForCausalLM.from_pretrained(
            MODEL_REPO, file_name=file_name, subfolder=ONNX_SUBFOLDER,
            provider=prov, session_options=so,
        )

    try:
        model = _load(provider)
    except Exception as e:
        print(f"[{tag}] load failed on {provider}: {e}", flush=True)
        if provider != "CPUExecutionProvider":
            print(f"[{tag}] retrying on CPUExecutionProvider", flush=True)
            try:
                model = _load("CPUExecutionProvider")
            except Exception as e2:
                return {"tag": tag, "error": f"{e2}"}
        else:
            return {"tag": tag, "error": f"{e}"}
    print(f"[{tag}] loaded in {time.time()-t_load:.0f}s", flush=True)

    eos = tok.eos_token_id
    by_domain: dict[str, list[int]] = {}
    name_c = args_c = exact_c = 0
    t0 = time.time()
    for i, s in enumerate(test):
        ids = tok(s["prompt"], return_tensors="pt")
        plen = ids["input_ids"].shape[1]
        if plen > 1000:
            ids["input_ids"] = ids["input_ids"][:, -1000:]
            ids["attention_mask"] = ids["attention_mask"][:, -1000:]
            plen = ids["input_ids"].shape[1]
        out = model.generate(
            **ids, max_new_tokens=80, do_sample=False, pad_token_id=eos,
        )
        text = tok.decode(out[0, plen:], skip_special_tokens=True)
        call = parse_call(text)
        n_ok, a_ok, e_ok = score(
            call["name"] if call else None,
            call["arguments"] if call else {},
            s["gold"],
        )
        name_c += n_ok
        args_c += a_ok
        exact_c += e_ok
        d = s.get("domain", "?")
        b = by_domain.setdefault(d, [0, 0, 0])
        b[0] += n_ok
        b[1] += a_ok
        b[2] += e_ok
        if (i + 1) % 50 == 0:
            print(f"[{tag}] {i+1}/{len(test)}  exact={exact_c/(i+1)*100:.1f}%  "
                  f"t={time.time()-t0:.0f}s", flush=True)

    n = len(test)
    dom = {
        d: {"n": sum(1 for s in test if s.get("domain", "?") == d),
            "name": v[0], "args": v[1], "exact": v[2]}
        for d, v in by_domain.items()
    }
    return {
        "tag": tag, "file": file_name, "provider": provider, "n": n,
        "name_acc": name_c / n, "args_acc": args_c / n, "exact_acc": exact_c / n,
        "elapsed_s": time.time() - t0, "by_domain": dom,
    }


def main() -> None:
    cuda = torch.cuda.is_available()
    print(f"[env] cuda={cuda}")
    test = fetch("sh_test.json")
    if LIMIT and LIMIT < len(test):
        test = test[:LIMIT]
    print(f"[test] {len(test)} items")

    tok = GPT2TokenizerFast.from_pretrained(MODEL_REPO)
    tok.pad_token = tok.eos_token

    results = []
    for tag, fname, provider in DTYPES:
        if provider == "CUDAExecutionProvider" and not cuda:
            print(f"\n[{tag}] SKIP — no CUDA device for fp16", flush=True)
            results.append({"tag": tag, "error": "no CUDA device"})
            continue
        results.append(bench_dtype(tag, fname, provider, test, tok))

    print("\n\n===== ONNX DTYPE ACCURACY (n={}) =====".format(len(test)))
    print(f"  {'dtype':<6} {'name':>8} {'args':>8} {'exact':>8}   time")
    for r in results:
        if r.get("error"):
            print(f"  {r['tag']:<6}  ERROR: {r['error']}")
            continue
        print(f"  {r['tag']:<6} {r['name_acc']*100:7.1f}% {r['args_acc']*100:7.1f}% "
              f"{r['exact_acc']*100:7.1f}%   {r['elapsed_s']:.0f}s")

    # model-specific filename so benching a new model never clobbers an
    # earlier model's dtype results on the Hub
    out = Path(f"onnx_dtype_bench_{MODEL_REPO.split('/')[-1]}.json")
    out.write_text(json.dumps({"model": MODEL_REPO, "n": len(test), "results": results}, indent=2))
    print(f"\n[save] {out}")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            HfApi().upload_file(
                path_or_fileobj=str(out), path_in_repo=out.name,
                repo_id=DATA_REPO, repo_type="dataset",
                commit_message="ONNX fp32/fp16/q8 accuracy bench on sh_test",
            )
            print(f"[push] -> {DATA_REPO}/{out.name}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
