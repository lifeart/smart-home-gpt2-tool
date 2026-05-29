# Smart-Home GPT-2 — Benchmarks

Every number here is sourced from [`HANDOFF.md`](HANDOFF.md) / [`PLAN.md`](PLAN.md) and is
not oversold. This is a research project whose history is deliberately honest, negative
results included. For the product overview see the [README](README.md); for how the pieces
work see [ARCHITECTURE.md](ARCHITECTURE.md).

## Tool-calling accuracy

Exact-match = name + arguments, n=300, `sh_test.json`:

| Config | Exact | Notes |
|---|---:|---|
| v5 + constrained (Iter 22, prior ship) | 57.3% | single decoder |
| **H1.2_con — v6→v9 cascade** | **59.3%** | **browser-native, no external API** |
| H1.3_con (+ 2-way Llama pick) | 61.3% | needs Llama API |
| synth v2 (Iter 32) | 78.7% | GPT-2 candidates → Llama-3.3-70B synthesis + canon |
| **synth v2 + enum-snap (best)** | **81.7%** | + enum value-snapping (Iter 38) |
| oracle ceiling | 87.3% | upper bound |

**The core finding:** the ~57% plateau is a *single-decoder* ceiling, not a knowledge
ceiling. The fine-tuned GPT-2 models know the domain; Llama-70B reasons but scores only
~53% unprompted (it doesn't know the function inventory). Composed — GPT-2 emits candidates,
Llama synthesizes — it reaches 78.7%, and 81.7% with enum-snap.

## Context window — name accuracy by prompt length

`bench_ctx_long.py`, `sh_test_long.json`:

| Model | Window | short | 1500 tok | 2500 tok | 3500 tok |
|---|---:|---:|---:|---:|---:|
| v9 | 1024 | 81.0% | 58.7% | 32.6% | 20.9% |
| v13-ctx4096 | 4096 | 77.3% | 86.0% | 81.4% | 75.0% |
| **v14-ctx4096** | 4096 | **83.3%** | **93.6%** | **90.7%** | **89.5%** |

`v14-ctx4096` is the best model at every length — it supersedes both v9 and v13.

## ONNX dtype (n=300)

fp16 matches fp32 on name accuracy (v14 = 83.3% on both); q8 = 80.0% (~−3 pp). See
[QUANTIZATION.md](QUANTIZATION.md) for the full quantization analysis.

## Limitations

- **The 124M ceiling is real.** A single decoder plateaus at ~57% exact-match / ~84% name
  accuracy. The v6→v9 cascade lifts that to 59.3% in-browser; 81.7% is only reached by the
  synthesis pipeline with an external Llama-70B. It does not scale with fine-tuning alone —
  `PLAN.md` repeatedly shows that adding data does not help (Iter 22, 24, 41).
- **The synthesis pipeline (81.7%) does not run in the browser** — it needs an external Llama
  endpoint, which conflicts with the browser-only constraint. The browser config is the 59.3% cascade.
- **B4 (an in-browser synthesis model) is validated but not shipped.** A third synthesis GPT-2
  was proven viable (the trained model reached the oracle best-of-candidates ceiling), but the
  first candidate set mis-framed v9 and the corrected re-run could not finish on flaky HF Jobs
  infra. See `HANDOFF.md` and `PLAN.md` Iter 42.
- **Smart-home domain only.** After fine-tuning the model is specialized for smart-home tool-calling.
- **The misc domain is the weakest** (~57% in the synthesis pipeline).
- **fp16 on WebGPU needs onnxruntime-web ≥1.26** (transformers.js ≥4.2): on older builds GPT-2's
  LayerNorm variance overflows fp16. `export_onnx.py` keeps LayerNorm/gelu in fp32.

## Reproduce

**Browser demo (recommended):**

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool/web
npm install
npm run dev
```

**Reproduce the headline number (synth v2 = 78.7% → 81.7% with enum-snap):** the
constrained-bench artifacts live in the dataset repo `lifeart/smart-home-sft-v2`. The
synthesis pipeline: `training/bench_h1_con_cloud.py` (HF Jobs t4) produces base+H1 candidates,
then `training/bench_h1p11_synth.py` adds Llama emission + synthesis (free HF Inference router,
needs `HF_TOKEN`); `training/verify_enum_snap.py` deterministically verifies the enum-snap gain.

Heavy compute ran on HF Jobs
(`hf jobs uv run --flavor {t4-small|l40sx1|cpu-upgrade} --secrets HF_TOKEN --detach`).
Cumulative project spend is about **$27**.
