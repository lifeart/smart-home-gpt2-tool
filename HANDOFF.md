# Smart-Home GPT-2 — Project Handoff (through Iter 36)

Last updated end of the Iter 23–36 work block. This supersedes the earlier
Iter-22 handoff. Read `PLAN.md` for the full blow-by-blow; this file is the
fast orientation.

## What it is

GPT-2 124M (frozen architecture) fine-tuned for smart-home tool-calling,
deployable in-browser via WebGPU + transformers.js (ONNX). The base model
is never changed — only data, decoding tricks, and (Iter 34-35) the
position-embedding table size.

## Where things live

| Asset | Location |
|---|---|
| Code | `/Users/lifeart/Repos/gpt-tool` · GitHub `lifeart/smart-home-gpt2-tool` |
| Iteration log | `PLAN.md` (~2100 lines — every iteration, numbers, verdicts) |
| Models | HF Hub `lifeart/smart-home-gpt2-v{1..11}`, `-v6r-args[-v2]`, `-v12-ctx2048`, `-v13-ctx4096`, `-v14-ctx4096` |
| Datasets + results | HF Hub `lifeart/smart-home-sft-v2` (dataset repo) |
| Browser demo | `web/` (Vite + transformers.js) — `cd web && npm run dev` → http://localhost:5173/ |
| Training/bench scripts | `training/` |

## Current best results

### Tool-calling accuracy (exact-match = name + args, n=300 `sh_test.json`)

The Iter 22 verdict ("60% is a 124M ceiling") was **broken** in Iter 23-33
without changing the model — by composition, not a better single decoder:

| Config | Exact | Notes |
|---|---|---|
| v5 + con (Iter 22 prior ship) | 57.3% | single decoder |
| H1.2_con (v6 name + v9 args, clean-gate) | 59.3% | **browser-native, no external API** |
| H1.3_con (+ 2-way Llama pick) | 61.3% | needs Llama API |
| **synth v2 (BEST)** | **78.7%** | GPT-2 candidates → Llama-3.3-70B synthesis + canon |
| oracle ceiling | 87.3% | upper bound |

**The core finding:** the 124M plateau was a *single-decoder* ceiling, not
a knowledge ceiling. GPT-2 fine-tunes know the smart-home domain; Llama-70B
reasons but scores only 53% unprompted (doesn't know the function
inventory). Composed — GPT-2 v6/v9 emit candidate calls, Llama-3.3-70B
*synthesizes* the final call using them as evidence, then value
canonicalization — reaches 78.7%. See `PLAN.md` Iter 26-33.

### Context window — `v14-ctx4096` is the current best at every length

Name accuracy by prompt length (`bench_ctx_long.py`, `sh_test_long.json`):

| Model | Context | short | 1500 tok | 2500 tok | 3500 tok |
|---|---|---|---|---|---|
| v9 | 1024 | 81.0% | 58.7% | 32.6% | 20.9% |
| v13-ctx4096 | 4096 | 77.3% | 86.0% | 81.4% | 75.0% |
| **v14-ctx4096** | 4096 | **83.3%** | **93.6%** | **90.7%** | **89.5%** |

**Ship `v14-ctx4096` — it supersedes both v9 and v13.** It is the best
short-prompt model *and* the only one usable past 1024 tokens. (v12-ctx2048
still exists but is superseded too.)

How v14 was built (Iter 36, `train_hf_v14_ctx4096.py`): the failure mode of
Iter 35's v13 was that `extend_wpe` interpolated the *whole* 1024-row `wpe`
table to 4096, so even short prompts hit distorted positions (−3.7 pp tax).
v14 instead does **block-preserving** extension — rows 0-1023 are v9's `wpe`
*verbatim* (and frozen during SFT, `weight_decay=0` so AdamW can't decay
them), rows 1024-4095 interpolated; it inits from v9 (not base GPT-2); and
it trains on `sh_train_v14_long.json` (78k originals + 22k schema-padded
long rows). Result: the short-prompt tax is erased (+2.3 pp over v9) and
long-context accuracy is genuinely high. The Iter 35 "−1.8 pp per 2×
stretch" tax was an artifact of whole-table interpolation, not inherent.

### ONNX dtype accuracy (n=300)

fp16 is **lossless** vs fp32 (v9 and v14 both score fp16 == fp32 on name
accuracy; v14 = 83.3%). q8 loses ~3 pp (v14 q8 80.0%). **fp16 is the
browser default on WebGPU** — half the download (~330 MB vs ~660 MB),
~50% less GPU memory, same accuracy.

Gotcha (Iter 37): fp16-on-WebGPU needs **onnxruntime-web ≥1.26**
(transformers.js ≥4.2). On older builds the WebGPU fp16 path garbled output
— GPT-2's LayerNorm variance `(x-mean)^2` reaches ~9.3e6, overflowing
fp16's 65504 ceiling. `export_onnx.py` now keeps the LayerNorm/gelu ops in
fp32 (`op_block_list`) so the fp16 ONNX is numerically sound; WASM has no
fp16 kernels and uses fp32.

Raw-greedy args/exact bench numbers are too brittle to compare (q8
nonsensically scores them *higher*) — use name accuracy. Data:
`lifeart/smart-home-sft-v2/onnx_dtype_bench{,_smart-home-gpt2-v14-ctx4096}.json`.

## The browser demo (`web/`) — current state

Fully in-browser, no server inference. What it does now:

- **Model:** defaults to `lifeart/smart-home-gpt2-v14-ctx4096` (Iter 36 —
  4096-ctx, streamed from HF Hub; v9 stays selectable as the local-first
  1024-ctx option). transformers.js resolves both natively; `vite.config.js`
  has a `models404` plugin that returns a real 404 for missing `/models/`
  paths (without it, Vite's SPA fallback returns HTML and transformers.js
  crashes parsing it). v14 ONNX (fp32/fp16/q8) lives in its HF repo.
- **Rich-schema presets** (`web/presets.js` + `web/tool_schemas.js`): 32
  short realistic commands (3 candidate functions, ~648 tokens) plus a
  "Long context (v14)" category — 2 presets with 13 full schemas (~3000
  tokens) that exercise the 4096 window. Each prompt is the full-schema SFT
  format. `tool_schemas.js` = 123 function schemas (79 mined from training
  data, 44 synthesized from `tool_registry.json`).
- **Voice** (`web/voice.js`): in-browser Whisper (`Xenova/whisper-base`,
  transformers.js) — mic → transcribe (`task: translate`) → injects into
  the prompt → auto-Generate. No API.
- **Value canonicalization** (`web/canon.js`): JS port of
  `training/canon.py` — normalizes predicted arg values (12h→24h time, day
  plural, float rounding). Shown as the "Parsed tool call".
- **Dtype dropdown:** fp16 (default on WebGPU) / fp32 / q8. Retrieval
  pre-rank defaults OFF so presets exercise the full schemas.
- Constrained decoding / typed-args / wide-names: `web/grammar.js`
  (`JsonSchemaLogitsProcessor`), all default ON.

`web/synth.js` was built then deleted — the synthesis pipeline needs an
external Llama API, which conflicts with the browser-only constraint. The
synthesis work lives in `training/` only.

## Key scripts (`training/`)

- `bench_common.py` — shared exact-match scorer (port of `web/bench.js`).
- `grammar.py` — Python port of `web/grammar.js` constrained decoder.
- `canon.py` — value canonicalization.
- `bench_h1_two_stage.py`, `bench_h1_con.py`, `bench_h1_con_cloud.py` —
  two-stage decode (H1 / H1.2).
- `bench_h1p3..h1p13*.py` — the picker → synthesis iteration scripts.
  `bench_h1p11_synth.py` holds the winning `SYNTH_SYSTEM` prompt.
- `bench_model_emit.py` — generic HF-router model emitter.
- `bench_onnx_dtypes.py` — fp32/fp16/q8 ONNX accuracy bench.
- `train_hf_v12_ctx2048.py`, `train_hf_v13_ctx4096.py` — Iter 34-35
  `extend_wpe` whole-table interpolation (superseded by v14).
- `train_hf_v14_ctx4096.py` — **current ctx model.** Block-preserving wpe
  (frozen native prefix + interpolated tail), init-from-v9, dynamic
  per-row padding.
- `build_longctx.py` — builds the long-context train/test data by padding
  rich-schema prompts with distractor schemas.
- `bench_ctx_long.py` — per-length-bucket name/exact bench (KV-cache).
- `verify_ctx2048.py` — >1024-token context probe.
- `export_onnx.py` — fp32+fp16+q8 ONNX export (unchanged, works for the
  context-extended models too).

## Key learnings

1. 124M GPT-2 single-decoder ceiling ≈ 57% exact / ~84% name. Confirmed.
2. **Synthesis beats selection.** A picker can't exceed the oracle;
   Llama synthesizing from GPT-2 candidates can merge/fix → 78.7%.
3. Value canonicalization (time/day/float formats) = +2.7 pp oracle, free.
4. Context extension: **block-preserving** wpe is the right way. Keep the
   native 0-1023 rows verbatim (frozen, `weight_decay=0`) and only
   interpolate the tail — short prompts then pay zero tax. Iter 35's
   whole-table interpolation looked like an inherent "1.8 pp per 2×"
   stretch tax; Iter 36 (v14) showed that was an artifact and erased it.
   Pair it with real long-context training data (schema-padded prompts).
5. fp16 ONNX is lossless vs fp32 and is the browser default (half the
   download). Needs onnxruntime-web ≥1.26 — GPT-2's LayerNorm variance
   overflows fp16 otherwise; `export_onnx.py` keeps LayerNorm/gelu in fp32.
   q8 costs ~3 pp.
6. Rejected this session: H2/H2' rerank (name-bound), v6r-args retrain
   (SH-only data lost cross-domain), synth voting, 2-pass refinement,
   dual-picker, stronger emitter models. All documented in PLAN.md.
7. Operational: run heavy compute on HF Jobs (`hf jobs uv run --flavor
   {t4-small|l40sx1|cpu-upgrade} --secrets HF_TOKEN --detach`). Mac has
   16 GB — loading 2 models locally OOMs. Don't spawn duplicate jobs on a
   slow/scheduling job. transformers.js model repos need `config.json` +
   `tokenizer.json` + `onnx/` at the repo root.

## Parked / untried

- **Long context to 16k-32k.** Dense-window stretching beyond 4096 still
  is not the path. Within 4096, block-preserving wpe works great (v14).
  Past 4096 the penalty-free route remains **streaming KV-cache
  (StreamingLLM) + recurrent memory tokens (RMT) + retrieval** — each
  forward pass stays native length. ONNX Runtime Web 1.25+ ships
  FlashAttention-2 on WebGPU. Estimated <$50, browser-deployable. User
  decision: "decide later."
- Misc domain is the synthesis pipeline's weakest (~57% in H1.4) — a
  misc-specific args dataset or a direct-Llama fallback could help.
- Browser integration of the synthesis pipeline (needs a Llama endpoint).

## Budget

Cumulative project spend ≈ **$27**. Iter 23-33 was ~$0.10 (all free HF
Inference router). Iter 34-35 context training ≈ $8. Iter 36 (v14 train +
long bench) ≈ $3-4. ONNX/dtype benches ≈ $1. Per HF Jobs run: t4-small
~$0.10-0.30, l40sx1 ~$1-7 depending on seq_len and rows.

## Reproduce the headline number (synth v2 = 78.7%)

The constrained-bench artifacts are on `lifeart/smart-home-sft-v2`
(`iter23_h1_con_results.json`, etc.). To re-run the synthesis pipeline:
`training/bench_h1_con_cloud.py` (HF Jobs t4) produces base+H1 candidates;
then `bench_h1p4` / `bench_h1p11_synth.py` add Llama emission + synthesis
locally (HF free Inference router, needs `HF_TOKEN`).
