# Smart-Home GPT-2 — Architecture

How the in-browser tool-calling stack is put together. For the product overview see the
[README](README.md); for the numbers behind every claim here see [BENCHMARKS.md](BENCHMARKS.md).

> **Try it first, no install:**
> [GitHub Pages demo](https://lifeart.github.io/smart-home-gpt2-tool/) ·
> [Hugging Face Space](https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool)

The base GPT-2 124M (OpenAI, 2019) is **never changed architecturally**. Only the data,
the decoding tricks, and (Iter 34–36) the size of the position-embedding table changed.
A single 124M decoder plateaus at ~57% exact-match, so the gains come from *composition*,
not a bigger model.

## 1. The v6→v9 cascade (browser-native, no external API)

The ~57% single-decoder ceiling was broken by composition:

- **v6** generates the function name;
- **v9** is an *arguments specialist* — given the name as a hint, it emits arguments only;
- a clean-gate picks the final call.

This is the `H1.2_con` cascade — **59.3% exact-match** (name + arguments), fully in the browser.

## 2. Constrained decoding (`web/grammar.js`)

A `JsonSchemaLogitsProcessor` masks logits at every generation step to only tokens
compatible with a JSON-schema FSM built from the candidate functions. It guarantees
syntactically valid JSON with correctly typed arguments per the function schema. On by
default. Typed-args sub-mode enforces per-key types (enums masked to allowed strings,
numeric keys disallow quotes, booleans only emit `true`/`false`).

## 3. Enum value-snapping (`web/canon.js`)

A predicted argument value is snapped to the nearest enum member from `tool_registry.json`
(`"gym"` → `"basement gym"`, `"living_room"` → `"living room"`). Only conservative rules —
case-insensitive exact match, space/underscore-insensitive, unique substring containment;
**no fuzzy matching**. On the synthesis pipeline this added +3.0 pp for free, with 0 regressions.

`web/canon.js` is a JS port of `training/canon.py`; it also normalizes time (12h→24h),
day plurals, and float rounding. The result is shown in the demo as the "Parsed tool call".

## 4. Retrieval pruning (`web/retrieval.js`, optional, default OFF)

MiniLM (`Xenova/all-MiniLM-L6-v2`) ranks the candidate functions and keeps the top-K *with
full typed schemas* in the prompt. It is a genuine speed/accuracy **trade** — at top-8 it
drops the gold function for ~4.7% of queries — so it ships opt-in. It helps most on
voice / ASR-noisy queries; on the multi-tool test set (which already ships the right
candidates) it ties or slightly loses.

## 5. Long context — `v14-ctx4096`

The window was extended 1024 → 4096 tokens via *block-preserving* extension of the `wpe`
table: rows 0–1023 are v9's verbatim (and frozen during SFT, `weight_decay=0`), rows
1024–4095 are interpolated. This erased the short-prompt tax that whole-table interpolation
(v13) suffered, so `v14-ctx4096` is the best model at *every* prompt length. It is the
current ship model — short prompts keep v9's exact native embeddings, while dozens of full
tool schemas now fit in one prompt.

## 6. fp16 — the WebGPU default

fp16 weights are lossless vs fp32 (identical name accuracy) but the download is half
(~330 MB vs ~660 MB) and uses ~50% less GPU memory. q8 costs ~3 pp. The WASM backend uses
fp32 (there are no fp16 kernels under WASM). fp16 on WebGPU needs onnxruntime-web ≥1.26
(transformers.js ≥4.2); `export_onnx.py` keeps LayerNorm/gelu in fp32 so the fp16 graph
stays numerically sound. See [QUANTIZATION.md](QUANTIZATION.md) for CPU/server options.

## API-side pipeline (not in-browser)

For research there is a `v6→v9→synth` pipeline: the GPT-2 candidates are handed to
Llama-3.3-70B, which *synthesizes* the final call, followed by value canonicalization.
This reaches **81.7%** — but it needs an external Llama API, so it is **not** part of the
browser app (it conflicts with the browser-only constraint). See [BENCHMARKS.md](BENCHMARKS.md)
and `PLAN.md` Iter 26–33, 38.

## Voice pipeline

In the browser demo (`web/voice.js`) voice is handled entirely locally via transformers.js Whisper:

```
Speech (mic, any of 99 languages)
   ↓  Whisper (Xenova/whisper-base), task: 'translate'
English text
   ↓  GPT-2 smart-home (v6→v9 cascade, in-browser)
JSON tool call
```

`task: 'translate'` means speech in any language is transcribed straight to English — no
language switch needed. No cloud, no API.

## Browser demo internals (`web/`)

Vite + transformers.js; all inference runs in the browser, no server.

- **Model** — defaults to `lifeart/smart-home-gpt2-v14-ctx4096` (4096-token window, streamed
  from the HF Hub); `v9` is selectable as the local-first 1024-token option.
- **Presets** (`web/presets.js` + `web/tool_schemas.js`) — 32 short realistic commands (3
  candidate functions each, ~648 tokens) plus a "Long context (v14)" category with 13 full
  schemas (~3000 tokens) that exercise the 4096 window. `tool_schemas.js` holds 123 function schemas.
- **Voice** (`web/voice.js`) — in-browser Whisper, mic → transcribe → inject → auto-Generate.
- **Value canonicalization** (`web/canon.js`) — see §3 above; shown as the "Parsed tool call".
- **Toggles** — constrained decoding, typed-args, and retrieval are switchable in the UI.
- **Dtype dropdown** — fp16 (default on WebGPU) / fp32 / q8.

Run locally:

```bash
cd web
npm install
npm run dev    # → http://localhost:5173/
```
