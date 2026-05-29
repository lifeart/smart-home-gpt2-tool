# 🏠 Smart-Home GPT-2

**Turn natural-language smart-home commands into structured JSON tool calls — with a 124M model that runs entirely in your browser. No server, no cloud, nothing leaves your device.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live demo — GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-2ea44f)](https://lifeart.github.io/smart-home-gpt2-tool/)
[![Live demo — HF Space](https://img.shields.io/badge/demo-HuggingFace%20Space-ffce00)](https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool)
[![Models on HF Hub](https://img.shields.io/badge/models-HF%20Hub-blue)](https://huggingface.co/lifeart)

> 🍴 **Fork.** Builds on **[barometech/smart-home-gpt2](https://github.com/barometech/smart-home-gpt2)** by **Pavel D. Popovich** (barometech) — the original GPT-2 smart-home fine-tune, dataset and training pipeline. This repo continues it with in-browser inference, the v6→v9 cascade, long-context `v14`, and voice. See [Acknowledgments](#acknowledgments).
>
> 🇷🇺 Описание на русском — в [TUTORIAL.md](TUTORIAL.md).

```
"dim the living room lights to 20%"
   ↓  GPT-2 124M (smart-home fine-tune, in your browser)
{ "name": "dim_light", "arguments": { "room": "living room", "brightness_pct": 20 } }
```

## ▶️ Try it now

Nothing to install — the model runs in your browser via WebGPU and is cached after the first load.

- **GitHub Pages:** <https://lifeart.github.io/smart-home-gpt2-tool/>
- **Hugging Face Space:** <https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool>

Built-in command presets, voice input (in-browser Whisper), and an fp16 / fp32 / q8 precision switch.

## Why this project

- 🔒 **Private by design.** Inference runs 100% in the browser (WebGPU + [transformers.js](https://github.com/huggingface/transformers.js)). No server, no cloud API, no data leaving the device. Weights stream once from the HF Hub and are cached.
- ⚡ **Small and fast.** A 124M-parameter model — ~330 MB as fp16, runs on consumer GPUs through WebGPU. Hosting is a static site (the live demos are free GitHub Pages / HF Space builds).
- 🎙️ **Voice in any language.** In-browser Whisper transcribes speech in 99 languages straight to English, then the model emits the tool call. No API.
- 🧩 **Always-valid JSON.** Constrained decoding masks the output to a JSON schema, so you get syntactically valid `{"name", "arguments"}` with correctly typed arguments — every time.
- 📚 **123 functions out of the box**, with a 4096-token context window (`v14`) so dozens of full tool schemas fit in one prompt.
- 🔌 **Pluggable.** The model emits a platform-neutral tool call; map it to Home Assistant, Zigbee2MQTT, ESPHome, Apple HomeKit, Tuya or plain MQTT — see [INTEGRATION.md](INTEGRATION.md).

## How it works

A single 124M decoder plateaus around 57% exact-match, so accuracy comes from **composition and decoding tricks**, not a bigger model:

- **v6→v9 cascade** — `v6` predicts the function name, `v9` (an arguments specialist) fills the arguments, a clean-gate picks the final call. Runs entirely in the browser.
- **Constrained decoding** keeps the JSON valid and typed.
- **Enum value-snapping** maps a loose value to the nearest registry enum (`"gym"` → `"basement gym"`).
- **`v14-ctx4096`** extends the context window to 4096 tokens for long tool lists, with no short-prompt accuracy tax.

Full design and rationale: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quick start

Use the hosted demos above, or run the browser app locally:

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool/web
npm install
npm run dev          # → http://localhost:5173/
```

To wire the output into a real smart-home stack, follow **[INTEGRATION.md](INTEGRATION.md)**.

## Accuracy at a glance

It's a 124M model with a **real, honest ceiling** — the numbers are not oversold.

| Configuration | Metric | Score |
|---|---|---:|
| **v6→v9 cascade** (in-browser, no external API) | exact-match, n=300 | **59.3%** |
| `v14-ctx4096` (default model) | name accuracy, short prompts | **83.3%** |
| `v14-ctx4096` | name accuracy, ~3500-token prompts | **89.5%** |
| Research synthesis pipeline (needs external Llama-70B) | exact-match | 81.7% |

The in-browser app ships the 59.3% cascade; the 81.7% figure needs an external Llama endpoint and is research-only. Full tables, the 124M-ceiling analysis, dtype trade-offs, and how to reproduce the headline number: **[BENCHMARKS.md](BENCHMARKS.md)**.

## Make it yours

- **Connect to a real home** — [INTEGRATION.md](INTEGRATION.md) has recipes for Home Assistant, Zigbee2MQTT, ESPHome, HomeKit, Tuya and MQTT.
- **Add your own functions** — schemas live in `web/tool_schemas.js` and `data/tool_registry.json`. Names close to ones the model already knows work right away; brand-new functions need re-training (`training/`).
- **Other languages** — Whisper's `task: 'translate'` handles 99 languages automatically; nothing to change.
- **Shrink / speed up** — fp16 (default) and q8 ship in the browser; CPU/server quantization is covered in [QUANTIZATION.md](QUANTIZATION.md).

## Documentation

| Doc | What's in it |
|---|---|
| [TUTORIAL.md](TUTORIAL.md) | From clone to your own function set (RU + EN) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The cascade, constrained decoding, enum-snap, retrieval, long context, fp16, the voice pipeline |
| [BENCHMARKS.md](BENCHMARKS.md) | Full accuracy tables, the 124M-ceiling analysis, limitations, reproduction |
| [INTEGRATION.md](INTEGRATION.md) | Plugging the model into Home Assistant, Zigbee2MQTT, ESPHome, HomeKit, MQTT |
| [QUANTIZATION.md](QUANTIZATION.md) | CPU / server-side quantization options |
| [HANDOFF.md](HANDOFF.md) · [PLAN.md](PLAN.md) | The full iteration log — source of every number |

## Models

Streamed from the HF Hub on first load, then cached by the browser:

- **[`lifeart/smart-home-gpt2-v14-ctx4096`](https://huggingface.co/lifeart/smart-home-gpt2-v14-ctx4096)** — default, 4096-token window.
- **[`lifeart/smart-home-gpt2-v9`](https://huggingface.co/lifeart/smart-home-gpt2-v9)** — 1024-token window.

## License

MIT — see [LICENSE](LICENSE). Fork freely.

## Citation

```
@misc{popovich_smart_home_gpt2_2026,
  title  = {Smart-Home GPT-2: in-browser tool-calling on a 124M model},
  author = {Popovich, Pavel D.},
  year   = {2026},
  url    = {https://github.com/lifeart/smart-home-gpt2-tool}
}
```

## Acknowledgments

This project is a fork of **[barometech/smart-home-gpt2](https://github.com/barometech/smart-home-gpt2)** by **Pavel D. Popovich** (barometech / Tekhnozhrets) — the original GPT-2 smart-home fine-tune, dataset and training pipeline, released under the MIT License (see [LICENSE](LICENSE)). The upstream repository is linked as a git remote in this fork.
