---
title: Smart-Home GPT-2 Tool-Calling
emoji: 🏠
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: GPT-2 124M → smart-home JSON tool calls, in-browser
---

# 🏠 Smart-Home GPT-2 · in-browser tool-calling

**Turn natural-language smart-home commands into structured JSON tool calls — with a
124M model that runs entirely in your browser. No server, no cloud, nothing leaves your device.**

```
"dim the living room lights to 20%"
   ↓  GPT-2 124M (smart-home fine-tune, in your browser)
{ "name": "dim_light", "arguments": { "room": "living room", "brightness_pct": 20 } }
```

- 🔒 **Private** — the model and the optional Whisper speech recognition run 100% in-browser
  via WebGPU + [transformers.js](https://github.com/huggingface/transformers.js). Weights stream
  once from the Hub, then cache. No API.
- 🎙️ **Voice in 99 languages** — speak a command, get the tool call.
- 🧩 **Always-valid JSON** — constrained decoding keeps the output schema-correct.
- 📚 **123 functions**, 4096-token context for long tool lists.

**Models** (streamed on first load, ~330 MB fp16, then cached):
[`lifeart/smart-home-gpt2-v14-ctx4096`](https://huggingface.co/lifeart/smart-home-gpt2-v14-ctx4096)
(default, 4096-ctx) · [`lifeart/smart-home-gpt2-v9`](https://huggingface.co/lifeart/smart-home-gpt2-v9)
(1024-ctx).

**Source & docs:** <https://github.com/lifeart/smart-home-gpt2-tool>

This Space hosts only the static front-end build (a few MB); the model weights are fetched
from the Hugging Face Hub at runtime.

<!--
=============================================================================
HOW TO BUILD AND PUBLISH THIS SPACE  (for the deploy step — not shown on-page)
=============================================================================

The Space is a *static* SDK Space: it serves whatever files are committed to
the Space repo, with index.html as the entry point. The deploy step must:

  1. Build the web/ app with the ROOT base path (Spaces serve from '/'):

       cd web
       npm ci
       SITE_BASE=/ npm run build      # → web/dist/  (a few MB, no models)

     The build excludes web/public/models/ automatically (vite.config.js
     builds from a models-free mirror of public/), so dist/ stays small even
     if a local model cache is present.

  2. Copy THIS file into the build output so it becomes the Space README with
     the YAML frontmatter above:

       cp web/space/README.md web/dist/README.md

  3. Upload the contents of web/dist/ (including README.md) to the Space repo
     huggingface.co/spaces/lifeart/smart-home-gpt2-tool. With huggingface_hub:

       from huggingface_hub import HfApi
       api = HfApi()
       api.upload_folder(
           folder_path="web/dist",
           repo_id="lifeart/smart-home-gpt2-tool",
           repo_type="space",
       )

     (Requires an HF token with write access — `huggingface-cli login` first.)

Notes for the deploy step:
  - The Space repo's root README.md must carry the frontmatter above or the
    Space will not configure correctly — that's why step 2 copies this file.
  - coi-serviceworker.js is included in dist/ and registered from index.html;
    it injects the COOP/COEP headers WASM multi-threading needs, since static
    Spaces cannot set HTTP headers. It self-resolves its registration path, so
    it works at the Space root ('/') unchanged.
  - GitHub Pages uses SITE_BASE=/smart-home-gpt2-tool/ instead; the HF Space
    MUST use SITE_BASE=/ (or unset, which defaults to '/').
=============================================================================
-->
