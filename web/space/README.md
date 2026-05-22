---
title: Smart-Home GPT-2 Tool-Calling
emoji: 🏠
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: GPT-2 124M turns smart-home commands into JSON tool calls, 100% in-browser via WebGPU.
---

# Smart-Home GPT-2 · in-browser tool-calling demo

A 124M-parameter GPT-2, fine-tuned to read a natural-language smart-home
command and emit the matching **structured JSON tool call**. The model — and
the optional Whisper speech recognition — run **entirely in your browser** via
WebGPU + [transformers.js](https://github.com/huggingface/transformers.js).
No server, no API, nothing leaves your device.

- Models: [`lifeart/smart-home-gpt2-v14-ctx4096`](https://huggingface.co/lifeart/smart-home-gpt2-v14-ctx4096)
  (4096-ctx, default) and [`lifeart/smart-home-gpt2-v9`](https://huggingface.co/lifeart/smart-home-gpt2-v9)
  (1024-ctx). Streamed from the Hub on first load (~330 MB fp16), then cached
  by the browser.
- Source: <https://github.com/lifeart/smart-home-gpt2-tool>

This Space hosts only the static front-end build (a few MB). The model weights
are fetched from the Hugging Face Hub at runtime.

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
