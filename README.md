# 🏠 Smart-Home GPT-2

**Превращает команды для умного дома на естественном языке в JSON tool-call'ы — 124M-моделью, целиком в браузере, без сервера и без облака.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live demo — GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-2ea44f)](https://lifeart.github.io/smart-home-gpt2-tool/)
[![Live demo — HF Space](https://img.shields.io/badge/demo-HuggingFace%20Space-ffce00)](https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool)
[![Models on HF Hub](https://img.shields.io/badge/models-HF%20Hub-blue)](https://huggingface.co/lifeart)

## ▶️ Live demo / Живая демонстрация

Запускается полностью у вас в браузере через WebGPU — модель скачивается с HF Hub один раз и кэшируется. Сервера для инференса нет.

- **GitHub Pages:** <https://lifeart.github.io/smart-home-gpt2-tool/>
- **Hugging Face Space:** <https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool>

Есть встроенные пресеты, голосовой ввод (Whisper в браузере) и переключатель точности (fp16 / fp32 / q8).

---

## 🇷🇺 Описание

- **Содержание**
  - [Что это](#что-это)
  - [Архитектура](#архитектура)
  - [Бенчмарки](#бенчмарки)
  - [Браузерная демонстрация](#браузерная-демонстрация)
  - [Голосовой пайплайн](#голосовой-пайплайн)
  - [Воспроизведение](#воспроизведение)
  - [Расширение под себя](#расширение-под-себя)
  - [Ограничения](#ограничения)
- **Дополнительные документы:**
  - [TUTORIAL.md](TUTORIAL.md) — от клонирования до собственного набора функций
  - [INTEGRATION.md](INTEGRATION.md) — подключение к Home Assistant, Zigbee2MQTT, ESPHome, HomeKit, MQTT
  - [QUANTIZATION.md](QUANTIZATION.md) — варианты квантизации модели для CPU
  - [HANDOFF.md](HANDOFF.md) / [PLAN.md](PLAN.md) — полный лог итераций (источник всех цифр)
- 🇬🇧 [English description below ↓](#english)

---

### Что это

GPT-2 124M (OpenAI, 2019) — архитектура заморожена, не меняется — дообученная превращать команды умного дома на естественном языке в JSON tool-call'ы:

```
"dim the living room lights to 20%"
  ↓ GPT-2 124M (smart-home fine-tune)
{"name": "dim_light", "arguments": {"room": "living room", "brightness_pct": 20}}
```

Главная особенность — модель работает **целиком в браузере** через WebGPU + transformers.js (ONNX). Нет сервера, нет облачного API, нет отправки данных наружу. Веса один раз стримятся с HF Hub и кэшируются.

Это исследовательский проект, и его история — намеренно честная, включая отрицательные результаты. Ключевой вывод: у модели на 124M параметров есть **реальный потолок точности**, и выигрыши пришли не от модели побольше, а от композиции и трюков с декодированием. Все цифры ниже взяты из [`HANDOFF.md`](HANDOFF.md) и [`PLAN.md`](PLAN.md) и не приукрашены.

---

### Архитектура

Базовый GPT-2 124M ни разу не меняли по архитектуре. Менялись только данные, трюки декодирования и (Iter 34–36) размер таблицы позиционных эмбеддингов.

**1. Каскад v6→v9 (браузерный, без внешнего API).** Один декодер на 124M упирается в потолок ~57% exact-match. Обойти его удалось *композицией*, а не более крупной моделью:

- **v6** генерирует имя функции;
- **v9** — специалист по аргументам: ему подсказывают имя и просят выдать только аргументы;
- clean-gate выбирает финальный вызов.

Так получается каскад `H1.2_con` — **59.3% exact-match** (имя + аргументы), полностью в браузере.

**2. Constrained decoding** (`web/grammar.js`) — `JsonSchemaLogitsProcessor` гарантирует синтаксически валидный JSON и аргументы правильного типа по схеме функции. Включён по умолчанию.

**3. Enum value-snapping** (`web/canon.js`) — предсказанное значение аргумента подтягивается к ближайшему enum-члену из `tool_registry.json` (`"gym"` → `"basement gym"`, `"living_room"` → `"living room"`). Только консервативные правила (точное совпадение без учёта регистра, пробел/подчёркивание, уникальное вхождение подстроки), без fuzzy-матчинга. На синтез-пайплайне это дало +3.0 pp бесплатно, 0 регрессий.

**4. Retrieval-прунинг** (`web/retrieval.js`, опционально, по умолчанию выключен) — MiniLM ранжирует кандидатов-функции и оставляет в промпте top-K *с полными типизированными схемами*. Это компромисс «скорость↔точность» (на top-8 теряет нужную функцию для ~4.7% запросов), поэтому он opt-in.

**5. Длинный контекст — `v14-ctx4096`.** Окно расширено с 1024 до 4096 токенов методом *block-preserving* расширения таблицы `wpe`: строки 0–1023 берутся из v9 дословно (и заморожены при дообучении, `weight_decay=0`), строки 1024–4095 интерполируются. Это убрало «налог» на короткие промпты, который был у наивной интерполяции всей таблицы (v13). `v14-ctx4096` — текущая ship-модель.

**6. fp16 — дефолт на WebGPU.** fp16-веса по точности неотличимы от fp32 (одинаковый name-accuracy), но скачивание вдвое меньше (~330 МБ против ~660 МБ) и ~50% меньше GPU-памяти. q8 теряет ~3 pp. WASM-бэкенд использует fp32 (под WASM нет fp16-ядер).

**API-сайд пайплайн (не в браузере).** Для исследований есть пайплайн `v6→v9→синтез`: GPT-2-кандидаты отдаются Llama-3.3-70B, которая *синтезирует* финальный вызов, затем канонизация значений. Это даёт **81.7%** — но требует внешнего Llama-API, поэтому в браузер не входит (конфликт с условием «только браузер»). Подробности — `PLAN.md`, Iter 26–33, 38.

---

### Бенчмарки

Все числа — из [`HANDOFF.md`](HANDOFF.md) / [`PLAN.md`](PLAN.md).

**Tool-calling accuracy (exact-match = имя + аргументы, n=300, `sh_test.json`):**

| Конфигурация | Exact | Примечание |
|---|---:|---|
| v5 + constrained (Iter 22, прежний ship) | 57.3% | один декодер |
| **H1.2_con — каскад v6→v9** | **59.3%** | **браузерный, без внешнего API** |
| H1.3_con (+ 2-way Llama-пик) | 61.3% | нужен Llama-API |
| synth v2 (Iter 32) | 78.7% | GPT-2-кандидаты → синтез Llama-3.3-70B + канонизация |
| **synth v2 + enum-snap (лучший)** | **81.7%** | + enum value-snapping (Iter 38) |
| oracle ceiling | 87.3% | верхняя граница |

Главный вывод: потолок ~57% — это потолок *одного декодера*, а не потолок знаний. Дообученные GPT-2 знают домен; Llama-70B рассуждает, но без подсказки набирает всего ~53% (не знает инвентарь функций). В композиции — GPT-2 выдаёт кандидатов, Llama их синтезирует — получается 78.7%, а с enum-snap 81.7%.

**Контекстное окно — name-accuracy по длине промпта** (`bench_ctx_long.py`, `sh_test_long.json`):

| Модель | Окно | короткий | 1500 ток | 2500 ток | 3500 ток |
|---|---:|---:|---:|---:|---:|
| v9 | 1024 | 81.0% | 58.7% | 32.6% | 20.9% |
| v13-ctx4096 | 4096 | 77.3% | 86.0% | 81.4% | 75.0% |
| **v14-ctx4096** | 4096 | **83.3%** | **93.6%** | **90.7%** | **89.5%** |

`v14-ctx4096` — лучшая модель и на коротких, и на длинных промптах; она вытесняет и v9, и v13.

**ONNX dtype (n=300):** fp16 по name-accuracy совпадает с fp32 (v14 = 83.3% на обоих); q8 — 80.0% (~−3 pp).

---

### Браузерная демонстрация

Каталог `web/` — Vite + transformers.js. Инференс целиком в браузере, сервера нет.

- **Модель:** по умолчанию `lifeart/smart-home-gpt2-v14-ctx4096` (4096-токенное окно, стримится с HF Hub); v9 доступна как локально-ориентированный 1024-токенный вариант.
- **Пресеты** (`web/presets.js` + `web/tool_schemas.js`): 32 коротких реалистичных команды (по 3 функции-кандидата, ~648 токенов) плюс категория «Long context (v14)» — пресеты с 13 полными схемами (~3000 токенов), которые задействуют окно 4096. `tool_schemas.js` содержит 123 схемы функций.
- **Голос** (`web/voice.js`): Whisper в браузере (`Xenova/whisper-base`, transformers.js) — микрофон → транскрипция (`task: translate`, любой язык → English) → вставка в промпт → авто-Generate. Без API.
- **Канонизация значений** (`web/canon.js`): JS-порт `training/canon.py` — нормализует время (12ч→24ч), дни, округление float, и делает enum value-snapping. Показывается как «Parsed tool call».
- **Переключатель dtype:** fp16 (дефолт на WebGPU) / fp32 / q8.
- **Constrained decoding / typed-args / retrieval** — переключатели в интерфейсе.

Запуск локально:

```bash
cd web
npm install
npm run dev    # → http://localhost:5173/
```

---

### Голосовой пайплайн

В браузерной демонстрации (`web/voice.js`) голос обрабатывается полностью локально через transformers.js Whisper:

```
Речь (микрофон, любой из 99 языков)
  ↓ Whisper (Xenova/whisper-base), task: 'translate'
English-текст
  ↓ GPT-2 smart-home (каскад v6→v9, в браузере)
JSON tool call
```

`task: 'translate'` означает, что речь на любом языке транскрибируется сразу в English — менять язык не нужно. Никакого облака, никаких API.

---

### Воспроизведение

**Браузерная демонстрация (рекомендуется):**

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool/web
npm install
npm run dev
```

**Воспроизвести headline-число (synth v2 = 78.7% → 81.7% с enum-snap):**
артефакты constrained-бенча лежат в датасет-репозитории `lifeart/smart-home-sft-v2`. Пайплайн синтеза:
`training/bench_h1_con_cloud.py` (HF Jobs t4) генерирует base+H1-кандидатов, затем `training/bench_h1p11_synth.py` добавляет эмиссию Llama и синтез (бесплатный HF Inference router, нужен `HF_TOKEN`); `training/verify_enum_snap.py` детерминированно проверяет прибавку enum-snap.

Тяжёлые вычисления запускались на HF Jobs (`hf jobs uv run --flavor {t4-small|l40sx1|cpu-upgrade} --secrets HF_TOKEN --detach`). Суммарный бюджет проекта — около **$27**.

---

### Расширение под себя

**1. Подключение к реальному дому.** См. [`INTEGRATION.md`](INTEGRATION.md) — рецепты для Home Assistant, Zigbee2MQTT, ESPHome, Apple HomeKit, Tuya и generic MQTT. Модель выдаёт `{"name": ..., "arguments": {...}}`; ваша задача — смапить `name` на вызов платформы.

**2. Свои функции.** Схемы функций живут в `web/tool_schemas.js` и `data/tool_registry.json`. Имена, близкие к уже знакомым модели, работают сразу. Совсем новые функции требуют дообучения (`training/`).

**3. Свой язык.** Whisper понимает 99 языков; `task: 'translate'` переводит любую речь в English автоматически — менять ничего не нужно.

**4. Уменьшить / ускорить.** В браузере уже есть fp16 (дефолт) и q8. Для CPU-инференса варианты квантизации разобраны в [`QUANTIZATION.md`](QUANTIZATION.md).

---

### Ограничения

Честно, без приукрашивания:

- **Потолок 124M-модели реален.** Один декодер упирается в ~57% exact-match / ~84% name-accuracy. Каскад v6→v9 поднимает до 59.3% в браузере; 81.7% достигается только синтез-пайплайном с внешней Llama-70B. Это не масштабируется одним лишь дообучением — `PLAN.md` неоднократно показывает, что наращивание данных не помогает (Iter 22, 24, 41).
- **Синтез-пайплайн (81.7%) не работает в браузере** — ему нужен внешний Llama-эндпойнт, что противоречит условию «только браузер». В браузере доступен каскад на 59.3%.
- **B4 (in-browser синтез-модель) валидирована, но не зашипана.** Идея третьей GPT-2-модели синтеза подтвердилась (обученная модель достигла oracle-потолка кандидатов), но первый набор кандидатов неверно «раскадрировал» v9, и финальную перепрогонку не удалось завершить на нестабильной HF Jobs инфраструктуре. Подробности — `HANDOFF.md` и `PLAN.md`, Iter 42.
- **Только домен умного дома.** После дообучения модель специализирована под smart-home tool-calling.
- **Domain misc — самый слабый** (~57% в синтез-пайплайне).
- **fp16 на WebGPU требует onnxruntime-web ≥1.26** (transformers.js ≥4.2): на старых сборках LayerNorm-дисперсия GPT-2 переполняет fp16. `export_onnx.py` держит LayerNorm/gelu в fp32.

---

<a id="english"></a>

## 🇬🇧 English

**Turns natural-language smart-home commands into JSON tool calls — with a 124M model, entirely in the browser, no server and no cloud.**

### What it is

GPT-2 124M (OpenAI, 2019) — architecture frozen, never changed — fine-tuned to turn natural-language smart-home commands into JSON tool calls:

```
"dim the living room lights to 20%"
  ↓ GPT-2 124M (smart-home fine-tune)
{"name": "dim_light", "arguments": {"room": "living room", "brightness_pct": 20}}
```

The defining feature: it runs **100% in the browser** via WebGPU + transformers.js (ONNX). No server, no cloud API, no data leaving the device. Weights stream once from the HF Hub and are cached.

This is a research project, and its history is deliberately honest, negative results included. The core finding: a 124M-parameter model has a **real accuracy ceiling**, and the gains came not from a bigger model but from composition and decoding tricks. Every number below is sourced from [`HANDOFF.md`](HANDOFF.md) / [`PLAN.md`](PLAN.md) and is not oversold.

### Architecture

The base GPT-2 124M is never changed architecturally. Only data, decoding tricks and (Iter 34–36) the position-embedding table size changed.

**1. The v6→v9 cascade (browser-native, no external API).** A single 124M decoder plateaus at ~57% exact-match. That ceiling was broken by *composition*, not a bigger model:

- **v6** generates the function name;
- **v9** is an *arguments specialist* — given the name as a hint, it emits arguments only;
- a clean-gate picks the final call.

This is the `H1.2_con` cascade — **59.3% exact-match** (name + args), fully in the browser.

**2. Constrained decoding** (`web/grammar.js`) — a `JsonSchemaLogitsProcessor` guarantees syntactically valid JSON with correctly typed arguments per the function schema. On by default.

**3. Enum value-snapping** (`web/canon.js`) — a predicted argument value is snapped to the nearest enum member from `tool_registry.json` (`"gym"` → `"basement gym"`, `"living_room"` → `"living room"`). Only conservative rules (case-insensitive exact, space/underscore-insensitive, unique substring containment); no fuzzy matching. On the synthesis pipeline this added +3.0 pp for free, 0 regressions.

**4. Retrieval pruning** (`web/retrieval.js`, optional, default OFF) — MiniLM ranks the candidate functions and keeps the top-K *with full typed schemas* in the prompt. It is a genuine speed/accuracy *trade* (at top-8 it drops the gold function for ~4.7% of queries), so it ships opt-in.

**5. Long context — `v14-ctx4096`.** The window was extended 1024→4096 tokens via *block-preserving* extension of the `wpe` table: rows 0–1023 are v9's verbatim (and frozen during SFT, `weight_decay=0`), rows 1024–4095 interpolated. This erased the short-prompt tax that whole-table interpolation (v13) suffered. `v14-ctx4096` is the current ship model.

**6. fp16 — the WebGPU default.** fp16 weights are lossless vs fp32 (identical name accuracy) but the download is half (~330 MB vs ~660 MB) and uses ~50% less GPU memory. q8 costs ~3 pp. The WASM backend uses fp32 (no fp16 kernels under WASM).

**API-side pipeline (not in-browser).** For research there is a `v6→v9→synth` pipeline: GPT-2 candidates are handed to Llama-3.3-70B, which *synthesizes* the final call, followed by value canonicalization. This reaches **81.7%** — but it needs an external Llama API, so it is not part of the browser app (conflicts with the browser-only constraint). See `PLAN.md` Iter 26–33, 38.

### Benchmarks

All numbers from [`HANDOFF.md`](HANDOFF.md) / [`PLAN.md`](PLAN.md).

**Tool-calling accuracy (exact-match = name + args, n=300, `sh_test.json`):**

| Config | Exact | Notes |
|---|---:|---|
| v5 + constrained (Iter 22, prior ship) | 57.3% | single decoder |
| **H1.2_con — v6→v9 cascade** | **59.3%** | **browser-native, no external API** |
| H1.3_con (+ 2-way Llama pick) | 61.3% | needs Llama API |
| synth v2 (Iter 32) | 78.7% | GPT-2 candidates → Llama-3.3-70B synthesis + canon |
| **synth v2 + enum-snap (best)** | **81.7%** | + enum value-snapping (Iter 38) |
| oracle ceiling | 87.3% | upper bound |

The core finding: the ~57% plateau is a *single-decoder* ceiling, not a knowledge ceiling. The fine-tuned GPT-2 models know the domain; Llama-70B reasons but scores only ~53% unprompted (it doesn't know the function inventory). Composed — GPT-2 emits candidates, Llama synthesizes — it reaches 78.7%, and 81.7% with enum-snap.

**Context window — name accuracy by prompt length** (`bench_ctx_long.py`, `sh_test_long.json`):

| Model | Window | short | 1500 tok | 2500 tok | 3500 tok |
|---|---:|---:|---:|---:|---:|
| v9 | 1024 | 81.0% | 58.7% | 32.6% | 20.9% |
| v13-ctx4096 | 4096 | 77.3% | 86.0% | 81.4% | 75.0% |
| **v14-ctx4096** | 4096 | **83.3%** | **93.6%** | **90.7%** | **89.5%** |

`v14-ctx4096` is the best model at every length — it supersedes both v9 and v13.

**ONNX dtype (n=300):** fp16 matches fp32 on name accuracy (v14 = 83.3% on both); q8 = 80.0% (~−3 pp).

### Browser demo

The `web/` directory — Vite + transformers.js. All inference runs in the browser, no server.

- **Model:** defaults to `lifeart/smart-home-gpt2-v14-ctx4096` (4096-token window, streamed from the HF Hub); v9 is selectable as the local-first 1024-token option.
- **Presets** (`web/presets.js` + `web/tool_schemas.js`): 32 short realistic commands (3 candidate functions each, ~648 tokens) plus a "Long context (v14)" category — presets with 13 full schemas (~3000 tokens) that exercise the 4096 window. `tool_schemas.js` holds 123 function schemas.
- **Voice** (`web/voice.js`): in-browser Whisper (`Xenova/whisper-base`, transformers.js) — mic → transcribe (`task: translate`, any language → English) → inject into the prompt → auto-Generate. No API.
- **Value canonicalization** (`web/canon.js`): a JS port of `training/canon.py` — normalizes time (12h→24h), day plurals, float rounding, and does enum value-snapping. Shown as the "Parsed tool call".
- **Dtype dropdown:** fp16 (default on WebGPU) / fp32 / q8.
- **Constrained decoding / typed-args / retrieval** — toggles in the UI.

Run locally:

```bash
cd web
npm install
npm run dev    # → http://localhost:5173/
```

### Voice pipeline

In the browser demo (`web/voice.js`) voice is handled entirely locally via transformers.js Whisper:

```
Speech (mic, any of 99 languages)
  ↓ Whisper (Xenova/whisper-base), task: 'translate'
English text
  ↓ GPT-2 smart-home (v6→v9 cascade, in-browser)
JSON tool call
```

`task: 'translate'` means speech in any language is transcribed straight to English — no language switch needed. No cloud, no API.

### Reproduce

**Browser demo (recommended):**

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool/web
npm install
npm run dev
```

**Reproduce the headline number (synth v2 = 78.7% → 81.7% with enum-snap):**
the constrained-bench artifacts live in the dataset repo `lifeart/smart-home-sft-v2`. The synthesis pipeline:
`training/bench_h1_con_cloud.py` (HF Jobs t4) produces base+H1 candidates, then `training/bench_h1p11_synth.py` adds Llama emission + synthesis (free HF Inference router, needs `HF_TOKEN`); `training/verify_enum_snap.py` deterministically verifies the enum-snap gain.

Heavy compute ran on HF Jobs (`hf jobs uv run --flavor {t4-small|l40sx1|cpu-upgrade} --secrets HF_TOKEN --detach`). Cumulative project spend is about **$27**.

### Extending it

**1. Connect to a real home.** See [`INTEGRATION.md`](INTEGRATION.md) — recipes for Home Assistant, Zigbee2MQTT, ESPHome, Apple HomeKit, Tuya and generic MQTT. The model emits `{"name": ..., "arguments": {...}}`; your job is to map `name` to a platform call.

**2. Your own functions.** Function schemas live in `web/tool_schemas.js` and `data/tool_registry.json`. Names close to what the model already knows work right away. Brand-new functions need re-training (`training/`).

**3. Your language.** Whisper handles 99 languages; `task: 'translate'` translates any speech to English automatically — nothing to change.

**4. Shrink / speed up.** The browser already offers fp16 (default) and q8. For CPU inference, quantization options are covered in [`QUANTIZATION.md`](QUANTIZATION.md).

### Limitations

Honest, not oversold:

- **The 124M ceiling is real.** A single decoder plateaus at ~57% exact-match / ~84% name accuracy. The v6→v9 cascade lifts that to 59.3% in-browser; 81.7% is only reached by the synthesis pipeline with an external Llama-70B. It does not scale with fine-tuning alone — `PLAN.md` repeatedly shows that adding data does not help (Iter 22, 24, 41).
- **The synthesis pipeline (81.7%) does not run in the browser** — it needs an external Llama endpoint, which conflicts with the browser-only constraint. The browser config is the 59.3% cascade.
- **B4 (an in-browser synthesis model) is validated but not shipped.** A third synthesis GPT-2 was proven viable (the trained model reached the oracle best-of-candidates ceiling), but the first candidate set mis-framed v9 and the corrected re-run could not finish on flaky HF Jobs infra. See `HANDOFF.md` and `PLAN.md` Iter 42.
- **Smart-home domain only.** After fine-tuning the model is specialized for smart-home tool-calling.
- **The misc domain is the weakest** (~57% in the synthesis pipeline).
- **fp16 on WebGPU needs onnxruntime-web ≥1.26** (transformers.js ≥4.2): on older builds GPT-2's LayerNorm variance overflows fp16. `export_onnx.py` keeps LayerNorm/gelu in fp32.

### License

MIT — see [LICENSE](LICENSE). Fork freely.

### Citation

```
@misc{popovich_smart_home_gpt2_2026,
  title  = {Smart-Home GPT-2: in-browser tool-calling on a 124M model},
  author = {Popovich, Pavel D.},
  year   = {2026},
  url    = {https://github.com/lifeart/smart-home-gpt2-tool}
}
```
