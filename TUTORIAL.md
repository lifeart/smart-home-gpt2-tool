# Туториал / Tutorial

> **Живая демонстрация (ничего ставить не нужно):**
> [GitHub Pages](https://lifeart.github.io/smart-home-gpt2-tool/) ·
> [Hugging Face Space](https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool)
>
> **Live demo (nothing to install):**
> [GitHub Pages](https://lifeart.github.io/smart-home-gpt2-tool/) ·
> [Hugging Face Space](https://huggingface.co/spaces/lifeart/smart-home-gpt2-tool)

Если вам нужен только готовый результат — откройте ссылки выше: модель запускается
целиком в браузере через WebGPU, сервера нет. Туториал ниже — для тех, кто хочет
запустить демонстрацию локально и расширить её под себя.

If you just want the running result, open the links above — the model runs entirely
in the browser via WebGPU, no server. The tutorial below is for running the demo
locally and extending it.

---

## 🇷🇺 Запуск браузерной демонстрации локально

### Шаг 1. Клонирование

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool
```

### Шаг 2. Запуск demo (`web/`)

Демонстрация — это Vite + transformers.js приложение. Нужен Node.js 18+.

```bash
cd web
npm install
npm run dev        # → http://localhost:5173/
```

При первом запуске браузер один раз скачает веса модели с HF Hub
(`lifeart/smart-home-gpt2-v14-ctx4096`, fp16 ≈ 330 МБ) и закэширует их.
Инференс идёт через WebGPU прямо в браузере — никакого сервера.

### Шаг 3. Пресеты

В интерфейсе есть готовые пресеты (`web/presets.js`): 32 коротких команды
(по 3 функции-кандидата) и категория «Long context (v14)» с полными схемами
на ~3000 токенов. Выберите пресет → **Generate** → внизу появится
«Parsed tool call» — финальный JSON-вызов.

### Шаг 4. Голосовой ввод

Нажмите кнопку микрофона. Whisper (`Xenova/whisper-base`) запускается прямо в
браузере, транскрибирует речь (`task: 'translate'` — любой из 99 языков сразу
в English) и подставляет текст в промпт с авто-Generate. Никакого API.

### Шаг 5. Переключатели

- **dtype:** fp16 (дефолт на WebGPU) / fp32 / q8. fp16 = fp32 по точности,
  но вдвое меньше скачивать. q8 теряет ~3 pp.
- **Constrained decoding / typed-args** — гарантируют валидный типизированный JSON.
- **Retrieval** — опционально, по умолчанию выключен; ранжирует функции-кандидаты
  через MiniLM (компромисс «скорость↔точность»).

### Шаг 6. Свой набор функций

Схемы функций живут в `web/tool_schemas.js` (123 схемы) и `data/tool_registry.json`.
Добавьте свою схему в том же формате и используйте её в пресете. Имена, близкие к
тем, что модель уже знает, работают сразу. Совсем новые функции требуют дообучения
(скрипты в `training/`).

### Шаг 7. Подключение к Home Assistant и др.

См. [`INTEGRATION.md`](INTEGRATION.md) — готовые рецепты для Home Assistant,
Zigbee2MQTT, ESPHome, HomeKit, Tuya, generic MQTT. Модель выдаёт
`{"name": ..., "arguments": {...}}`; ваша задача — смапить `name` на вызов платформы.

### Шаг 8. Воспроизведение бенчмарков

Все цифры проекта — в [`HANDOFF.md`](HANDOFF.md) и [`PLAN.md`](PLAN.md).
Браузерные бенчи лежат в `web/bench.js` / `web/voice_bench.js`; скрипты
тренировки и серверных бенчей — в `training/`. Тяжёлые прогоны запускались на
HF Jobs (`hf jobs uv run --flavor t4-small --secrets HF_TOKEN --detach`).

---

## 🇬🇧 Running the browser demo locally

### Step 1. Clone

```bash
git clone https://github.com/lifeart/smart-home-gpt2-tool
cd smart-home-gpt2-tool
```

### Step 2. Run the demo (`web/`)

The demo is a Vite + transformers.js app. Needs Node.js 18+.

```bash
cd web
npm install
npm run dev        # → http://localhost:5173/
```

On first run the browser downloads the model weights once from the HF Hub
(`lifeart/smart-home-gpt2-v14-ctx4096`, fp16 ≈ 330 MB) and caches them.
Inference runs through WebGPU in the browser — no server.

### Step 3. Presets

The UI ships presets (`web/presets.js`): 32 short commands (3 candidate
functions each) and a "Long context (v14)" category with ~3000-token full
schemas. Pick a preset → **Generate** → the "Parsed tool call" at the bottom
is the final JSON call.

### Step 4. Voice input

Click the microphone button. Whisper (`Xenova/whisper-base`) runs in the
browser, transcribes speech (`task: 'translate'` — any of 99 languages straight
to English) and injects the text into the prompt with auto-Generate. No API.

### Step 5. Toggles

- **dtype:** fp16 (default on WebGPU) / fp32 / q8. fp16 == fp32 on accuracy
  but half the download. q8 costs ~3 pp.
- **Constrained decoding / typed-args** — guarantee valid typed JSON.
- **Retrieval** — optional, default OFF; ranks candidate functions via MiniLM
  (a speed/accuracy trade-off).

### Step 6. Your own functions

Function schemas live in `web/tool_schemas.js` (123 schemas) and
`data/tool_registry.json`. Add yours in the same format and use it in a preset.
Names close to what the model already knows work out of the box; brand-new
functions need re-training (scripts in `training/`).

### Step 7. Home Assistant and others

See [`INTEGRATION.md`](INTEGRATION.md) for ready recipes — Home Assistant,
Zigbee2MQTT, ESPHome, HomeKit, Tuya, generic MQTT. The model emits
`{"name": ..., "arguments": {...}}`; your job is to map `name` to a platform call.

### Step 8. Reproducing benchmarks

Every project number is in [`HANDOFF.md`](HANDOFF.md) and [`PLAN.md`](PLAN.md).
Browser benches are in `web/bench.js` / `web/voice_bench.js`; training and
server-side bench scripts are in `training/`. Heavy runs used HF Jobs
(`hf jobs uv run --flavor t4-small --secrets HF_TOKEN --detach`).

---

## Troubleshooting

| симптом / symptom | причина / cause | фикс / fix |
|---|---|---|
| Модель выдаёт пробелы под fp16 / garbled fp16 output | старый onnxruntime-web | нужен transformers.js ≥4.2 (onnxruntime-web ≥1.26) |
| `404` на `/models/...` в браузере / 404 on `/models/` | Vite SPA-fallback | `vite.config.js` плагин `models404` уже это чинит |
| WebGPU недоступен / WebGPU unavailable | старый браузер | актуальный Chrome/Edge; иначе демо падает на WASM-fp32 |
| Долгая первая загрузка / slow first load | веса (~330 МБ fp16) качаются один раз | подождать; дальше из кэша (~1.4 с) |
| Голос не транскрибирует / voice not transcribing | нет доступа к микрофону | разрешить микрофон для localhost |
