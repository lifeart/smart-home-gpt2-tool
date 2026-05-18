# 🏠 Smart-Home GPT-2

**Голосовое управление умным домом на GPT-2 124M. CPU. Offline. Без облака.**

## 🇷🇺 Описание

- **Содержание**
  - [Что это](#что-это)
  - [Архитектура](#архитектура)
  - [Бенчмарки](#бенчмарки)
  - [Голосовой пайплайн](#голосовой-пайплайн)
  - [Воспроизведение](#воспроизведение)
  - [Расширение под себя](#расширение-под-себя)
  - [Файлы](#файлы)
  - [Ограничения](#ограничения)
- **Дополнительные документы:**
  - [TUTORIAL.md](TUTORIAL.md) — 8 шагов от клонирования до своего микрофона
  - [INTEGRATION.md](INTEGRATION.md) — подключение к Home Assistant, Zigbee2MQTT, ESPHome, HomeKit, MQTT
  - [QUANTIZATION.md](QUANTIZATION.md) — как ужать веса до 125 МБ и ускорить в 3-5×
- 🇬🇧 [English description below ↓](#english)

---

### Что это

GPT-2 124M (OpenAI, 2019), дообученный на **1500 multi-tool примеров умного дома**. Понимает английские команды → выдаёт JSON tool call. Через локальную связку Whisper + опциональный переводчик принимает русскую речь.

**Пайплайн:**

```
Русская речь (микрофон / WAV)
  ↓ Silero TTS / faster-whisper STT (translate-mode)
Английский текст
  ↓ GPT-2 smart_home_v2 (124M, 475 MB)
JSON: {"name": "turn_on_light", "arguments": {"room": "kitchen"}}
  ↓ эмулятор / реальное устройство
Состояние обновлено
```

Всё локально. Никаких облачных API. CPU-only.

---

### Архитектура

| компонент | назначение | размер |
|---|---|---:|
| **silero TTS** (RU) | синтез русской речи для генерации тестов | ~50 МБ |
| **faster-whisper medium** | STT + перевод RU→EN за один шаг (`task="translate"`) | ~770 МБ |
| **GPT-2 + Full FT** (наше) | English → JSON tool call | 475 МБ |
| Fuzzy matcher | пост-обработка имени tool: ближайшее в registry | <1 КБ |

GPT-2 учили на 1500 SFT items от 10 параллельных агентов Claude Opus 4.7 в 10 доменах: lighting / climate / security / media / kitchen / garden / blinds / cleaning / timers / sensors. **100 уникальных function names**, 3-5 кандидатов в каждом промпте (модель выбирает один).

---

### Бенчмарки

**1. Multi-tool selection (300 held-out items из датасета):**

```
overall: 215/300 = 71.7%

by domain:
  garden     96.2%
  climate    86.2%
  misc       77.8%
  kitchen    73.9%
  security   71.0%
  blinds     69.2%
  media      68.8%
  cleaning   55.2%
  lighting   43.8%
```

**2. Voice end-to-end (30 русских голосовых команд):**

```
RU speech → Whisper translate → English → GPT-2 → tool call

Результат: 14/30 = 46.7% (с fuzzy match)
                7/30 = 23.3% (без fuzzy)

Разбивка по времени (на команду):
  TTS:   ~0.5 c
  STT:   ~5 c (Whisper medium на CPU)
  GPT-2: ~25 c
  всего: ~30 c
```

**Падение 71.7% → 46.7% при переходе на голос** обусловлено:
- Whisper иногда заменяет слова: "гостиной" → "hotel", "21 градус" → "a degree"
- GPT-2 124M путает похожие функции: turn_on vs turn_off, query_temperature vs set_thermostat

---

### Голосовой пайплайн

```python
from faster_whisper import WhisperModel
import torch, json, re

# 1. STT с переводом (Whisper делает RU→EN за один шаг)
stt = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = stt.transcribe("command.wav", language="ru", task="translate")
en_text = " ".join(s.text for s in segments).strip()
# e.g. "Turn on the light in the kitchen"

# 2. GPT-2 smart-home → JSON
# (см. src/voice_pipeline.py для полного кода)
```

Хочешь под другой язык? Whisper понимает 99 языков. Поменяй `language="ru"` на `"de"`, `"fr"`, `"es"`, etc. Перевод в English работает из коробки.

---

### Воспроизведение

```bash
git clone https://github.com/barometech/smart-home-gpt2
cd smart-home-gpt2
git lfs pull   # 475 МБ — модель

# зависимости
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper soundfile omegaconf transformers

# базовая GPT-2 (подтянется автоматически при первом запуске)

# тренировка с нуля (~2 ч CPU):
python src/train.py

# бенч на 300 held-out:
python src/bench.py

# голосовой end-to-end (30 русских команд):
python src/voice_pipeline.py
```

---

### Расширение под себя

**1. Подключение к реальному дому.** См. `INTEGRATION.md`. Готовые рецепты для:
- Home Assistant (REST + Conversation Agent)
- Zigbee2MQTT (MQTT publish)
- ESPHome (native API)
- Apple HomeKit (через HAP-python / homebridge)
- Tuya Cloud
- Generic MQTT

**2. Свои функции.** Открой `src/voice_pipeline.py` → `TOOL_REGISTRY`. 12 функций по умолчанию, всего в датасете 100 уникальных имён. Близкие к существующим (`set_light_color`, `play_radio_station`) — работают сразу. Совсем новые — нужен дообуч (см. Шаг 8 в `TUTORIAL.md`).

**3. Свой язык.** Whisper понимает 99 языков. В `src/voice_pipeline.py` найди `stt_translate(stt, TMP_WAV, source_lang="ru")` и поменяй `"ru"` на `"de"`/`"fr"`/`"es"`/etc. Перевод в English — встроенный.

**4. Уменьшить модель.** См. `QUANTIZATION.md`. С `torch.quantization` dynamic int8 модель ужимается до **~125 МБ** и ускоряется в **3-5×** на CPU. Один Python-скрипт, без переобучения.

**5. Raspberry Pi.** См. `INTEGRATION.md` раздел Pi. Pi 4 (4GB) с int8 квантованием — рабочая конфигурация. Wake-word через openwakeword + Whisper-small + квантованный GPT-2.

**6. Расширение датасета.** Если нужны новые функции — генерим ещё items через любой LLM (промпт-шаблон в `src/build_sft_v2.py`), кладём в `data/sh_<domain>.json`, прогоняем `build_sft_v2.py` + `train.py`. Цикл ~3 часа CPU.

---

### Файлы

```
src/
  train.py              — SFT GPT-2 на 1200 multi-tool items, ~2 ч CPU
  bench.py              — оценка на 300 held-out
  voice_pipeline.py     — end-to-end: TTS → STT/translate → GPT-2 → JSON
  build_sft_v2.py       — собирает 1500 sh_*.json в train/test split
  build_tool_registry.py — извлекает 100 уникальных tool specs из датасета
  gen_synthetic_demo.py  — генератор 500 template-based примеров (демо)

data/
  sh_lighting.json sh_climate.json sh_security.json sh_media.json
  sh_kitchen.json sh_garden.json sh_blinds.json sh_cleaning.json
  sh_timers.json sh_sensors.json     — по 150 items, всего 1500
  sh_train.json (1200) sh_test.json (300) — split для SFT
  tool_registry.json — все 100 уникальных функций с параметрами

weights/
  smart_home_v2.pt      — 475 МБ через Git LFS, full FT GPT-2 124M

results/
  bench_v2_results.json — сырой результат на 300 held-out
  voice_pipeline_results.json — сырой результат на 30 голосовых тестах
```

---

### Ограничения

- **GPT-2 124M потолок:** 72% на чистом English, 47% на голосе. Для production нужна модель побольше (Qwen 4B → 95%+).
- **Whisper-medium:** иногда заменяет слова на похожие. Whisper-large качественнее, но медленнее (~15 с на CPU).
- **Только смарт-дом домен:** модель забыла остальные tool-call задачи после SFT. Если нужен универсальный, бери `barometech/gpt2-tool-call` (общий).
- **Inference 25-35 с на команду CPU:** для real-time нужен GPU или модель поменьше.
- **Tool registry 100 функций:** для обширного дома хватит. Если функций больше — нужно расширять SFT.

---

<a id="english"></a>

## 🇬🇧 English

**Voice-controlled smart home on GPT-2 124M. CPU. Offline. No cloud.**

### What

GPT-2 124M (OpenAI, 2019), fine-tuned on **1500 multi-tool smart-home examples**. Takes English commands → emits JSON tool call. Accepts Russian (and 99 other languages) speech via local Whisper translate-mode.

**Pipeline:**

```
Russian/any-language speech (mic / WAV)
  ↓ Silero TTS (test only) / faster-whisper translate-mode
English text
  ↓ GPT-2 smart_home_v2 (124M, 475 MB)
JSON: {"name": "turn_on_light", "arguments": {"room": "kitchen"}}
  ↓ emulator / real device
State updated
```

All local. No cloud APIs. CPU-only.

### Architecture

| component | role | size |
|---|---|---:|
| silero TTS (RU) | speech synth for test generation | ~50 MB |
| faster-whisper medium | STT + translate in one pass | ~770 MB |
| GPT-2 + Full FT (ours) | English → JSON tool call | 475 MB |
| Fuzzy matcher | snap hallucinated names to nearest in registry | <1 KB |

Trained on 1500 SFT items from 10 parallel Claude Opus 4.7 agents covering 10 domains: lighting / climate / security / media / kitchen / garden / blinds / cleaning / timers / sensors. **100 unique function names**, 3-5 candidate functions per prompt.

### Benchmarks

**1. Multi-tool selection (300 held-out items):**

```
overall: 215/300 = 71.7%

by domain:
  garden     96.2%
  climate    86.2%
  misc       77.8%
  kitchen    73.9%
  security   71.0%
  blinds     69.2%
  media      68.8%
  cleaning   55.2%
  lighting   43.8%
```

**2. Voice end-to-end (30 Russian voice commands):**

```
RU speech → Whisper translate → English → GPT-2 → tool call

Result: 14/30 = 46.7% (with fuzzy match)
         7/30 = 23.3% (without fuzzy match)

Latency per command:
  TTS:   ~0.5 s
  STT:   ~5 s (Whisper-medium on CPU)
  GPT-2: ~25 s
  total: ~30 s
```

**Drop 71.7% → 46.7% on voice** is due to:
- Whisper occasionally replaces words ("гостиной" / living room → "hotel")
- GPT-2 124M confuses similar functions (turn_on vs turn_off, query_temperature vs set_thermostat)

### Reproduce

```bash
git clone https://github.com/barometech/smart-home-gpt2
cd smart-home-gpt2
git lfs pull

pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper soundfile omegaconf transformers

python src/train.py             # ~2h CPU
python src/bench.py             # multi-tool eval on 300 held-out
python src/voice_pipeline.py    # 30 Russian voice commands end-to-end
```

### Multi-language

Whisper handles 99 languages. Change `language="ru"` to `"de"` / `"fr"` / `"es"` / etc. in `src/voice_pipeline.py` — translation to English is built-in.

### Limitations

- **GPT-2 124M ceiling:** 72% on clean English, 47% via voice. For production use a 4B+ model.
- **Whisper-medium:** sometimes substitutes similar words. Whisper-large is better but ~3× slower on CPU.
- **Smart-home only:** model forgot general tool-calling after SFT. For universal tool calling use [`barometech/gpt2-tool-call`](https://github.com/barometech/gpt2-tool-call).
- **Inference 25-35 s/command on CPU:** for real-time use GPU or a smaller model.
- **100-function registry:** sufficient for most homes; extend SFT for more.

### License

MIT. Dataset of 1500 items is open-source — fork freely.

### Citation

```
@misc{popovich_smart_home_gpt2_2026,
  title  = {Smart-Home GPT-2: Voice-controlled local agent on 124M params},
  author = {Popovich, Pavel D.},
  year   = {2026},
  note   = {Also known as "Tekhnozhrets" (Техножнец). GitHub: barometech},
  url    = {https://github.com/barometech/smart-home-gpt2}
}
```
