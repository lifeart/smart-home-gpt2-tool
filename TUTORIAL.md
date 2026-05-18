# Туториал / Tutorial

## 🇷🇺 С нуля до работающего голосового агента (30 минут)

### Шаг 1. Подготовка окружения

```bash
# Linux/WSL2/macOS. На голой Windows проще через WSL2.
git clone https://github.com/barometech/smart-home-gpt2
cd smart-home-gpt2

# Git LFS должен быть установлен заранее (https://git-lfs.com)
git lfs install
git lfs pull            # подтянет 475 МБ smart_home_v2.pt

# Python 3.11+ в venv
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper soundfile omegaconf tokenizers safetensors huggingface-hub
```

### Шаг 2. Базовый GPT-2 (одноразово)

Первый запуск любого скрипта сам подтянет `openai-community/gpt2` (~520 МБ) в `weights/base_gpt2/`. Или вручную:

```bash
huggingface-cli download openai-community/gpt2 --local-dir weights/base_gpt2
```

### Шаг 3. Простой текстовый тест

```bash
python src/bench.py
```

Прогоняет 300 held-out команд из `data/sh_test.json` через `weights/smart_home_v2.pt` и печатает accuracy. На 4 ядрах CPU занимает ~2 часа. Ожидаемый результат: **~72% overall**.

Если хочется быстро глянуть что работает, отредактируй `bench.py` строку `items = json.load(...)` → `items = json.load(...)[:30]` — займёт 15 минут.

### Шаг 4. Голосовой пайплайн

```bash
python src/voice_pipeline.py
```

Скрипт:
1. Скачает silero TTS (~50 МБ) и faster-whisper medium (~770 МБ) при первом запуске.
2. Озвучит 30 русских команд через silero.
3. Распознает их через Whisper translate-mode (RU речь → EN текст).
4. Прогонит каждую через GPT-2 → JSON tool call.
5. Применит fuzzy match (если модель галлюцинирует имя — подтянет к ближайшему в registry).
6. Распечатает result + сохранит `results/voice_pipeline_results.json`.

Ожидаемый результат: **~47% accuracy**, ~30 сек/команда на CPU.

### Шаг 5. Свой микрофон

Замени блок `synth_ru` в `voice_pipeline.py` на запись с микрофона:

```python
import sounddevice as sd, soundfile as sf
def record_command(out_path, duration=5):
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(out_path), audio, 16000)

# вместо synth_ru(tts, ru_text, TMP_WAV):
record_command(TMP_WAV, duration=5)
```

Установи: `pip install sounddevice`.

### Шаг 6. Свой набор функций

Открой `src/voice_pipeline.py` → `TOOL_REGISTRY` (12 функций по умолчанию). Замени на нужные:

```python
TOOL_REGISTRY = {
    "my_turn_on_light": {"params": {"room": "string", "brightness": "integer"}},
    "my_send_telegram": {"params": {"chat_id": "string", "text": "string"}},
    # ...
}
```

Если функция близка к тем, что в датасете (например `set_light_color`), модель попадёт сразу. Если совсем новая — нужен дообуч (см. ниже).

### Шаг 7. Подключение к Home Assistant

См. отдельный документ `INTEGRATION.md` — там готовый код для Home Assistant REST API, Zigbee2MQTT, ESPHome, HomeKit, MQTT.

Минимальный пример (Home Assistant):

```python
import requests
HA_URL = "http://homeassistant.local:8123"
HA_TOKEN = "YOUR_LONG_LIVED_TOKEN"

def execute(call):
    if call["name"] == "turn_on_light":
        requests.post(
            f"{HA_URL}/api/services/light/turn_on",
            headers={"Authorization": f"Bearer {HA_TOKEN}"},
            json={"entity_id": f"light.{call['arguments']['room'].replace(' ','_')}"},
        )
```

### Шаг 8. Расширение датасета

Если нужны функции, которых нет в наших 100:

1. Сгенерируй ~200 примеров через любую LLM (см. промпт-шаблон в `src/build_sft_v2.py`).
2. Положи в `data/sh_my_domain.json` (формат `{"id", "function": [{spec},...], "prompt", "gold_name", "gold_args"}`).
3. Запусти `python src/build_sft_v2.py` (склеит с существующими в `sh_train.json`).
4. Запусти `python src/train.py` (2-3 часа CPU).

---

## 🇬🇧 Zero to working voice agent (30 min)

### Step 1. Environment

```bash
git clone https://github.com/barometech/smart-home-gpt2
cd smart-home-gpt2

git lfs install && git lfs pull   # pulls 475 MB smart_home_v2.pt

python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper soundfile omegaconf tokenizers safetensors huggingface-hub
```

### Step 2. Base GPT-2 (once)

First run of any script auto-downloads `openai-community/gpt2` to `weights/base_gpt2/`. Manual: `huggingface-cli download openai-community/gpt2 --local-dir weights/base_gpt2`.

### Step 3. Text bench

```bash
python src/bench.py        # 300 held-out items, ~2h CPU, target ~72%
```

For quick sanity, edit `items = ...[:30]` — 15 min.

### Step 4. Voice pipeline

```bash
python src/voice_pipeline.py
```

Downloads silero TTS + faster-whisper medium on first run. Runs 30 Russian commands end-to-end. Expected: ~47% accuracy, ~30s/command CPU.

### Step 5. Your microphone

Replace `synth_ru` block with `sounddevice` recording (snippet in Russian section above).

### Step 6. Your tools

Edit `TOOL_REGISTRY` in `voice_pipeline.py`. Names close to existing ones (e.g. `set_light_color`) work out of the box. Brand-new names need re-SFT.

### Step 7. Home Assistant

See `INTEGRATION.md` for HA, Zigbee2MQTT, ESPHome, HomeKit, MQTT recipes.

### Step 8. Extending the dataset

1. Generate ~200 items in same format (see `src/build_sft_v2.py` for schema).
2. Drop into `data/sh_<your_domain>.json`.
3. `python src/build_sft_v2.py && python src/train.py` (~2-3h CPU).

---

## Troubleshooting

| симптом / symptom | причина / cause | фикс / fix |
|---|---|---|
| `ModuleNotFoundError: faster_whisper` | `pip install faster-whisper` |
| Whisper-medium качает 770 МБ долго | первый запуск; кэшируется | подождать |
| `model.load_state_dict` mismatch | base_gpt2 неполный | `rm -rf weights/base_gpt2 && rerun` |
| `start_vacuum_cleaner` not in registry | модель галлюцинирует | fuzzy match уже включён, проверь `pred_name_raw` |
| inference > 60s/команда | слабый CPU или Whisper-large | переключись на `WhisperModel("small")` |
| Voice 0% accuracy | TOOL_REGISTRY пустой | заполни словарём |
