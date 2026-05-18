# Smart-Home GPT-2 — Integration Guide

Practical paths to plug `smart_home_v2.pt` (GPT-2 124M FT, 475 MB, ~25 s/command on x86 CPU) into real smart-home stacks and run it on small hardware. No fluff, no marketing.

The model emits one JSON object per command, e.g.:

```json
{"name": "turn_on_light", "arguments": {"room": "kitchen", "brightness_pct": 80}}
```

Your job is to map the `name` → a platform call and pass `arguments` as the payload. Below: how to do that for six common stacks, then Pi feasibility, then one recommended stack.

---

## 1. Integration with existing smart-home platforms

### 1.1 Home Assistant

Three viable wiring patterns. Pick by how deep you want the integration.

**Pattern A — REST `services/<domain>/<service>` from a Python glue.** Simplest. The GPT-2 process stays separate from HA and just POSTs to HA's REST API.

```python
# glue.py — runs next to the model
import requests, json

HA_URL   = "http://homeassistant.local:8123"
HA_TOKEN = "eyJhbGciOiJI..."   # Long-Lived Access Token from HA profile

# Map our tool names to HA service calls
TOOL_TO_HA = {
    "turn_on_light":  ("light",  "turn_on"),
    "turn_off_light": ("light",  "turn_off"),
    "dim_light":      ("light",  "turn_on"),     # with brightness arg
    "set_thermostat": ("climate","set_temperature"),
    "set_fan_speed":  ("fan",    "set_percentage"),
}

def call_ha(tool_json: dict):
    name = tool_json["name"]
    args = tool_json["arguments"]
    domain, service = TOOL_TO_HA[name]

    # Build HA payload. "room" → entity_id via naming convention.
    entity_id = f"{domain}.{args['room'].replace(' ', '_')}"
    payload = {"entity_id": entity_id}

    if "brightness_pct" in args:
        payload["brightness_pct"] = args["brightness_pct"]
    if "temperature_c" in args:
        payload["temperature"] = args["temperature_c"]

    r = requests.post(
        f"{HA_URL}/api/services/{domain}/{service}",
        headers={"Authorization": f"Bearer {HA_TOKEN}"},
        json=payload, timeout=5,
    )
    r.raise_for_status()
    return r.json()
```

Caveats:
- Entity IDs in HA do not always follow `domain.room`. You must either rename HA entities to match the model's `room` slot, or maintain an explicit lookup table.
- 100-function registry overshoots HA's flat service model. Many of our tools (`blink_light`, `set_light_scene`) need to be lowered to a sequence of `light.turn_on` calls.

**Pattern B — Custom Conversation Agent.** Subclass `homeassistant.components.conversation.AbstractConversationAgent.async_process()`, call the model, synthesise an `IntentResponse`. Installed via HACS or `custom_components/gpt2_smart_home/__init__.py`. You get HA's UI, voice button and history for free.

**Pattern C — `voice_assistant` pipeline.** HA already chains wake-word → STT → conversation agent → TTS. Only replace the conversation agent (Pattern B); the rest of the pipeline is HA's. Best if HA is already running.

### 1.2 Zigbee2MQTT

Direct, no HA needed. Z2M exposes every device as `zigbee2mqtt/<friendly_name>` and accepts JSON on `zigbee2mqtt/<friendly_name>/set`.

```python
# z2m_glue.py
import paho.mqtt.publish as publish, json

BROKER = "192.168.1.10"

TOOL_TO_Z2M = {
    "turn_on_light":  lambda a: ("set", {"state":"ON",  "brightness": int(a.get("brightness_pct",100)*2.55)}),
    "turn_off_light": lambda a: ("set", {"state":"OFF"}),
    "set_light_color":lambda a: ("set", {"color":{"hex": _to_hex(a["color"])}}),
    "toggle_outlet":  lambda a: ("set", {"state": a["state"].upper()}),
}

def call_z2m(tool_json):
    name = tool_json["name"]; args = tool_json["arguments"]
    suffix, payload = TOOL_TO_Z2M[name](args)
    topic = f"zigbee2mqtt/{args['room']}_light/{suffix}"   # friendly_name convention
    publish.single(topic, json.dumps(payload), hostname=BROKER)
```

Caveats:
- `friendly_name` is set in Z2M's `configuration.yaml`. Adopt convention `<room>_<device>` (e.g. `kitchen_light`) and the mapping is trivial.
- Brightness in Z2M is 0–254, our tool uses 0–100 → multiply by 2.55.
- Color: our `color` is a string (`"red"`, `"warm"`). Z2M wants `{"r":..,"g":..,"b":..}` or `hex`. Maintain a 20-entry lookup table.
- No ACK by default. Subscribe to `zigbee2mqtt/+/+` if you want confirmation.

### 1.3 ESPHome

ESPHome devices expose either the native API (port 6053, used by HA) or `web_server:` (port 80, REST + WebSocket). For a non-HA setup, use `web_server`.

```yaml
# kitchen_light.yaml on the ESP32
web_server:
  port: 80
light:
  - platform: monochromatic
    name: "kitchen light"
    id: kitchen_light
    output: pwm_out
```

```python
# esphome_glue.py
import requests
DEVICES = {"kitchen": "http://10.0.0.51"}

def call_esphome(tool_json):
    a = tool_json["arguments"]
    base = DEVICES[a["room"]]
    if tool_json["name"] == "turn_on_light":
        params = {}
        if "brightness_pct" in a:
            params["brightness"] = int(a["brightness_pct"]*2.55)
        requests.post(f"{base}/light/kitchen_light/turn_on", params=params, timeout=3)
    elif tool_json["name"] == "turn_off_light":
        requests.post(f"{base}/light/kitchen_light/turn_off", timeout=3)
```

Caveats:
- `web_server` is unauthenticated by default — only run on trusted LAN, or set `web_server: auth:`.
- Native API (`aioesphomeapi`) is faster and supports push events but is overkill if you only flip 5 devices.
- One YAML per device; you need a flat `room → IP` map.

### 1.4 Apple HomeKit (HAP-python / Homebridge)

Impersonate HomeKit accessories so iOS/Siri/HomeKit Hubs see them as native. Two routes:

- **HAP-python**: write the bridge in Python. Each tool becomes an `Accessory` subclass with `set_*` characteristics. When GPT-2 emits JSON you call `acc.on.set_value(True)` and HomeKit broadcasts.
- **Homebridge with `homebridge-http-switch`** or `homebridge-mqtt`: each room is an HTTP switch whose ON URL hits your glue. Lighter.

```python
# hap_glue.py (HAP-python skeleton)
from pyhap.accessory import Accessory
from pyhap.accessory_driver import AccessoryDriver
from pyhap.const import CATEGORY_LIGHTBULB

class KitchenLight(Accessory):
    category = CATEGORY_LIGHTBULB
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        s = self.add_preload_service('Lightbulb', chars=['On','Brightness'])
        self.on = s.configure_char('On', setter_callback=lambda v: print("on",v))

driver = AccessoryDriver(port=51826)
driver.add_accessory(accessory=KitchenLight(driver, 'Kitchen'))
driver.start()
```

Caveats: persistent pairing state file; ≤50 accessories per bridge recommended; first pairing is interactive from an iPhone.

### 1.5 Tuya Cloud API

Tuya is cloud-only for the official API (`https://openapi.tuyaeu.com`). Local-only is possible via `tinytuya` if you have the device's `local_key`.

```python
# tuya_glue.py — local, tinytuya
import tinytuya
DEVICES = {
    "kitchen": tinytuya.OutletDevice('bf...', '10.0.0.80', 'localkey...'),
}
DEVICES["kitchen"].set_version(3.4)

def call_tuya(tool_json):
    d = DEVICES[tool_json["arguments"]["room"]]
    if tool_json["name"] == "turn_on_light":  d.turn_on()
    if tool_json["name"] == "turn_off_light": d.turn_off()
```

Caveats:
- `local_key` extraction requires a Tuya IoT developer account once, then survives offline.
- Tuya devices over cloud add ~500 ms latency on top of our 25 s — irrelevant.
- Many cheap Tuya bulbs do not expose color/brightness via local protocol (only on/off).

### 1.6 Generic MQTT broker

The lowest-friction path. Publish `home/<room>/<device>/set` with your own JSON schema; let any subscriber decide what to do.

```python
import paho.mqtt.publish as publish, json
def call_mqtt(tool_json):
    a = tool_json["arguments"]
    topic = f"home/{a['room']}/light/set"
    publish.single(topic, json.dumps({
        "action": tool_json["name"],
        "args": a,
    }), hostname="localhost")
```

This is what you actually want if you are building from scratch: one Mosquitto broker, every device subscribes to its own topic, your glue translates GPT-2 JSON 1:1.

---

## 2. Raspberry Pi feasibility

End-to-end stack memory budget (idle, after weights are loaded):

| Component | RAM |
|---|---:|
| Python + torch 2.4 CPU | ~250 MB |
| GPT-2 124M fp32 | 475 MB |
| GPT-2 124M int8 | ~140 MB |
| faster-whisper medium int8 | ~770 MB on disk, ~500 MB resident |
| faster-whisper small  int8 | ~245 MB on disk, ~200 MB resident |
| openwakeword         | ~50 MB |
| ALSA + pipewire/pulse | ~80 MB |

CPU-bound latency on Pi is dominated by GPT-2 generation (~120 forward passes for a typical JSON output of 30 tokens × ~4 ops/token at batch=1).

| Board | RAM headroom | GPT-2 int8 latency | End-to-end | Verdict |
|---|---|---|---|---|
| Pi 4 4 GB | ~2.5 GB free | ~45–55 s | ~50–60 s | barely usable |
| Pi 4 8 GB | ~6 GB free | ~45–55 s | ~50–60 s | extra RAM unused unless +HA Core |
| Pi 5 8 GB | ~7 GB free | ~17–22 s | ~20–25 s | **usable** |
| Pi 5 16 GB | ~15 GB free | ~17–22 s | ~20–25 s | RAM wasted unless +HA |
| Pi Zero 2 W | 512 MB total | 3–5 min | n/a | **not feasible** for full stack |

Notes:
- Pi 4: Cortex-A72 @ 1.8 GHz × 4. `bitsandbytes` not available on ARM — use `torch.ao.quantization.quantize_dynamic`. Whisper-small int8 ~3–4 s per 5 s clip.
- Pi 5: Cortex-A76 @ 2.4 GHz × 4, ~2.5× single-thread of Pi 4 (public llama.cpp benches). Whisper-small int8 ~1.5 s.
- int4 (GGUF Q4_0 via `llama.cpp` `convert-hf-to-gguf.py`) saves another 1.4× but costs 2–4 pp accuracy; only justified on Pi 4.
- Pi Zero 2 W: use as **edge mic + wake-word only**, streams 16 kHz PCM to a Pi 5 / x86. openwakeword model is 1.5 MB and runs <100 ms per 0.1 s frame.

### Quantisation, Whisper, wake-word, audio

- **Quantisation**: int8 dynamic (PyTorch built-in) is mandatory on Pi. GPT-2 475 MB → ~140 MB, ~1.6× speedup on A72/A76, <1 pp accuracy loss. int4 (GGUF Q4_0) gives another 1.4× at the cost of 2–4 pp. faster-whisper `compute_type="int8"` is non-negotiable on ARM (float runs 4–6× slower).
- **Whisper-small (244 MB) vs medium (770 MB)**: small WER on RU Common Voice ~14%, medium ~9%. The 5-pp WER gap costs ~8–10 pp end-to-end. Use medium on Pi 5 8 GB, small on Pi 4.
- **Wake-word**: **openwakeword** (Apache-2.0, custom keywords trainable in ~1 h, 1.5–3 MB) recommended; **Porcupine** for closed-source low-CPU; **snowboy** deprecated since 2020 — do not use.
- **Audio**: PipeWire (Bookworm default) for Pi 5. PulseAudio still works (~30 ms latency). ALSA only for dedicated single-process boxes. USB conference mic (Anker PowerConf S330, ~$50) drops Whisper WER ~4 pp vs HAT/3.5 mm mics.

---

## 3. Recommended stack for a hobbyist, today

One box, one weekend, usable result:

**Hardware**
- Raspberry Pi 5, 8 GB — $80
- Active cooler (mandatory, GPT-2 inference pegs all 4 cores for 20 s; without cooling you throttle to ~60% within 90 s) — $5
- 64 GB A2 microSD or NVMe HAT + 256 GB NVMe (NVMe halves model load time) — $15–40
- Anker PowerConf S330 USB mic — $50
- Any 3 W USB speaker — $10
- Total: **~$170**

**Software**
- Raspberry Pi OS Bookworm 64-bit (PipeWire default)
- Python 3.11
- `torch==2.4.0` CPU wheel for aarch64
- `faster-whisper==1.0.3` with `model="small"`, `compute_type="int8"`, `language="ru"`, `task="translate"`
- GPT-2: `smart_home_v2.pt` quantised to int8 dynamic at load:
  ```python
  model = torch.load("smart_home_v2.pt", map_location="cpu")
  model = torch.ao.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8)
  ```
- Wake-word: `openwakeword` with custom keyword (e.g. "Hey Tekhno") — train once on 200 of your own utterances
- Mosquitto broker (`apt install mosquitto`) on the same Pi
- Glue: 200-line Python script translating GPT-2 JSON → MQTT publishes
- Devices: ESPHome flashed ESP32 boards (one per room) subscribed to `home/<room>/+/set`

**Expected latency (Pi 5, 8 GB, int8 everywhere)**

| Stage | Time |
|---|---:|
| Wake-word detection | <0.2 s |
| 4 s audio capture | 4.0 s (real-time) |
| Whisper-small int8 translate | ~1.5 s |
| GPT-2 int8 generate ~30 tokens | ~18 s |
| MQTT publish + ESPHome action | ~0.1 s |
| **Total mic-on → light-on** | **~24 s** |

**Memory footprint at steady state**
- GPT-2 int8: 140 MB
- Whisper-small int8: 200 MB
- openwakeword + python + torch: ~300 MB
- Mosquitto: ~10 MB
- Pi OS desktop-less: ~250 MB
- **Total: ~900 MB on a 8 GB Pi** — 7 GB free for HA Core if you add it later.

**Honest expected accuracy**
- README baseline: 47% RU voice end-to-end with Whisper-medium. With Whisper-small you lose another ~8 pp → **~38–42% on Pi 5 voice**. Fuzzy match on tool name is mandatory.
- For better than 50%, drop GPT-2 and run a 1–4B model in `llama.cpp` (Qwen2.5-3B-Instruct Q4_0 fits in 2.5 GB, gives ~90%+ on this task per public BFCL-style evals, but bumps latency to ~40–60 s on Pi 5 — same trade-off the README mentions).

**What this stack will not do**
- Real-time conversation. 24 s/command — fine for lights/blinds/thermostat, painful for queries.
- Multi-turn context. GPT-2 was SFT'd one-shot → one-shot. No history.
- Reliable Russian without fuzzy match. Always keep the 100-name matcher in the loop.

**When to graduate**
- More accuracy: swap GPT-2 → Qwen2.5-3B Q4_0 on the same Pi 5. Same wiring, ~2× the latency, ~2× the accuracy.
- More speed: move inference to an x86 mini-PC (N100, ~$150). Same Pi 5 stays as the mic/wake endpoint over MQTT. End-to-end drops to ~6 s.
- Production: HA + Whisper-large + Qwen-7B on a desktop with a 12 GB GPU. Out of hobbyist scope.

---

## Appendix: minimal glue

```python
# main.py — wake → record → STT → GPT-2 → MQTT
import json, torch, sounddevice as sd, soundfile as sf
from faster_whisper import WhisperModel
import paho.mqtt.publish as publish
from openwakeword.model import Model as WW

ww  = WW(wakeword_models=["hey_tekhno.onnx"])
stt = WhisperModel("small", device="cpu", compute_type="int8")
gpt = torch.load("smart_home_v2.pt", map_location="cpu")
gpt = torch.ao.quantization.quantize_dynamic(gpt, {torch.nn.Linear}, dtype=torch.qint8)
gpt.eval()

while True:
    # wait for wake-word
    while True:
        a = sd.rec(int(0.1*16000), 16000, 1, dtype='int16'); sd.wait()
        if ww.predict(a.flatten())["hey_tekhno"] > 0.5: break
    # record 4s, transcribe RU→EN, run GPT-2, publish
    a = sd.rec(int(4*16000), 16000, 1, dtype='int16'); sd.wait()
    sf.write("/tmp/cmd.wav", a, 16000)
    segs,_ = stt.transcribe("/tmp/cmd.wav", language="ru", task="translate")
    en = " ".join(s.text for s in segs).strip()
    tj = gpt_to_json(en)   # see src/voice_pipeline.py
    publish.single(f"home/{tj['arguments']['room']}/cmd",
                   json.dumps(tj), hostname="localhost")
```

Replace the `publish.single(...)` line with the glue function for your platform — that is the only platform-specific code in the whole system.
