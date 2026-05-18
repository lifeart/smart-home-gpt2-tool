"""End-to-end voice pipeline for smart home:
  Russian audio (wav) → Whisper STT → Russian text
  → Helsinki RU→EN translator → English text
  → GPT-2 smart_home_v2 → JSON tool call
  → emulator → state update

Reproducible test: generate 30 Russian commands via silero TTS, feed pipeline, count correct.
"""
import os, sys, json, re, time
from pathlib import Path
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrated_gpt2_torch import GPT2, encode, decode

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "smart_home_v2.pt"
TMP_WAV = Path("/tmp/sh_voice.wav")


# ---------- Smart home tool specs (subset for demo) ----------
TOOL_REGISTRY = {
    "turn_on_light": {"params": {"room": "string"}},
    "turn_off_light": {"params": {"room": "string"}},
    "set_thermostat": {"params": {"room": "string", "temperature_c": "integer"}},
    "lock_door": {"params": {"door": "string"}},
    "unlock_door": {"params": {"door": "string"}},
    "play_music": {"params": {"song": "string", "room": "string"}},
    "stop_music": {"params": {"room": "string"}},
    "open_curtains": {"params": {"room": "string"}},
    "close_curtains": {"params": {"room": "string"}},
    "start_vacuum": {"params": {"area": "string"}},
    "set_alarm": {"params": {"time": "string"}},
    "query_temperature": {"params": {"room": "string"}},
}


# ---------- 1. TTS (Russian) for test sample generation ----------
def get_tts():
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts",
                              language="ru", speaker="v4_ru", trust_repo=True)
    return model


def synth_ru(tts, text: str, out_path: Path):
    audio = tts.apply_tts(text=text, speaker="aidar", sample_rate=48000)
    sf.write(str(out_path), audio.numpy(), 48000)


# ---------- 2. STT (multilingual) ----------
def get_stt(size: str = "medium"):
    """size: tiny/base/small/medium/large-v3"""
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def stt_translate(stt, wav_path: Path, source_lang: str = None) -> str:
    """Whisper translate-mode: any-language audio → English text in one pass.
    Set source_lang=None to auto-detect, or specify e.g. 'ru', 'de', 'fr'.
    """
    segments, info = stt.transcribe(str(wav_path), language=source_lang, task="translate")
    return " ".join(s.text for s in segments).strip()


# ---------- 4. GPT-2 smart-home tool call ----------
def get_gpt2():
    m = GPT2()
    m.load_state_dict(torch.load(str(WEIGHTS), map_location='cpu'))
    m.to(DEVICE); m.eval()
    return m


def build_prompt(en_text: str, tool_names: list) -> str:
    """Build multi-tool prompt: all candidate tool specs in SYSTEM."""
    specs = []
    for name in tool_names:
        spec = TOOL_REGISTRY[name]
        specs.append({
            "name": name,
            "description": f"Smart home: {name.replace('_',' ')}",
            "parameters": {"type": "object",
                           "properties": {k: {"type": v} for k, v in spec["params"].items()},
                           "required": list(spec["params"].keys())},
        })
    return (f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
            f"{json.dumps(specs, indent=2)[:1200]}\n\n\n"
            f"USER: {en_text}\n\n\nASSISTANT: <functioncall> ")


@torch.no_grad()
def gpt2_generate(model, prompt: str, max_new=60):
    ids = encode(prompt)
    ids = ids[-900:] if len(ids) > 900 else list(ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= 1024: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits, _ = model(ii)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0]: break
    return decode(ids[L:]).strip()


def parse_call(text: str):
    name_m = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    name = name_m.group(1) if name_m else None
    args_m = re.search(r'["\'`]?arguments["\'`]?\s*:\s*(\{[^}]*\})', text)
    args = {}
    if args_m:
        try: args = json.loads(args_m.group(1))
        except: args = {}
    return name, args


def fuzzy_match(name: str, registry_names: list) -> str:
    """If exact name not in registry, find closest by SequenceMatcher."""
    if not name or name in registry_names:
        return name
    import difflib
    matches = difflib.get_close_matches(name, registry_names, n=1, cutoff=0.4)
    return matches[0] if matches else name


# ---------- 5. End-to-end pipeline ----------
def run_pipeline(ru_text, expected_tool, all_tools, tts, stt, model):
    t0 = time.time()
    # TTS (only for test sample generation)
    synth_ru(tts, ru_text, TMP_WAV)
    t_tts = time.time() - t0
    # STT with translate-mode: RU audio → EN text in one pass (Whisper handles it)
    t1 = time.time()
    en_text = stt_translate(stt, TMP_WAV, source_lang="ru")
    t_stt = time.time() - t1
    # GPT-2
    t3 = time.time()
    prompt = build_prompt(en_text, all_tools)
    raw = gpt2_generate(model, prompt)
    name_raw, args = parse_call(raw)
    name = fuzzy_match(name_raw, all_tools)
    t_gpt = time.time() - t3
    ok = (name == expected_tool)
    return {
        "ru_input": ru_text, "en_from_whisper": en_text,
        "gpt2_raw": raw[:120], "pred_name_raw": name_raw, "pred_name": name, "pred_args": args,
        "expected": expected_tool, "ok": ok,
        "time_tts": t_tts, "time_stt": t_stt, "time_gpt": t_gpt,
        "time_total": time.time() - t0,
    }


# ---------- 6. Test corpus (30 Russian commands) ----------
RUSSIAN_TESTS = [
    ("Включи свет в гостиной", "turn_on_light"),
    ("Выключи свет на кухне", "turn_off_light"),
    ("Установи температуру в спальне на 22 градуса", "set_thermostat"),
    ("Запри входную дверь", "lock_door"),
    ("Открой заднюю дверь", "unlock_door"),
    ("Поставь джазовый плейлист в гостиной", "play_music"),
    ("Останови музыку в кухне", "stop_music"),
    ("Открой шторы в спальне", "open_curtains"),
    ("Закрой шторы в офисе", "close_curtains"),
    ("Запусти робот-пылесос в гостиной", "start_vacuum"),
    ("Поставь будильник на семь утра", "set_alarm"),
    ("Какая температура в спальне", "query_temperature"),
    ("Включи лампу в детской", "turn_on_light"),
    ("Выключи освещение в гараже", "turn_off_light"),
    ("Сделай в комнате 21 градус", "set_thermostat"),
    ("Закрой гаражную дверь", "lock_door"),
    ("Разблокируй патио", "unlock_door"),
    ("Включи классическую музыку в спальне", "play_music"),
    ("Замолчи на кухне", "stop_music"),
    ("Подними жалюзи на кухне", "open_curtains"),
    ("Опусти жалюзи в гостиной", "close_curtains"),
    ("Запусти уборку на кухне", "start_vacuum"),
    ("Разбуди меня в шесть тридцать", "set_alarm"),
    ("Скажи температуру в ванной", "query_temperature"),
    ("Свет включи в коридоре", "turn_on_light"),
    ("Тепло сделай в детской двадцать три градуса", "set_thermostat"),
    ("Дверь входная заблокировать", "lock_door"),
    ("Музыку поставь рок в офисе", "play_music"),
    ("Шторы открой в кухне", "open_curtains"),
    ("Пылесос запусти в спальне", "start_vacuum"),
]


def main():
    print("[Voice pipeline test — Whisper translate-mode] loading models...")
    tts = get_tts()
    stt = get_stt()
    model = get_gpt2()
    all_tools = list(TOOL_REGISTRY.keys())
    print(f"  ready. {len(RUSSIAN_TESTS)} tests, {len(all_tools)} tools in spec\n")
    correct = 0; total = 0
    results = []
    t_all = time.time()
    for i, (ru, expected) in enumerate(RUSSIAN_TESTS):
        r = run_pipeline(ru, expected, all_tools, tts, stt, model)
        results.append(r)
        ok = "OK" if r["ok"] else "XX"
        if r["ok"]: correct += 1
        total += 1
        print(f"[{i+1:2d}/{len(RUSSIAN_TESTS)}] [{ok}] '{ru}'")
        print(f"       EN (Whisper translate): '{r['en_from_whisper']}'")
        raw_note = f"  (raw={r['pred_name_raw']})" if r['pred_name_raw'] != r['pred_name'] else ""
        print(f"       PRED: {r['pred_name']}{raw_note}  args={r['pred_args']}")
        print(f"       time: TTS={r['time_tts']:.1f}  STT={r['time_stt']:.1f}  GPT={r['time_gpt']:.1f}  TOTAL={r['time_total']:.1f}s\n", flush=True)
    print(f"=== Voice pipeline result: {correct}/{total} = {correct/total*100:.1f}% ({time.time()-t_all:.0f}s total) ===")
    Path(__file__).resolve().parent.joinpath("voice_pipeline_results.json").write_text(
        json.dumps({"acc": correct/total, "n": total, "correct": correct, "results": results}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
