# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "torchaudio>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
#   "numpy>=1.26",
#   "omegaconf>=2.3",
#   "librosa>=0.10",
# ]
# ///
"""Voice E2E bench: RU speech -> noisy EN -> GPT-2 -> function name.

Two modes per model:
  cached: use `en_from_whisper` already in results/voice_pipeline_results.json
  live:   regenerate noisy EN via Silero TTS + faster-whisper translate

Candidate tool list is constructed deterministically as:
    gold function + 4 random near-domain neighbours from data/tool_registry.json
where "near-domain" is identified by prefix grouping (e.g., turn_on_light ->
the lighting bucket).

Compares pred_name to expected over 30 items. Reports per-model:
    acc_cached, acc_live, per-item (cached + live).

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        -e MODEL_REPOS="lifeart/smart-home-gpt2,lifeart/smart-home-gpt2-v3" \\
        training/bench_voice_hf.py
"""
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_REPOS = os.environ.get(
    "MODEL_REPOS",
    "lifeart/smart-home-gpt2,lifeart/smart-home-gpt2-v2,"
    "lifeart/smart-home-gpt2-v3",
).split(",")

ROOT = Path(__file__).resolve().parent.parent
LOCAL_VOICE = ROOT / "results" / "voice_pipeline_results.json"
LOCAL_TOOLS = ROOT / "data" / "tool_registry.json"

VOICE_URL = (
    "https://raw.githubusercontent.com/barometech/smart-home-gpt2/master/"
    "results/voice_pipeline_results.json"
)
TOOLS_URL = (
    "https://raw.githubusercontent.com/barometech/smart-home-gpt2/master/"
    "data/tool_registry.json"
)

RESULTS_REPO = os.environ.get("RESULTS_REPO", "lifeart/smart-home-sft-v2")

PROMPT_TMPL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n{tools}\n\n\n"
    "USER: {query}\n\n\nASSISTANT: <functioncall> "
)

NAME_RE = re.compile(r"""["'`]?name["'`]?\s*:\s*["']([^"'(\s,\}]+)""")

# Coarse domain buckets keyed by substring in the function name.
DOMAIN_KEYS = [
    ("light", "light"),
    ("lamp", "light"),
    ("blind", "blinds"),
    ("curtain", "blinds"),
    ("shade", "blinds"),
    ("shutter", "blinds"),
    ("thermostat", "climate"),
    ("ac_", "climate"),
    ("hvac", "climate"),
    ("fan", "climate"),
    ("humidi", "climate"),
    ("temperature", "climate"),
    ("radiator", "climate"),
    ("climate", "climate"),
    ("oven", "kitchen"),
    ("stove", "kitchen"),
    ("microwave", "kitchen"),
    ("coffee", "kitchen"),
    ("kettle", "kitchen"),
    ("fridge", "kitchen"),
    ("dishwasher", "kitchen"),
    ("scene", "media"),
    ("speaker", "media"),
    ("tv", "media"),
    ("volume", "media"),
    ("music", "media"),
    ("media", "media"),
    ("alarm", "sec"),
    ("camera", "sec"),
    ("lock", "sec"),
    ("doorbell", "sec"),
    ("siren", "sec"),
    ("security", "sec"),
    ("garage", "sec"),
    ("garden", "garden"),
    ("sprinkler", "garden"),
    ("irrigation", "garden"),
    ("plant", "garden"),
    ("clean", "clean"),
    ("vacuum", "clean"),
    ("mop", "clean"),
    ("timer", "misc"),
    ("reminder", "misc"),
    ("notification", "misc"),
    ("outlet", "misc"),
    ("plug", "misc"),
    ("switch", "misc"),
]


def parse_name(text: str) -> str | None:
    m = NAME_RE.search(text)
    return m.group(1) if m else None


def load_voice() -> list[dict]:
    if LOCAL_VOICE.exists():
        with LOCAL_VOICE.open() as f:
            data = json.load(f)
    else:
        try:
            p = hf_hub_download(
                repo_id="lifeart/smart-home-sft-v2",
                filename="voice_pipeline_results.json",
                repo_type="dataset",
            )
            with open(p) as f:
                data = json.load(f)
        except Exception:
            with urllib.request.urlopen(VOICE_URL) as r:
                data = json.loads(r.read().decode("utf-8"))
    return data.get("results", data)


def load_tools() -> dict:
    if LOCAL_TOOLS.exists():
        with LOCAL_TOOLS.open() as f:
            return json.load(f)
    try:
        p = hf_hub_download(
            repo_id="lifeart/smart-home-sft-v2",
            filename="tool_registry.json",
            repo_type="dataset",
        )
        with open(p) as f:
            return json.load(f)
    except Exception:
        with urllib.request.urlopen(TOOLS_URL) as r:
            return json.loads(r.read().decode("utf-8"))


def bucket_of(name: str) -> str:
    n = name.lower()
    for k, d in DOMAIN_KEYS:
        if k in n:
            return d
    return "misc"


def build_buckets(tool_names: list[str]) -> dict[str, list[str]]:
    bs: dict[str, list[str]] = {}
    for n in tool_names:
        b = bucket_of(n)
        bs.setdefault(b, []).append(n)
    return bs


def candidate_list(gold: str, buckets: dict[str, list[str]], rng: random.Random, all_names: list[str]) -> list[str]:
    bucket = bucket_of(gold)
    same = [n for n in buckets.get(bucket, []) if n != gold]
    rng.shuffle(same)
    picks = [gold] + same[:4]
    # Top up from global pool if bucket was small
    if len(picks) < 5:
        others = [n for n in all_names if n not in picks]
        rng.shuffle(others)
        picks += others[: 5 - len(picks)]
    rng.shuffle(picks)
    return picks


@torch.no_grad()
def generate_call(model, tok, prompt: str, device, max_new: int = 80) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    close_brace = tok.encode("}", add_special_tokens=False)[0]
    newline = tok.encode("\n", add_special_tokens=False)[0]
    for _ in range(max_new):
        if cur.shape[1] >= 1024:
            break
        out = model(cur)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[0, -1, :].argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == close_brace or nxt == newline:
            break
    new_ids = cur[0, L:].tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()


SILERO_SR = 24_000
WHISPER_SR = 16_000
WHISPER_REPO = "openai/whisper-medium"


def load_silero():
    print("[tts] loading Silero v4 RU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ret = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
        trust_repo=True,
    )
    model = ret[0] if isinstance(ret, (tuple, list)) else ret
    model.to(device)
    return model


def make_whisper():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stt] {WHISPER_REPO} on {device}")
    processor = WhisperProcessor.from_pretrained(WHISPER_REPO)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_REPO).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return processor, model, device


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


@torch.no_grad()
def tts_then_whisper(silero, whisper, text: str) -> str:
    audio = silero.apply_tts(
        text=text, speaker="aidar", sample_rate=SILERO_SR,
        put_accent=True, put_yo=True,
    ).cpu().numpy()
    processor, model, device = whisper
    a = _resample(audio, SILERO_SR, WHISPER_SR)
    inputs = processor(a, sampling_rate=WHISPER_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    if device.type == "cuda":
        feats = feats.half()
    out = model.generate(
        feats,
        language="russian",
        task="translate",
        max_new_tokens=128,
        num_beams=1,
        do_sample=False,
    )
    text_out = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text_out.strip()


def bench_model_mode(repo: str, voice: list[dict], buckets, all_names, device, mode: str,
                     silero=None, whisper=None) -> dict:
    print(f"\n[load] {repo} mode={mode}")
    tok = GPT2TokenizerFast.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()

    correct = 0
    rng = random.Random(42)
    rows = []
    t0 = time.time()
    for i, item in enumerate(voice):
        gold = item["expected"]
        if mode == "cached":
            noisy_en = item["en_from_whisper"]
        else:
            ru = item["ru_input"]
            try:
                noisy_en = tts_then_whisper(silero, whisper, ru)
            except Exception as e:
                print(f"  [{i}] live pipeline failed: {e}")
                noisy_en = item["en_from_whisper"]
        cands = candidate_list(gold, buckets, rng, all_names)
        tools_block = json.dumps(cands, indent=2)
        prompt = PROMPT_TMPL.format(tools=tools_block, query=noisy_en)
        out = generate_call(model, tok, prompt, device)
        pred = parse_name(out)
        ok = pred == gold
        if ok:
            correct += 1
        rows.append({
            "i": i,
            "ru_input": item["ru_input"],
            "noisy_en": noisy_en,
            "candidates": cands,
            "expected": gold,
            "pred": pred,
            "raw": out[:200],
            "ok": ok,
        })
        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{len(voice)}] acc={correct/(i+1)*100:.1f}% "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    acc = correct / len(voice)
    print(f"\n=== {repo} [{mode}] ===")
    print(f"  Accuracy: {correct}/{len(voice)} = {acc*100:.1f}%")
    return {
        "repo": repo,
        "mode": mode,
        "acc": acc,
        "n": len(voice),
        "correct": correct,
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    voice = load_voice()
    tools = load_tools()
    all_names = list(tools.keys())
    # Augment with names that appear in voice expected but not in registry
    # (e.g. set_alarm, set_timer, set_reminder live in sh_train, not registry).
    extra = ["set_alarm", "set_timer", "set_reminder"]
    for n in extra:
        if n not in all_names:
            all_names.append(n)
    buckets = build_buckets(all_names)
    print(f"[voice] {len(voice)} items  tools={len(all_names)}  buckets={len(buckets)}")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[device] {device}")

    run_live = os.environ.get("LIVE", "1") != "0"
    silero = whisper = None
    if run_live:
        silero = load_silero()
        whisper = make_whisper()

    all_results: list[dict] = []
    for repo in MODEL_REPOS:
        repo = repo.strip()
        if not repo:
            continue
        all_results.append(
            bench_model_mode(
                repo, voice, buckets, all_names, device, "cached"
            )
        )
        if run_live:
            all_results.append(
                bench_model_mode(
                    repo, voice, buckets, all_names, device, "live",
                    silero=silero, whisper=whisper,
                )
            )

    print("\n\n===== SUMMARY (voice) =====")
    for r in all_results:
        print(
            f"  {r['repo']:<40} {r['mode']:<8} "
            f"{r['acc']*100:5.1f}% ({r['correct']}/{r['n']})"
        )

    out_path = Path("bench_voice_results.json")
    out_path.write_text(json.dumps({"results": all_results}, indent=2, ensure_ascii=False))
    print(f"[save] wrote {out_path}")

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo="bench_voice_results.json",
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                commit_message="voice E2E bench v1/v2/v3 (cached + live)",
            )
            print(f"[push] -> {RESULTS_REPO}/bench_voice_results.json")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
