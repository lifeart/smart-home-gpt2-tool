# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "torchaudio>=2.4",
#   "transformers>=4.45",
#   "sentencepiece>=0.2",
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
#   "numpy>=1.26",
#   "soundfile>=0.12",
#   "omegaconf>=2.3",
#   "librosa>=0.10",
# ]
# ///
"""Build Whisper-noise-augmented SH train set.

Pipeline per item:
  1. Take clean EN user query from data/sh_train.json (1200 items).
  2. Translate EN -> RU with Helsinki-NLP/opus-mt-en-ru (small + open).
  3. Synthesize RU TTS with Silero v4 (24 kHz, speaker `aidar`).
  4. Run faster-whisper medium with task='translate' to get noisy EN.
  5. Replace the USER line in the original prompt with the noisy EN.

Drops items where:
  - Whisper output is empty.
  - Noisy EN equals clean EN (no actual noise injected) AND ru !=  empty.

Final size targeted ~1000-1200 rows.

Output:
  - data/sh_train_noisy.json (local)
  - sh_train_noisy.json + sh_train_v3.json pushed to dataset repo
    `lifeart/smart-home-sft-v2`.

Composes `sh_train_v3.json` = sh_train_v2.json (8651) + noisy x 3 (~3.3k)
~ 11.6k rows.

Run on HF Jobs (t4-small recommended for GPU):
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 2h \\
        training/build_noisy_dataset.py
"""
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DATA_REPO = "lifeart/smart-home-sft-v2"
TRAIN_V1_NAME = "sh_train.json"          # original clean SH 1200
TRAIN_V2_NAME = "sh_train_v2.json"       # 8651 mixed
OUT_NOISY_NAME = "sh_train_noisy.json"
OUT_V3_NAME = "sh_train_v3.json"
LOCAL_DIR = Path("data")
LOCAL_DIR.mkdir(exist_ok=True, parents=True)

NLLB_REPO = "Helsinki-NLP/opus-mt-en-ru"  # ~300 MB, open, fine for short prompts
NOISY_REPEAT = 3                          # repeat noisy set 3x in v3

SR = 24_000
SILERO_SPEAKER = "aidar"


def extract_user_query(prompt: str) -> str | None:
    """Pull the last USER: ... segment before ASSISTANT:."""
    u = prompt.rfind("USER:")
    if u < 0:
        return None
    a = prompt.find("ASSISTANT:", u)
    if a < 0:
        return None
    return prompt[u + len("USER:"):a].strip()


def replace_user_query(prompt: str, new_query: str) -> str | None:
    u = prompt.rfind("USER:")
    if u < 0:
        return None
    a = prompt.find("ASSISTANT:", u)
    if a < 0:
        return None
    head = prompt[: u + len("USER:")]
    tail = prompt[a:]
    return f"{head} {new_query}\n\n\n{tail}"


def load_clean_sh() -> list[dict]:
    # Prefer dataset repo copy if running on HF Jobs (no local file copy).
    local = Path("data") / TRAIN_V1_NAME
    if local.exists():
        print(f"[data] using local {local}")
        with local.open() as f:
            return json.load(f)
    print(f"[data] fetching {DATA_REPO}/{TRAIN_V1_NAME}")
    p = hf_hub_download(
        repo_id=DATA_REPO, filename=TRAIN_V1_NAME, repo_type="dataset"
    )
    with open(p) as f:
        return json.load(f)


def load_v2() -> list[dict]:
    local = Path("data") / TRAIN_V2_NAME
    if local.exists():
        print(f"[data] using local {local}")
        with local.open() as f:
            return json.load(f)
    print(f"[data] fetching {DATA_REPO}/{TRAIN_V2_NAME}")
    p = hf_hub_download(
        repo_id=DATA_REPO, filename=TRAIN_V2_NAME, repo_type="dataset"
    )
    with open(p) as f:
        return json.load(f)


def translate_en_to_ru(queries: list[str], device) -> list[str]:
    """Batch translate EN -> RU."""
    print(f"[mt] loading {NLLB_REPO} on {device}")
    tok = MarianTokenizer.from_pretrained(NLLB_REPO)
    model = MarianMTModel.from_pretrained(NLLB_REPO).to(device).eval()
    out: list[str] = []
    bs = 16
    t0 = time.time()
    for i in range(0, len(queries), bs):
        batch = queries[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=128, num_beams=2, do_sample=False
            )
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(decoded)
        if (i // bs) % 5 == 0:
            print(
                f"  [mt] {i+len(batch)}/{len(queries)} "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )
    return out


def load_silero():
    """Load Silero v4 RU TTS model via torch hub."""
    print("[tts] loading Silero v4 RU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ret = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
        trust_repo=True,
    )
    # API returned either (model, example_text) (old) or just model (v4+)
    model = ret[0] if isinstance(ret, (tuple, list)) else ret
    model.to(device)
    return model, device


def synth_one(silero_model, device, text: str) -> np.ndarray | None:
    try:
        audio = silero_model.apply_tts(
            text=text,
            speaker=SILERO_SPEAKER,
            sample_rate=SR,
            put_accent=True,
            put_yo=True,
        )
        return audio.cpu().numpy()
    except Exception as e:
        # Silero rejects empty / too-short / non-Russian text occasionally
        print(f"[tts] fail: {e!r} text={text[:60]!r}")
        return None


WHISPER_REPO = "openai/whisper-medium"
WHISPER_SR = 16_000


def make_whisper():
    """Load transformers Whisper (works with torch's CUDA; no cuBLAS hassle)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stt] {WHISPER_REPO} on {device}")
    processor = WhisperProcessor.from_pretrained(WHISPER_REPO)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_REPO).to(device).eval()
    if device.type == "cuda":
        model = model.half()  # fp16 to fit / accelerate
    return processor, model, device


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


@torch.no_grad()
def transcribe_translate(whisper_bundle, audio: np.ndarray, sr: int) -> str:
    processor, model, device = whisper_bundle
    a = _resample(audio, sr, WHISPER_SR)
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
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()


def push(api: HfApi, local_path: Path, remote_name: str, msg: str) -> None:
    print(f"[push] {local_path} -> {DATA_REPO}/{remote_name}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_name,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=msg,
    )


def main() -> None:
    clean = load_clean_sh()
    print(f"[base] {len(clean)} SH items")

    # 1) Translate EN -> RU
    user_queries = []
    for r in clean:
        q = extract_user_query(r["prompt"])
        user_queries.append(q or "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rus = translate_en_to_ru(user_queries, device)
    assert len(rus) == len(clean)
    print(f"[mt] done, sample: {rus[0]!r}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) Silero TTS + 3) faster-whisper translate
    silero_model, silero_device = load_silero()
    whisper = make_whisper()

    noisy_rows: list[dict] = []
    drop_empty_ru = 0
    drop_tts_fail = 0
    drop_stt_empty = 0
    drop_identical = 0
    n_kept = 0
    t0 = time.time()

    for i, (row, ru) in enumerate(zip(clean, rus)):
        clean_user = extract_user_query(row["prompt"]) or ""
        ru_strip = ru.strip()
        if not ru_strip:
            drop_empty_ru += 1
            continue
        audio = synth_one(silero_model, silero_device, ru_strip)
        if audio is None or len(audio) < SR // 4:
            drop_tts_fail += 1
            continue
        noisy = transcribe_translate(whisper, audio, SR).strip()
        if not noisy:
            drop_stt_empty += 1
            continue
        # Compare normalized (lower, strip punctuation), keep even if identical
        # SOMETIMES (it's still a useful sample). But filter out cases where
        # noisy is virtually identical AND domain is `light` — these add no
        # robustness signal. Actually: keep all, since duplication is fine.
        # But if noisy is literally the RU text or empty, drop it.
        # Heuristic: if noisy is fully non-ASCII, Whisper failed to translate.
        if noisy and not any(c.isascii() and c.isalpha() for c in noisy):
            drop_stt_empty += 1
            continue
        # Build prompt
        new_prompt = replace_user_query(row["prompt"], noisy)
        if not new_prompt:
            drop_stt_empty += 1
            continue
        is_identical = noisy.strip().lower() == clean_user.strip().lower()
        if is_identical:
            drop_identical += 1
            # Still keep some identical rows? No — they add no new signal vs
            # the upsampled clean copies already in v2. Drop them.
            continue
        noisy_rows.append({
            "prompt": new_prompt,
            "gold": row["gold"],
            "gold_name": row["gold_name"],
            "domain": row["domain"] + "_noisy",
            "clean_user": clean_user,
            "ru_user": ru_strip,
            "noisy_user": noisy,
        })
        n_kept += 1
        if (i + 1) % 25 == 0:
            print(
                f"  [pipe] {i+1}/{len(clean)} kept={n_kept} "
                f"drop(empty_ru={drop_empty_ru} tts={drop_tts_fail} "
                f"stt={drop_stt_empty} id={drop_identical}) "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    print(
        f"\n[noisy] kept {n_kept} / {len(clean)}  "
        f"drops: empty_ru={drop_empty_ru} tts={drop_tts_fail} "
        f"stt={drop_stt_empty} identical={drop_identical}"
    )
    if n_kept < 800:
        print("[FAIL] kept too few noisy rows — aborting v3 compose")
        sys.exit(2)

    noisy_path = LOCAL_DIR / OUT_NOISY_NAME
    noisy_path.write_text(json.dumps(noisy_rows, ensure_ascii=False))
    print(f"[save] {noisy_path} ({noisy_path.stat().st_size/1e6:.1f} MB)")

    # Strip audit fields for training set (keep only model-input fields)
    noisy_for_train = [
        {
            "prompt": r["prompt"],
            "gold": r["gold"],
            "gold_name": r["gold_name"],
            "domain": r["domain"],
        }
        for r in noisy_rows
    ]

    # Compose v3 = v2 + noisy * NOISY_REPEAT
    v2 = load_v2()
    print(f"[v2] {len(v2)} items")
    merged = list(v2) + (noisy_for_train * NOISY_REPEAT)
    random.seed(42)
    random.shuffle(merged)
    print(f"[v3] {len(merged)} items (noisy={len(noisy_for_train)*NOISY_REPEAT})")

    v3_path = LOCAL_DIR / OUT_V3_NAME
    v3_path.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[save] {v3_path} ({v3_path.stat().st_size/1e6:.1f} MB)")

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        api = HfApi()
        push(api, noisy_path, OUT_NOISY_NAME, "noisy SH (TTS+Whisper) 1.2k -> ~1k kept")
        push(api, v3_path, OUT_V3_NAME, "v3 = v2 + noisy x3 (~11.6k)")


if __name__ == "__main__":
    main()
