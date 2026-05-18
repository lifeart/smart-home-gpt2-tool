"""Generate synthetic SFT data via HF Inference (router endpoint, OpenAI-compatible).

Iter 13.1 — HF-Inference-generated SFT data to target Iter 7.1 weak spots:
- 3+ key items (misc/security/media): biggest failure class (15.2% args acc on 3+keys).
- Numeric value precision: temperatures, brightness, volume, duration, time.
- Missing-key recovery: user query mentions multiple slots.
- Twin-confusion in regressed domains (clean/media/sec).

Usage:
    python gen_synthetic_hf.py --target 300 --out data/sh_synthetic_pilot.json
    python gen_synthetic_hf.py --target 3000 --out data/sh_train_synthetic.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "web" / "public" / "eval" / "tool_registry.json"

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_FALLBACKS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

# Domain map of registry names (must cover all 123).
DOMAIN_MAP: Dict[str, List[str]] = {
    "light": [
        "set_light_temperature_k", "dim_light", "turn_on_light", "set_light_color",
        "blink_light", "set_light_scene", "turn_off_light", "toggle_outlet",
        "query_light_state", "set_motion_sensitivity",
    ],
    "climate": [
        "set_thermostat", "set_fan_speed", "query_humidity", "toggle_humidifier",
        "set_ac_mode", "set_radiator_valve", "schedule_climate_program",
        "toggle_dehumidifier", "set_humidity_target", "query_temperature",
    ],
    "sec": [
        "query_alarm_status", "start_camera_recording", "lock_door",
        "set_camera_motion_sensitivity", "unlock_door", "arm_alarm_system",
        "stop_camera_recording", "view_camera_stream", "set_alarm_pin",
        "trigger_panic_alarm", "query_door_status", "disarm_alarm_system",
    ],
    "media": [
        "play_music", "set_tv_input", "turn_off_tv", "stop_music", "turn_on_tv",
        "pause_music", "mute_audio", "switch_speaker_room", "skip_track",
        "play_podcast", "set_tv_volume", "play_radio_station", "set_tv_channel",
        "queue_song", "set_volume",
    ],
    "kit": [
        "set_fridge_temperature", "preheat_oven", "set_oven_timer",
        "pause_dishwasher", "stop_oven", "set_kitchen_lights", "start_microwave",
        "start_coffee_brew", "stop_microwave", "query_oven_state",
        "start_dishwasher", "query_fridge_contents", "set_coffee_strength",
    ],
    "garden": [
        "start_irrigation_zone", "stop_irrigation_zone", "turn_on_outdoor_light",
        "turn_off_outdoor_light", "set_pool_heater", "set_pool_pump",
        "turn_on_pool_cover", "turn_off_pool_cover", "query_soil_moisture",
        "query_pool_temperature", "set_outdoor_light_color", "set_garden_lawnmower",
        "set_outdoor_speaker", "schedule_irrigation",
    ],
    "blinds": [
        "open_skylight", "lock_window", "close_window", "retract_awning",
        "raise_blinds", "close_skylight", "close_curtains", "set_blinds_angle",
        "set_blinds_position", "open_window", "open_curtains", "extend_awning",
        "lower_blinds",
    ],
    "clean": [
        "start_vacuum", "stop_vacuum", "schedule_vacuum", "dock_vacuum",
        "start_mop", "stop_mop", "set_mop_water_level", "turn_on_air_purifier",
        "turn_off_air_purifier", "set_air_purifier_speed", "query_air_quality",
        "empty_vacuum_bin", "query_vacuum_battery",
    ],
    "misc": [
        "activate_scene", "save_current_scene", "schedule_routine", "cancel_alarm",
        "snooze_alarm", "query_alarms", "cancel_timer", "query_timers",
        "cancel_reminder", "query_motion_sensor", "query_smoke_alarm",
        "query_water_leak", "query_garage_door", "query_solar_production",
        "list_active_devices", "generate_status_report", "set_alarm", "set_timer",
        "set_reminder", "query_battery_level", "query_power_usage",
        "query_water_meter", "query_window_status",
    ],
}

NAME_TO_DOMAIN: Dict[str, str] = {
    n: d for d, names in DOMAIN_MAP.items() for n in names
}


def load_registry() -> Dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text())


def hf_chat(
    token: str,
    messages: List[Dict[str, str]],
    model: str,
    *,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    timeout: int = 90,
) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        print(f"  [http exc] {exc}", flush=True)
        return None
    if r.status_code != 200:
        print(f"  [http {r.status_code}] {r.text[:240]}", flush=True)
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        print(f"  [parse exc] {exc}; raw={r.text[:240]}", flush=True)
        return None


# ---------------- bucket sampling ----------------

BUCKETS = ["multi_key", "numeric", "twin", "generic"]


def _sample_gold_for_bucket(
    bucket: str, registry: Dict[str, Any]
) -> Tuple[str, str]:
    """Return (gold_name, domain) for a chosen bucket."""
    if bucket == "multi_key":
        # 3+ params, weight by domain (misc/sec/media)
        candidates = [
            n for n, info in registry.items()
            if len(info.get("params", {})) >= 3
        ]
        # Bias toward misc/sec/media via duplication
        weighted: List[str] = []
        for n in candidates:
            dom = NAME_TO_DOMAIN.get(n, "misc")
            weight = 3 if dom in ("misc", "sec", "media") else 1
            weighted.extend([n] * weight)
        name = random.choice(weighted)
    elif bucket == "numeric":
        candidates = [
            n for n, info in registry.items()
            if any(
                t in ("integer", "number", "float")
                for t in info.get("params", {}).values()
            )
        ]
        name = random.choice(candidates)
    elif bucket == "twin":
        # Twin pairs in clean/media/sec
        twin_groups = [
            ["start_vacuum", "stop_vacuum"],
            ["start_mop", "stop_mop"],
            ["dock_vacuum", "schedule_vacuum"],
            ["turn_on_air_purifier", "turn_off_air_purifier"],
            ["play_music", "pause_music", "stop_music"],
            ["turn_on_tv", "turn_off_tv"],
            ["lock_door", "unlock_door"],
            ["arm_alarm_system", "disarm_alarm_system"],
            ["start_camera_recording", "stop_camera_recording"],
            ["open_curtains", "close_curtains"],
            ["raise_blinds", "lower_blinds"],
            ["turn_on_light", "turn_off_light"],
        ]
        group = random.choice(twin_groups)
        name = random.choice(group)
    else:  # generic — any name uniformly across domains
        all_names = list(registry.keys())
        name = random.choice(all_names)
    return name, NAME_TO_DOMAIN.get(name, "misc")


def _twin_sibling(name: str) -> Optional[str]:
    """Return paired sibling for twin bucket so candidate list includes confusables."""
    pairs = {
        "start_vacuum": "stop_vacuum", "stop_vacuum": "start_vacuum",
        "start_mop": "stop_mop", "stop_mop": "start_mop",
        "dock_vacuum": "schedule_vacuum", "schedule_vacuum": "dock_vacuum",
        "turn_on_air_purifier": "turn_off_air_purifier",
        "turn_off_air_purifier": "turn_on_air_purifier",
        "play_music": "pause_music", "pause_music": "play_music",
        "stop_music": "play_music",
        "turn_on_tv": "turn_off_tv", "turn_off_tv": "turn_on_tv",
        "lock_door": "unlock_door", "unlock_door": "lock_door",
        "arm_alarm_system": "disarm_alarm_system",
        "disarm_alarm_system": "arm_alarm_system",
        "start_camera_recording": "stop_camera_recording",
        "stop_camera_recording": "start_camera_recording",
        "open_curtains": "close_curtains", "close_curtains": "open_curtains",
        "raise_blinds": "lower_blinds", "lower_blinds": "raise_blinds",
        "turn_on_light": "turn_off_light", "turn_off_light": "turn_on_light",
    }
    return pairs.get(name)


def _candidate_list(gold_name: str, registry: Dict[str, Any], bucket: str) -> List[str]:
    """Return 3-5 candidate function names. Must include gold_name."""
    cands = {gold_name}
    sibling = _twin_sibling(gold_name) if bucket == "twin" else None
    if sibling and sibling in registry:
        cands.add(sibling)
    domain = NAME_TO_DOMAIN.get(gold_name, "misc")
    domain_peers = [n for n in DOMAIN_MAP.get(domain, []) if n != gold_name]
    random.shuffle(domain_peers)
    target = random.choice([3, 4, 4, 5])
    for peer in domain_peers:
        if len(cands) >= target:
            break
        cands.add(peer)
    # Topup from registry if domain peers too few.
    if len(cands) < target:
        rest = [n for n in registry if n not in cands]
        random.shuffle(rest)
        for n in rest:
            if len(cands) >= target:
                break
            cands.add(n)
    out = list(cands)
    random.shuffle(out)
    return out


# ---------------- prompt builder ----------------


def _params_summary(name: str, registry: Dict[str, Any]) -> str:
    info = registry.get(name, {})
    params = info.get("params", {}) or {}
    required = set(info.get("required", []) or [])
    enums = info.get("enums", {}) or {}
    lines = []
    for key, ty in params.items():
        is_req = "REQUIRED" if key in required else "optional"
        en = enums.get(key)
        if en:
            sample = en[:6]
            lines.append(f"  - {key} ({ty}, {is_req}) enum sample: {sample}")
        else:
            lines.append(f"  - {key} ({ty}, {is_req})")
    return "\n".join(lines) if lines else "  (no parameters)"


def build_meta_prompt(
    bucket: str, gold_name: str, candidates: List[str], registry: Dict[str, Any]
) -> str:
    info = registry.get(gold_name, {})
    params_block = _params_summary(gold_name, registry)
    bucket_brief = {
        "multi_key": (
            "Generate a query whose answer requires AT LEAST 3 distinct argument keys. "
            "Mention all the slot values naturally in the query."
        ),
        "numeric": (
            "Generate a query that includes at least one numeric value. "
            "Phrase the number in one of these forms (pick one): digits ('22'), "
            "words ('twenty-two'), with unit ('22°C', '22 degrees'), "
            "or informal ('a bit warmer to 22'). Args must capture the numeric value."
        ),
        "twin": (
            "Generate a query that clearly disambiguates the polarity/state of the action "
            "vs its twin sibling (e.g. start vs stop, on vs off, lock vs unlock). "
            "Make the user's INTENT unambiguous."
        ),
        "generic": (
            "Generate a realistic natural user query for this function."
        ),
    }.get(bucket, "")

    return f"""You generate one synthetic smart-home SFT training example.

Target function: {gold_name}
Function parameters:
{params_block}

Required candidate function list (must include all, in this exact order):
{json.dumps(candidates)}

Task:
{bucket_brief}

Output STRICTLY a single-line JSON object with these keys (no markdown, no commentary):
- "user_query": natural English query, 4-25 words.
- "args": JSON object with arguments for {gold_name}. ONLY use the declared parameter keys. Required keys must be present. Values must be coherent with the user_query.

Examples of expected JSON shape:
{{"user_query":"Set the kitchen thermostat to 22 in cool mode.","args":{{"room":"kitchen","temperature_c":22,"mode":"cool"}}}}
{{"user_query":"Lock the front door.","args":{{"door":"front"}}}}

Return ONLY the JSON object. No prose. No fence."""


# ---------------- parse + validate ----------------


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Heuristic: find first balanced {...} substring and JSON-decode."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # remove first newline-prefixed language label
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rstrip("` \n")
    # find first '{' and balance
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def validate_item(
    parsed: Dict[str, Any],
    gold_name: str,
    registry: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (cleaned, reason). If valid, cleaned has {user_query, args}."""
    if not isinstance(parsed, dict):
        return None, "not_dict"
    q = parsed.get("user_query")
    args = parsed.get("args")
    if not isinstance(q, str):
        return None, "no_query"
    words = q.strip().split()
    if not (3 <= len(words) <= 30):
        return None, f"len_{len(words)}"
    # ASCII-only check (English-ish)
    try:
        q.encode("ascii")
    except UnicodeEncodeError:
        return None, "non_ascii"
    if not isinstance(args, dict):
        return None, "args_not_obj"
    info = registry.get(gold_name, {})
    allowed = set(info.get("params", {}).keys())
    required = set(info.get("required", []) or [])
    for k in args.keys():
        if k not in allowed:
            return None, f"unknown_key:{k}"
    if not required.issubset(args.keys()):
        return None, f"missing_required:{required - set(args.keys())}"
    return {"user_query": q.strip(), "args": args}, "ok"


def render_dataset_row(
    user_query: str,
    args: Dict[str, Any],
    gold_name: str,
    candidates: List[str],
    domain: str,
) -> Dict[str, Any]:
    tools_block = json.dumps(candidates, indent=2)
    prompt = (
        "SYSTEM: You are a helpful assistant with access to the following functions. "
        "Use them if required -\n"
        f"{tools_block}\n\n\n"
        f"USER: {user_query}\n\n\n"
        "ASSISTANT: <functioncall> "
    )
    gold = json.dumps({"name": gold_name, "arguments": args}, separators=(",", ":"))
    return {
        "prompt": prompt,
        "gold": gold,
        "gold_name": gold_name,
        "domain": domain,
    }


# ---------------- main loop ----------------


def pick_bucket(target_split: Dict[str, int], counts: Dict[str, int]) -> str:
    # Choose bucket whose remaining quota (target - counts) is largest.
    remaining = {b: target_split[b] - counts.get(b, 0) for b in BUCKETS}
    pool: List[str] = []
    for b, rem in remaining.items():
        if rem > 0:
            pool.extend([b] * rem)
    if not pool:
        return random.choice(BUCKETS)
    return random.choice(pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="explicit model id; default tries MODEL_FALLBACKS in order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-attempts-mult", type=float, default=2.5)
    args = ap.parse_args()

    random.seed(args.seed)
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path("~/.cache/huggingface/token").expanduser()
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("ERROR: HF_TOKEN not set and ~/.cache/huggingface/token missing")
        sys.exit(2)

    registry = load_registry()
    models = [args.model] if args.model else MODEL_FALLBACKS

    # Bucket targets (10/8/6/6 -> normalize)
    bucket_ratios = {"multi_key": 10, "numeric": 8, "twin": 6, "generic": 6}
    s = sum(bucket_ratios.values())
    target_split = {
        b: int(round(args.target * (r / s))) for b, r in bucket_ratios.items()
    }
    # Adjust to match target exactly
    delta = args.target - sum(target_split.values())
    target_split["generic"] += delta
    print(f"[plan] target split: {target_split}", flush=True)

    out_items: List[Dict[str, Any]] = []
    bucket_counts: Dict[str, int] = {b: 0 for b in BUCKETS}
    drop_reasons: Dict[str, int] = {}
    attempt = 0
    max_attempts = int(args.target * args.max_attempts_mult)

    chosen_model: Optional[str] = None
    last_heartbeat = time.time()

    while len(out_items) < args.target and attempt < max_attempts:
        attempt += 1
        bucket = pick_bucket(target_split, bucket_counts)
        gold_name, domain = _sample_gold_for_bucket(bucket, registry)
        cand = _candidate_list(gold_name, registry, bucket)
        meta = build_meta_prompt(bucket, gold_name, cand, registry)

        text = None
        for model in models:
            text = hf_chat(
                token,
                [
                    {"role": "system",
                     "content": "You are a precise JSON generator for smart-home SFT data. Reply with one JSON object only."},
                    {"role": "user", "content": meta},
                ],
                model=model,
                temperature=0.85,
                max_tokens=512,
            )
            if text is not None:
                chosen_model = model
                models = [model] + [m for m in models if m != model]
                break
        if text is None:
            drop_reasons["http_all_failed"] = drop_reasons.get("http_all_failed", 0) + 1
            continue

        parsed = _extract_json(text)
        if parsed is None:
            drop_reasons["unparseable"] = drop_reasons.get("unparseable", 0) + 1
        else:
            cleaned, reason = validate_item(parsed, gold_name, registry)
            if cleaned is None:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                row = render_dataset_row(
                    cleaned["user_query"],
                    cleaned["args"],
                    gold_name,
                    cand,
                    domain,
                )
                row["_bucket"] = bucket  # stripped at save time
                out_items.append(row)
                bucket_counts[bucket] += 1

        if time.time() - last_heartbeat > 30:
            print(
                f"[hb] attempt={attempt} kept={len(out_items)}/{args.target} "
                f"buckets={bucket_counts} drops={drop_reasons} model={chosen_model}",
                flush=True,
            )
            last_heartbeat = time.time()

        # Light pacing to avoid rate limits
        if attempt % 20 == 0:
            time.sleep(1.0)

    accept = len(out_items) / max(1, attempt)
    print(
        f"\n[done] kept={len(out_items)}/{args.target} attempts={attempt} "
        f"accept={accept:.1%}",
        flush=True,
    )
    print(f"[buckets] {bucket_counts}", flush=True)
    print(f"[drops] {drop_reasons}", flush=True)
    print(f"[model] {chosen_model}", flush=True)

    # Strip internal buckets before writing
    for row in out_items:
        row.pop("_bucket", None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_items, indent=2))
    # Also dump per-bucket counts side-file
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "model": chosen_model,
        "attempts": attempt,
        "kept": len(out_items),
        "accept_rate": accept,
        "bucket_counts": bucket_counts,
        "drop_reasons": drop_reasons,
        "target_split": target_split,
    }, indent=2))
    print(f"[wrote] {out_path}  meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
