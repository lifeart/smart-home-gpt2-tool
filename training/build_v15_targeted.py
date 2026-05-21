# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Iter 40 — failure-driven targeted augmentation for v15 (accuracy idea B3).

The synthesis-pipeline error analysis showed a class of failures the GPT-2
candidates simply cannot do: inferring an enum value / mode from *implied*
language — "it's freezing in here" → heat, "kill the lights" → off,
"deploy the awning" → extend. v9's data has the explicit forms but thin
coverage of the implied ones.

This generates ~5k rich-schema rows (same format as sh_train_v9) that pair
implication-style user phrasings with the correct call. It is ADDITIVE —
mixed on top of the full v9 set, never replacing it (Iter 24's v6r-args
*replaced* SH data and lost cross-domain ability — the documented trap).

Output: data/sh_train_v15.json  =  sh_train_v9.json  +  ~5k targeted rows.
Deterministic (seed 42). Run: python training/build_v15_targeted.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
TARGET_N = 5000
VARIANTS = 3   # rows per (phrasing, room) — re-sampled candidate sets

ROOMS = [
    "living room", "bedroom", "kitchen", "bathroom", "office", "nursery",
    "master bedroom", "hallway", "dining room", "garage", "sunroom", "study",
]

DESC = {
    "set_ac_mode": "Set the air-conditioning mode for a room.",
    "set_thermostat": "Set the target temperature of a room's thermostat.",
    "set_fan_speed": "Set the fan speed in a room.",
    "dim_light": "Dim the lights in a room to a brightness percentage.",
    "turn_on_light": "Turn on the lights in a room.",
    "turn_off_light": "Turn off the lights in a room.",
    "extend_awning": "Extend (deploy) the awning of a room.",
    "retract_awning": "Retract (pull in) the awning of a room.",
    "open_curtains": "Open the curtains in a room.",
    "close_curtains": "Close the curtains in a room.",
}


def build_schema(fn: str, registry: dict) -> dict:
    spec = registry[fn]
    params = spec.get("params") or {}
    enums = spec.get("enums") or {}
    props = {}
    for p, t in params.items():
        d = {"type": t}
        if p in enums:
            d["enum"] = enums[p]
        props[p] = d
    return {
        "name": fn,
        "description": DESC.get(fn, f"{fn.replace('_', ' ')}."),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": spec.get("required") or [],
        },
    }


def make_prompt(cands: list, user: str) -> str:
    return (
        "SYSTEM: You are a helpful assistant with access to the following "
        "functions. Use them if required -\n"
        + json.dumps(cands, indent=2)
        + f"\n\n\nUSER: {user}\n\n\nASSISTANT: <functioncall> "
    )


# (gold_fn, distractor pool, [(phrasing template, gold-args builder)])
# Each phrasing template has {room}; the args builder returns the gold args.
def patterns():
    P = []

    # --- climate mode inference -------------------------------------------
    climate_distractors = ["set_ac_mode", "set_thermostat", "set_fan_speed",
                           "set_humidity_target", "toggle_dehumidifier"]
    mode_phrases = {
        "heat": ["it's freezing in the {room}", "the {room} is so cold",
                 "I'm shivering in the {room}", "warm up the {room}",
                 "the {room} is chilly", "it's frigid in the {room}"],
        "cool": ["it's boiling in the {room}", "the {room} is too hot",
                 "I'm sweating in the {room}", "cool down the {room}",
                 "the {room} is stuffy and warm", "it's sweltering in the {room}"],
        "eco": ["set the {room} AC to eco", "energy-saving mode for the {room}",
                "eco mode in the {room}"],
        "night": ["night mode for the {room}", "set the {room} for sleeping",
                  "the {room} AC to night mode"],
        "away": ["we're leaving — set the {room} to away",
                 "away mode for the {room} AC"],
        "off": ["turn off the AC in the {room}", "shut the {room} AC off",
                "kill the air conditioning in the {room}"],
    }
    for mode, phrases in mode_phrases.items():
        for ph in phrases:
            P.append(("set_ac_mode", climate_distractors, ph,
                      lambda room, m=mode: {"room": room, "mode": m}))

    # --- fan speed inference ---------------------------------------------
    fan_phrases = {
        "high": ["crank the {room} fan", "the {room} fan on full blast",
                 "max the fan in the {room}"],
        "low": ["the {room} fan on a gentle setting", "barely run the {room} fan",
                "{room} fan on low"],
        "medium": ["{room} fan on a normal setting", "moderate fan in the {room}"],
    }
    for sp, phrases in fan_phrases.items():
        for ph in phrases:
            P.append(("set_fan_speed", climate_distractors, ph,
                      lambda room, s=sp: {"room": room, "speed": s}))

    # --- light state inference -------------------------------------------
    light_distractors = ["turn_on_light", "turn_off_light", "dim_light",
                         "set_light_color", "set_light_scene"]
    on_phrases = ["power up the {room} lights", "I need light in the {room}",
                  "illuminate the {room}", "lights on in the {room}",
                  "brighten the {room} up", "switch the {room} lights on"]
    off_phrases = ["kill the {room} lights", "lights out in the {room}",
                   "douse the {room} lights", "the {room} is too bright — lights off",
                   "shut off the {room} lights", "no light in the {room} please"]
    for ph in on_phrases:
        P.append(("turn_on_light", light_distractors, ph,
                  lambda room: {"room": room}))
    for ph in off_phrases:
        P.append(("turn_off_light", light_distractors, ph,
                  lambda room: {"room": room}))
    # dim with explicit pct
    for pct in (10, 20, 25, 30, 40, 50, 60, 70):
        for tmpl in ["dim the {room} lights to {pct} percent",
                     "set the {room} lights to {pct}%",
                     "bring the {room} down to {pct} percent brightness"]:
            P.append(("dim_light", light_distractors,
                      tmpl.replace("{pct}", str(pct)),
                      lambda room, p=pct: {"room": room, "brightness_pct": p}))

    # --- awning / curtain polarity ---------------------------------------
    awn_distractors = ["extend_awning", "retract_awning", "open_curtains",
                       "close_curtains", "set_blinds_position", "open_window"]
    for ph in ["deploy the awning in the {room}", "extend the {room} awning",
               "roll out the awning by the {room}", "put the {room} awning out"]:
        P.append(("extend_awning", awn_distractors, ph,
                  lambda room: {"room": room}))
    for ph in ["retract the {room} awning", "pull in the awning in the {room}",
               "roll up the {room} awning", "bring the {room} awning back in"]:
        P.append(("retract_awning", awn_distractors, ph,
                  lambda room: {"room": room}))
    for ph in ["open the curtains in the {room}", "draw back the {room} curtains",
               "part the {room} curtains"]:
        P.append(("open_curtains", awn_distractors, ph,
                  lambda room: {"room": room}))
    for ph in ["close the curtains in the {room}", "draw the {room} curtains",
               "shut the {room} curtains"]:
        P.append(("close_curtains", awn_distractors, ph,
                  lambda room: {"room": room}))
    return P


def main() -> None:
    rng = random.Random(SEED)
    registry = json.loads((DATA / "tool_registry.json").read_text())
    pats = patterns()

    rows = []
    # expand: every (pattern, room) combination; per combo emit up to
    # VARIANTS rows with distinct re-sampled candidate sets (same query →
    # same gold, varied prompt context).
    combos = [(p, room) for p in pats for room in ROOMS]
    rng.shuffle(combos)
    for (gold_fn, dpool, phrase, argf), room in combos:
        if len(rows) >= TARGET_N:
            break
        pool = [d for d in dpool if d != gold_fn]
        user = phrase.format(room=room)
        gold = {"name": gold_fn, "arguments": argf(room)}
        gold_str = json.dumps(gold, separators=(",", ":"))
        seen = set()
        for _ in range(VARIANTS):
            rng.shuffle(pool)
            cand_fns = [gold_fn] + pool[:rng.choice([2, 3])]
            key = tuple(sorted(cand_fns))
            if key in seen:
                continue
            seen.add(key)
            order = list(cand_fns)
            rng.shuffle(order)
            cands = [build_schema(f, registry) for f in order]
            rows.append({
                "prompt": make_prompt(cands, user),
                "gold": gold_str,
                "gold_name": gold_fn,
                "domain": "v15_targeted",
                "task": "full",
            })

    from collections import Counter
    print(f"[gen] {len(rows)} targeted rows")
    print(f"[gen] by function: {dict(Counter(r['gold_name'] for r in rows))}")

    # Oversample the targeted rows 3x so the patterns get ~3 exposures in a
    # single continued-finetune epoch over v9's already-learned 78k.
    OVERSAMPLE = 3
    v9 = json.loads((DATA / "sh_train_v9.json").read_text())
    mix = v9 + rows * OVERSAMPLE
    rng.shuffle(mix)
    out = DATA / "sh_train_v15.json"
    out.write_text(json.dumps(mix))
    print(f"[mix] wrote {out.name}: {len(mix)} rows "
          f"({len(v9)} v9 + {len(rows)}x{OVERSAMPLE} targeted)")


if __name__ == "__main__":
    main()
