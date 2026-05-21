"""Iter 29 — value canonicalization post-processor.

Hard-floor analysis (PLAN.md Iter 29) found a large share of the
oracle-miss items are value FORMAT mismatches, not semantic errors:
  - time:  "3 PM" / "3pm" / "5:30 AM" / "8am"   vs gold "15:00" / "05:30" / "08:00"
  - float: 24.444444  (unrounded F→C)            vs gold 24.4

The models often extract the right value in the wrong surface form. This
module canonicalizes predicted argument values toward the dataset's gold
conventions, BEFORE scoring. It is a legitimate inference-time component:
the browser pipeline would apply the same normalization to its model
output. The scorer itself is unchanged (still mirrors web/bench.js).

`canonicalize_args(d)` returns a new dict with values rewritten.
"""
from __future__ import annotations

import re
from typing import Any


# 12-hour clock → 24-hour "HH:MM"
_TIME_12H = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b",
    re.IGNORECASE,
)
# bare "HH:MM" already 24h-ish
_TIME_24H = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _to_24h(h: int, m: int, ampm: str) -> str:
    ampm = ampm.lower()
    if ampm == "a":
        if h == 12:
            h = 0
    else:  # pm
        if h != 12:
            h += 12
    return f"{h:02d}:{m:02d}"


def canonicalize_time_string(s: str) -> str:
    """Rewrite any 12h-clock substrings inside `s` to 24h HH:MM.

    Handles standalone ("3 PM" → "15:00") and embedded
    ("Saturday 8am" → "Saturday 08:00"). Leaves already-24h "HH:MM"
    zero-padded.
    """
    def repl(m: re.Match) -> str:
        h = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) else 0
        return _to_24h(h, mm, m.group(3))

    out = _TIME_12H.sub(repl, s)

    # Zero-pad bare "H:MM" → "HH:MM"
    def pad(m: re.Match) -> str:
        h = int(m.group(1))
        mm = int(m.group(2))
        return f"{h:02d}:{mm:02d}"

    out = _TIME_24H.sub(pad, out)
    return out


def _looks_like_time_key(key: str) -> bool:
    k = key.lower()
    return any(t in k for t in ("time", "alarm", "_at", "when", "schedule"))


_DAY_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def _canon_day_token(tok: str) -> str:
    """'Saturdays' -> 'Saturday'. Leaves 'weekdays'/'weekends'/'daily' alone."""
    low = tok.lower()
    if low.endswith("s") and low[:-1] in _DAY_NAMES:
        # Preserve original capitalization of the singular stem.
        return tok[:-1]
    return tok


def _looks_like_day_key(key: str) -> bool:
    return key.lower() in ("day", "days", "weekday", "dayofweek", "day_of_week")


def canonicalize_value(key: str, v: Any) -> Any:
    """Canonicalize one argument value."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        # Round to 1 decimal — matches the dataset's temperature convention
        # (24.4, 21.5, ...). Integers-as-float collapse to int.
        r = round(v, 1)
        return int(r) if r == int(r) else r
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        t = v.strip()
        # Numeric string → number (the scorer also coerces, but doing it
        # here lets us round unrounded conversions like "24.4444").
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        if re.fullmatch(r"-?\d+\.\d+", t):
            r = round(float(t), 1)
            return int(r) if r == int(r) else r
        # Time normalization ONLY on time-like keys — never mangle a song
        # title or message that happens to contain "3 PM".
        if _looks_like_time_key(key) and (_TIME_12H.search(t) or _TIME_24H.search(t)):
            return canonicalize_time_string(t)
        # Day plural → singular on day-like keys ("Saturdays" → "Saturday").
        if _looks_like_day_key(key):
            return _canon_day_token(t)
        return t
    if isinstance(v, list):
        return [canonicalize_value(key, x) for x in v]
    if isinstance(v, dict):
        return canonicalize_args(v)
    return v


def canonicalize_args(args: Any) -> dict:
    """Return a new dict with every value canonicalized."""
    if not isinstance(args, dict):
        return {}
    return {k: canonicalize_value(k, v) for k, v in args.items()}


# ---------------------------------------------------------------------------
# Iter 37 — enum value snapping.
#
# The accuracy report (Iter 37) found the synthesizer's top remaining failure
# mode is argument *values* that are semantically right but in a surface form
# the dataset's enum doesn't use: "gym" vs "basement gym", "living_room" vs
# "living room". `tool_registry.json` carries the legal enum for each typed
# argument; snapping a predicted value to its enum member fixes these without
# touching the scorer. Conservative — three levels, no fuzzy/edit-distance
# matching, and only snaps when the target is unambiguous.
# ---------------------------------------------------------------------------

def _loose(s: str) -> str:
    """case- + underscore/space-insensitive key for enum comparison."""
    return re.sub(r"[\s_]+", " ", s.strip().lower())


def snap_enum_value(v: Any, enum: list) -> Any:
    """Snap one value to its closest enum member, or return it unchanged.

    Level 1+2: exact match ignoring case and underscore/space — also picks
      the enum's *canonical* spelling (e.g. 'living_room' -> 'living room'
      when both forms are listed; the first listed wins, deterministically).
    Level 3: unique substring containment ('gym' -> 'basement gym').
    No fuzzy/edit-distance matching — too risky to over-correct.
    """
    if not isinstance(v, str) or not enum:
        return v
    lv = _loose(v)
    for e in enum:                       # level 1+2
        if isinstance(e, str) and _loose(e) == lv:
            return e
    subs = [
        e for e in enum
        if isinstance(e, str) and (lv in _loose(e) or _loose(e) in lv)
    ]
    if len(subs) == 1:                   # level 3 — only if unambiguous
        return subs[0]
    return v


def snap_enums(name: Any, args: Any, registry: dict) -> dict:
    """Snap every enum-typed argument value of call `name` to its registry
    enum. Non-enum keys, unknown functions, and non-string values pass
    through untouched."""
    if not isinstance(args, dict):
        return {}
    enums = {}
    if isinstance(name, str) and isinstance(registry, dict):
        enums = (registry.get(name) or {}).get("enums") or {}
    out = {}
    for k, v in args.items():
        if k in enums:
            out[k] = snap_enum_value(v, enums[k])
        else:
            out[k] = v
    return out


def canonicalize_call(name: Any, args: Any, registry: dict | None = None) -> dict:
    """Full post-process for a predicted call's arguments: enum-snap (if a
    registry is given) then value canonicalization."""
    a = snap_enums(name, args, registry) if registry else args
    return canonicalize_args(a)


if __name__ == "__main__":
    # Self-test against the hard-floor examples from Iter 28 analysis.
    assert canonicalize_time_string("3 PM") == "15:00", canonicalize_time_string("3 PM")
    assert canonicalize_time_string("5:30 AM") == "05:30"
    assert canonicalize_time_string("8am") == "08:00"
    assert canonicalize_time_string("Saturday 8am") == "Saturday 08:00"
    assert canonicalize_time_string("12 AM") == "00:00"
    assert canonicalize_time_string("12 PM") == "12:00"
    assert canonicalize_time_string("15:00") == "15:00"
    assert canonicalize_time_string("9:05") == "09:05"
    assert canonicalize_value("temperature_c", 24.444444) == 24.4
    assert canonicalize_value("temperature_c", "24.44") == 24.4
    assert canonicalize_value("time", "3 PM") == "15:00"
    # Time normalization is key-gated: a song title is left untouched.
    assert canonicalize_value("song", "3 PM party mix") == "3 PM party mix"
    assert canonicalize_value("time", "Saturday 8am") == "Saturday 08:00"
    assert canonicalize_args({"time": "2 PM", "message": "meeting"}) == {
        "time": "14:00", "message": "meeting"
    }
    # Day plural normalization, key-gated.
    assert canonicalize_value("day", "Saturdays") == "Saturday"
    assert canonicalize_value("day", "weekdays") == "weekdays"
    assert canonicalize_value("day", "daily") == "daily"
    assert canonicalize_value("label", "Mondays") == "Mondays"  # not a day key
    # Enum snapping.
    _room = ["living room", "bedroom", "basement gym", "living_room"]
    assert snap_enum_value("Living Room", _room) == "living room"   # case
    assert snap_enum_value("living_room", _room) == "living room"   # underscore
    assert snap_enum_value("gym", _room) == "basement gym"          # substring
    assert snap_enum_value("bedroom", _room) == "bedroom"           # exact
    assert snap_enum_value("garage", _room) == "garage"             # no match
    _reg = {"dim_light": {"enums": {"room": _room}}}
    assert snap_enums("dim_light", {"room": "gym", "brightness_pct": 30}, _reg) == {
        "room": "basement gym", "brightness_pct": 30
    }
    assert snap_enums("unknown_fn", {"room": "gym"}, _reg) == {"room": "gym"}
    print("OK: canon.py self-test pass")
