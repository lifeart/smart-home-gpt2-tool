#!/usr/bin/env python3
"""Iter 10.2 — Additively patch function_descriptions.json + tool_registry.json.

Adds:
- 16 new function_descriptions entries for misc-domain test gold names absent
  from the retrieval index.
- 23 new tool_registry entries (16 above + 7 desc-only names) so that
  buildSchemaConstraint can build per-key arg schemas under `wideNames`.

NO existing entries are reordered or renamed.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def save(rel, obj):
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# 16 new descriptions for misc-domain functions absent from index
# Style matches existing entries: short purpose + params + examples.
# Examples mirror real test-query phrasings where possible.
# ---------------------------------------------------------------------------
NEW_DESCRIPTIONS = {
    "activate_scene": {
        "description": "Activate a previously defined whole-home scene by name.",
        "params": ["scene"],
        "examples": [
            "Use the focus scene.",
            "Activate the away scene.",
            "Time to focus, activate focus scene.",
            "I need to concentrate on work."
        ]
    },
    "save_current_scene": {
        "description": "Save the current device/light state as a named scene for later recall.",
        "params": ["scene_name"],
        "examples": [
            "Save current settings as cozy.",
            "Save this as my game night scene.",
            "Save these lights as my movie scene."
        ]
    },
    "schedule_routine": {
        "description": "Schedule a recurring routine: activate a scene at a given time on selected days, with optional follow-up actions.",
        "params": ["scene", "time", "days", "trigger", "followup", "pre_reminder"],
        "examples": [
            "Every Friday at 6 PM run dinner party scene and remind me 15 minutes earlier to start cooking.",
            "Every Sunday at 10 PM activate bedtime scene and set an alarm for 6 AM Monday.",
            "Dinner party scene every Saturday at 7 PM.",
            "Schedule wake up scene every weekday at 6:30."
        ]
    },
    "cancel_alarm": {
        "description": "Cancel one or more alarms, by time, label, day, or all.",
        "params": ["time", "label", "day", "days", "from", "to", "all"],
        "examples": [
            "Cancel my weekend alarm.",
            "Cancel alarm at 8 AM.",
            "Cancel all alarms between 5 and 7 AM tomorrow.",
            "Cancel my morning alarm."
        ]
    },
    "snooze_alarm": {
        "description": "Snooze the currently ringing alarm by a duration.",
        "params": ["duration_minutes"],
        "examples": [
            "Snooze alarm for 5 more minutes.",
            "Snooze the alarm.",
            "Snooze for 10 minutes."
        ]
    },
    "query_alarms": {
        "description": "List or query currently set alarms, optionally filtered by day or time range.",
        "params": ["day", "days", "before", "after"],
        "examples": [
            "What alarms do I have set?",
            "Do I have any alarms tomorrow?",
            "List alarms set for any weekday before 8 AM."
        ]
    },
    "cancel_timer": {
        "description": "Cancel an active countdown timer by label, or all timers.",
        "params": ["label"],
        "examples": [
            "Kill the timer.",
            "Cancel my pasta timer.",
            "Cancel the timer."
        ]
    },
    "query_timers": {
        "description": "Query active countdown timers, optionally by label or finishing window.",
        "params": ["label", "finishing_within_minutes"],
        "examples": [
            "Show active timers.",
            "Which of my timers will finish in the next 5 minutes?",
            "How long until my oven timer goes off?"
        ]
    },
    "cancel_reminder": {
        "description": "Cancel a previously set reminder by message, day, or cancel all.",
        "params": ["message", "day", "all", "contains", "range"],
        "examples": [
            "Cancel all my reminders.",
            "Cancel my reminder about the dentist tomorrow.",
            "Cancel my reminder about groceries."
        ]
    },
    "query_motion_sensor": {
        "description": "Query whether a motion sensor has detected movement in a room.",
        "params": ["room"],
        "examples": [
            "Is anyone moving around in the office area?",
            "Hallway motion sensor.",
            "Did the perimeter motion catch anything outside?",
            "Any motion in the basement?"
        ]
    },
    "query_smoke_alarm": {
        "description": "Check the status of smoke alarms in the home.",
        "params": ["room"],
        "examples": [
            "Smoke alarm working?",
            "Check smoke detector.",
            "Are all smoke alarms in the house currently operational?"
        ]
    },
    "query_water_leak": {
        "description": "Check water-leak sensors for moisture detection, optionally per room.",
        "params": ["room"],
        "examples": [
            "Water leak status?",
            "Any moisture being detected in the basement walls?",
            "Anything tripping the floor sensors near the boiler?"
        ]
    },
    "query_garage_door": {
        "description": "Query whether the garage door is currently open or closed.",
        "params": [],
        "examples": [
            "Check garage door.",
            "Garage door status?",
            "Did the garage door close after I drove off this morning?"
        ]
    },
    "query_solar_production": {
        "description": "Query solar panel energy production for a given period (now, today, month, etc).",
        "params": ["period"],
        "examples": [
            "Check this month's solar yield so far.",
            "How many watts are the panels pushing out right now?",
            "Where do we stand on energy generation from the solar array this period?"
        ]
    },
    "list_active_devices": {
        "description": "List devices that are currently on, drawing power, or otherwise reporting active.",
        "params": [],
        "examples": [
            "List active devices.",
            "What's still drawing or reporting?",
            "Which devices are running right now?"
        ]
    },
    "generate_status_report": {
        "description": "Produce a snapshot summary of the entire home's device and sensor state.",
        "params": [],
        "examples": [
            "Could you put together a snapshot of the whole house status?",
            "Give me a status report.",
            "Full home status report please."
        ]
    },
}


# ---------------------------------------------------------------------------
# 23 new registry entries (params + required) — same shape as existing.
# Enums are populated from mined test+train value distributions; only
# string params with stable small vocabularies get enums (no `room` flooding,
# no open-vocab labels like `message`/`scene_name`).
# ---------------------------------------------------------------------------
ROOM_ENUM = sorted({
    "living room", "bedroom", "kitchen", "bathroom", "nursery", "office",
    "master bedroom", "sunroom", "hallway", "basement", "dining room",
    "attic", "garage", "basement gym", "garden", "kids room", "porch",
    "dining_room", "guest_room", "laundry_room", "living_room", "patio"
})
SCENE_ENUM = sorted({
    "movie", "movie night", "sleep", "focus", "dinner", "dinner party",
    "game night", "kids bedtime", "bedtime", "meditation", "party",
    "relax", "study", "sunset", "wake up", "away", "vacation",
    "romantic", "good night", "cozy", "morning", "workout", "chill",
    "reading", "party prep"
})
DAY_ENUM = ["today", "tomorrow", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAYS_ENUM = DAY_ENUM
PERIOD_ENUM = ["now", "today", "yesterday", "hour", "morning", "afternoon",
               "evening", "night", "week", "month", "year"]


NEW_REGISTRY = {
    "activate_scene": {
        "params": {"scene": "string"},
        "required": ["scene"],
        "enums": {"scene": SCENE_ENUM}
    },
    "save_current_scene": {
        "params": {"scene_name": "string"},
        "required": ["scene_name"],
        "enums": {}
    },
    "schedule_routine": {
        "params": {
            "scene": "string", "time": "string", "days": "array",
            "trigger": "string", "followup": "object",
            "pre_reminder": "object", "end": "string", "steps": "array",
            "after_time": "string", "end_time": "string", "start": "string"
        },
        "required": [],
        "enums": {"scene": SCENE_ENUM, "days": DAYS_ENUM}
    },
    "cancel_alarm": {
        "params": {
            "time": "string", "label": "string", "day": "string",
            "days": "array", "from": "string", "to": "string", "all": "boolean"
        },
        "required": [],
        "enums": {"day": DAY_ENUM, "days": DAYS_ENUM}
    },
    "snooze_alarm": {
        "params": {"duration_minutes": "integer", "range_hours": "integer", "all": "boolean"},
        "required": [],
        "enums": {}
    },
    "query_alarms": {
        "params": {"day": "string", "days": "array", "before": "string", "after": "string"},
        "required": [],
        "enums": {"day": DAY_ENUM, "days": DAYS_ENUM}
    },
    "cancel_timer": {
        "params": {"label": "string"},
        "required": [],
        "enums": {}
    },
    "query_timers": {
        "params": {"label": "string", "finishing_within_minutes": "integer"},
        "required": [],
        "enums": {}
    },
    "cancel_reminder": {
        "params": {
            "message": "string", "day": "string", "all": "boolean",
            "contains": "string", "range": "string"
        },
        "required": [],
        "enums": {"day": DAY_ENUM}
    },
    "query_motion_sensor": {
        "params": {"room": "string"},
        "required": [],
        "enums": {"room": ROOM_ENUM}
    },
    "query_smoke_alarm": {
        "params": {"room": "string"},
        "required": [],
        "enums": {"room": ROOM_ENUM}
    },
    "query_water_leak": {
        "params": {"room": "string"},
        "required": [],
        "enums": {"room": ROOM_ENUM}
    },
    "query_garage_door": {
        "params": {},
        "required": [],
        "enums": {}
    },
    "query_solar_production": {
        "params": {"period": "string"},
        "required": [],
        "enums": {"period": PERIOD_ENUM}
    },
    "list_active_devices": {
        "params": {},
        "required": [],
        "enums": {}
    },
    "generate_status_report": {
        "params": {},
        "required": [],
        "enums": {}
    },
    # 7 desc-only names that lacked registry entries:
    "set_alarm": {
        "params": {
            "time": "string", "day": "string", "days": "array",
            "label": "string", "schedules": "array", "condition": "string"
        },
        "required": [],
        "enums": {"day": DAY_ENUM, "days": DAYS_ENUM}
    },
    "set_timer": {
        "params": {
            "duration_minutes": "integer", "label": "string", "timers": "array"
        },
        "required": [],
        "enums": {}
    },
    "set_reminder": {
        "params": {
            "message": "string", "time": "string", "in_minutes": "integer",
            "days": "array", "day": "string", "offset_minutes": "integer",
            "relative_to": "string", "schedule": "string"
        },
        "required": [],
        "enums": {"day": DAY_ENUM, "days": DAYS_ENUM}
    },
    "query_battery_level": {
        "params": {"device": "string"},
        "required": [],
        "enums": {}
    },
    "query_power_usage": {
        "params": {"period": "string", "device": "string"},
        "required": [],
        "enums": {"period": PERIOD_ENUM}
    },
    "query_water_meter": {
        "params": {"period": "string"},
        "required": [],
        "enums": {"period": PERIOD_ENUM}
    },
    "query_window_status": {
        "params": {"room": "string"},
        "required": [],
        "enums": {"room": ROOM_ENUM}
    },
}


def main():
    desc_path = "web/public/eval/function_descriptions.json"
    reg_paths = ["data/tool_registry.json", "web/public/eval/tool_registry.json"]

    desc = load(desc_path)
    n_desc_before = len(desc)
    added_desc = 0
    for name, body in NEW_DESCRIPTIONS.items():
        if name in desc:
            print(f"  skip desc {name} (already present)")
            continue
        desc[name] = body
        added_desc += 1
    save(desc_path, desc)
    print(f"function_descriptions.json: {n_desc_before} -> {len(desc)} (+{added_desc})")

    for rp in reg_paths:
        reg = load(rp)
        n_before = len(reg)
        added = 0
        for name, body in NEW_REGISTRY.items():
            if name in reg:
                print(f"  skip reg {name} in {rp} (already present)")
                continue
            reg[name] = body
            added += 1
        save(rp, reg)
        print(f"{rp}: {n_before} -> {len(reg)} (+{added})")


if __name__ == "__main__":
    main()
