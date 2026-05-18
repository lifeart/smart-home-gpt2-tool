"""Generate 500 SFT pairs for smart-home tool-calling. Templates per tool."""
import json, random
from pathlib import Path

random.seed(42)

ROOMS = ["living room", "kitchen", "bedroom", "bathroom", "office", "hallway", "garage", "garden"]
SONGS = ["jazz playlist", "rock anthems", "lofi beats", "classical piano", "ambient soundscape", "morning radio"]
DOORS = ["front door", "back door", "garage door", "patio door"]

TOOLS = {
    "turn_on_light": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "Turn on the lights in the {room}",
            "Lights on in the {room} please",
            "Switch on the {room} lights",
            "Can you light up the {room}?",
            "Power on the {room} lighting",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
    "turn_off_light": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "Turn off the lights in the {room}",
            "Switch off the {room} lights",
            "{room} lights off",
            "Kill the lights in the {room}",
            "Power down the {room} lighting",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
    "set_thermostat": {
        "params": {"room": {"type": "string"}, "temperature_c": {"type": "integer"}}, "required": ["room", "temperature_c"],
        "templates": [
            "Set the {room} thermostat to {temperature_c} degrees",
            "I want the {room} at {temperature_c} celsius",
            "Heat the {room} to {temperature_c} degrees",
            "Cool the {room} down to {temperature_c}",
            "Make the {room} {temperature_c} degrees please",
        ],
        "args": lambda: {"room": random.choice(ROOMS), "temperature_c": random.randint(16, 26)},
    },
    "lock_door": {
        "params": {"door": {"type": "string"}}, "required": ["door"],
        "templates": [
            "Lock the {door}",
            "Please lock the {door}",
            "Engage the {door} lock",
            "Secure the {door}",
            "Bolt the {door}",
        ],
        "args": lambda: {"door": random.choice(DOORS)},
    },
    "unlock_door": {
        "params": {"door": {"type": "string"}}, "required": ["door"],
        "templates": [
            "Unlock the {door}",
            "Open the {door} lock",
            "Disengage the {door} lock",
            "Release the {door}",
        ],
        "args": lambda: {"door": random.choice(DOORS)},
    },
    "play_music": {
        "params": {"song": {"type": "string"}, "room": {"type": "string"}}, "required": ["song", "room"],
        "templates": [
            "Play {song} in the {room}",
            "I want to hear {song} in the {room}",
            "Start {song} in the {room}",
            "Put on {song} in the {room} speaker",
        ],
        "args": lambda: {"song": random.choice(SONGS), "room": random.choice(ROOMS)},
    },
    "stop_music": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "Stop the music in the {room}",
            "Silence the {room} speaker",
            "Kill the music in the {room}",
            "Pause music in the {room}",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
    "set_alarm": {
        "params": {"time": {"type": "string"}}, "required": ["time"],
        "templates": [
            "Set an alarm for {time}",
            "Wake me up at {time}",
            "Alarm at {time} please",
            "Schedule an alarm for {time} tomorrow",
        ],
        "args": lambda: {"time": f"{random.randint(5,11):02d}:{random.choice(['00','15','30','45'])}"},
    },
    "open_curtains": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "Open the curtains in the {room}",
            "Pull back the {room} curtains",
            "Raise the {room} blinds",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
    "close_curtains": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "Close the curtains in the {room}",
            "Draw the {room} curtains shut",
            "Lower the {room} blinds",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
    "start_robot_vacuum": {
        "params": {}, "required": [],
        "templates": [
            "Start the robot vacuum",
            "Run the Roomba",
            "Begin vacuuming",
            "Let the vacuum clean up",
        ],
        "args": lambda: {},
    },
    "query_temperature": {
        "params": {"room": {"type": "string"}}, "required": ["room"],
        "templates": [
            "What is the temperature in the {room}?",
            "How warm is the {room}?",
            "Check the {room} thermostat reading",
            "Tell me the {room} temperature",
        ],
        "args": lambda: {"room": random.choice(ROOMS)},
    },
}


def fn_spec(name):
    t = TOOLS[name]
    return {
        "name": name,
        "description": f"Smart home action: {name.replace('_',' ')}",
        "parameters": {"type": "object", "properties": t["params"], "required": t["required"]},
    }


def gen_one():
    name = random.choice(list(TOOLS.keys()))
    t = TOOLS[name]
    args = t["args"]()
    template = random.choice(t["templates"])
    user_text = template.format(**args)
    spec = fn_spec(name)
    prompt = (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        f"{json.dumps(spec, indent=2)}\n\n\n"
        f"USER: {user_text}\n\n\nASSISTANT: <functioncall> "
    )
    gold = json.dumps({"name": name, "arguments": args}, separators=(",", ":"))
    return {"prompt": prompt, "gold": gold, "name": name, "args": args, "user_text": user_text}


def main():
    out = []
    while len(out) < 500:
        out.append(gen_one())
    Path(__file__).resolve().parent.joinpath("smart_home_sft.json").write_text(json.dumps(out, indent=2))
    # split test 50
    test = []
    while len(test) < 50:
        test.append(gen_one())
    Path(__file__).resolve().parent.joinpath("smart_home_test.json").write_text(json.dumps(test, indent=2))
    counts = {}
    for s in out:
        counts[s["name"]] = counts.get(s["name"], 0) + 1
    print(f"SFT: {len(out)}, TEST: {len(test)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
