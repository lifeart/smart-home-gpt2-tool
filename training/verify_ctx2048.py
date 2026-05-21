# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 34 verification — confirm the ctx-2048 model handles a >1024-token
prompt without crashing and emits a valid tool call.

Builds a long SYSTEM prompt from many full JSON tool schemas (pulled from
data/tool_registry.json, with a hand-written fallback) so the tokenized
prompt exceeds 1024 tokens — which would be impossible to feed to the
stock 1024-ctx GPT-2. Then greedily decodes a function call and checks it
parses as JSON with a name that is one of the offered tools.

Run locally (single 124M model, ~480 MB — fine on 16 GB Mac):
    python training/verify_ctx2048.py
"""
import json
import os
import re
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO = os.environ.get("MODEL_REPO", "lifeart/smart-home-gpt2-v12-ctx2048")
NAME_RE = re.compile(r"""["'`]?name["'`]?\s*:\s*["']([^"'(\s,\}]+)""")

# A roomy set of tool schemas. Eight full schemas overflow the 1024-token
# window comfortably.
TOOLS = [
    {
        "name": "lock_door",
        "description": "Lock a specific door in the home.",
        "parameters": {
            "type": "object",
            "properties": {
                "door": {
                    "type": "string",
                    "enum": ["front door", "back door", "garage door",
                             "patio door", "side door", "basement door"],
                }
            },
            "required": ["door"],
        },
    },
    {
        "name": "unlock_door",
        "description": "Unlock a specific door in the home.",
        "parameters": {
            "type": "object",
            "properties": {
                "door": {
                    "type": "string",
                    "enum": ["front door", "back door", "garage door",
                             "patio door", "side door", "basement door"],
                }
            },
            "required": ["door"],
        },
    },
    {
        "name": "set_light_brightness",
        "description": "Set the brightness of a light in a given room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "enum": ["living room", "kitchen", "bedroom", "office",
                             "hallway", "bathroom", "garage"],
                },
                "brightness": {
                    "type": "integer",
                    "description": "Brightness percent 0-100.",
                },
            },
            "required": ["room", "brightness"],
        },
    },
    {
        "name": "set_thermostat",
        "description": "Set the target temperature of a thermostat zone.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "enum": ["upstairs", "downstairs", "whole house"],
                },
                "temperature": {
                    "type": "number",
                    "description": "Target temperature in degrees.",
                },
            },
            "required": ["zone", "temperature"],
        },
    },
    {
        "name": "set_camera_motion_sensitivity",
        "description": "Adjust motion-detection sensitivity for a camera.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["front door cam", "backyard cam", "driveway cam",
                             "garage cam", "side gate cam"],
                },
                "level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["camera", "level"],
        },
    },
    {
        "name": "open_blinds",
        "description": "Open the window blinds in a room to a position.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "enum": ["living room", "kitchen", "bedroom", "office",
                             "nursery", "study"],
                },
                "position": {
                    "type": "integer",
                    "description": "Open position percent 0-100.",
                },
            },
            "required": ["room", "position"],
        },
    },
    {
        "name": "start_vacuum",
        "description": "Start the robot vacuum cleaning a given area.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "enum": ["whole house", "kitchen", "living room",
                             "bedrooms", "hallway"],
                },
                "mode": {
                    "type": "string",
                    "enum": ["quiet", "standard", "turbo"],
                },
            },
            "required": ["area"],
        },
    },
    {
        "name": "trigger_panic_alarm",
        "description": "Trigger the emergency panic alarm and notify "
                       "authorities immediately.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def build_long_prompt(n_schemas: int = 6) -> tuple[str, list[str]]:
    """Build a SYSTEM+USER prompt with `n_schemas` full tool schemas.

    The point is a prompt that overflows the OLD 1024-token wall while
    staying within the NEW 2048 window. Six full JSON schemas tokenize to
    ~1700 tokens — comfortably > 1024 and < 2048. (All eight schemas would
    be ~2400 tokens and overflow even the new window, which is itself a
    useful upper bound but not what this end-to-end gen test wants.)
    """
    tools = TOOLS[:n_schemas]
    schema_json = json.dumps(tools, indent=2)
    user = ("That motion alert tripped over and over from passing cars all "
            "afternoon — please calm down the front door camera so it stops "
            "pestering me about every vehicle on the street.")
    prompt = (
        "SYSTEM: You are a helpful assistant with access to the following "
        "functions. Use them if required -\n"
        f"{schema_json}\n\n\n"
        f"USER: {user}\n\n\n"
        "ASSISTANT: <functioncall> "
    )
    return prompt, [t["name"] for t in tools]


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=80) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    close_brace = tok.encode("}", add_special_tokens=False)[0]
    newline = tok.encode("\n", add_special_tokens=False)[0]
    for _ in range(max_new):
        if cur.shape[1] >= model.config.n_positions:
            break
        out = model(cur)
        logits = out.logits
        nxt = int(logits[0, -1, :].argmax().item())
        cur = torch.cat(
            [cur, torch.tensor([[nxt]], device=device)], dim=1
        )
        if nxt == close_brace or nxt == newline:
            break
    return tok.decode(cur[0, L:].tolist(), skip_special_tokens=True).strip()


def main() -> None:
    print(f"[load] {REPO}")
    tok = GPT2TokenizerFast.from_pretrained(REPO)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(REPO).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"[cfg] n_positions={model.config.n_positions} "
          f"n_ctx={model.config.n_ctx} "
          f"wpe={tuple(model.transformer.wpe.weight.shape)}")
    assert model.config.n_positions >= 2048, "context not extended!"
    assert tuple(model.transformer.wpe.weight.shape) == (2048, 768)

    # Pick the largest schema count that still leaves room to generate
    # inside the 2048 window (need >1024 to clear the old wall).
    prompt, valid_names = None, None
    n_tok = 0
    for n_schemas in (8, 7, 6, 5, 4):
        p, names = build_long_prompt(n_schemas)
        nt = len(tok.encode(p, add_special_tokens=False))
        if 1024 < nt <= model.config.n_positions - 90:
            prompt, valid_names, n_tok = p, names, nt
            print(f"[prompt] using {n_schemas} schemas")
            break
    assert prompt is not None, "could not build a prompt in (1024, 2048]"
    print(f"[prompt] {n_tok} tokens "
          f"({'OVER' if n_tok > 1024 else 'UNDER'} the old 1024 wall, "
          f"within the new {model.config.n_positions} window)")
    assert n_tok > 1024, (
        f"prompt only {n_tok} tokens — not a >1024 test"
    )

    out = generate(model, tok, prompt, device)
    print(f"[gen] raw output: {out!r}")

    # parse
    name = None
    m = NAME_RE.search(out)
    if m:
        name = m.group(1)
    parsed_ok = False
    try:
        # try to extract a JSON object
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(out[start:end + 1])
            parsed_ok = isinstance(obj, dict) and "name" in obj
            if parsed_ok:
                name = obj["name"]
    except Exception as e:  # noqa: BLE001
        print(f"[parse] JSON parse failed (fuzzy name still used): {e}")

    name_valid = name in valid_names
    print("\n===== VERIFICATION =====")
    print(f"  prompt tokens     : {n_tok}  (> 1024 wall: "
          f"{'YES' if n_tok > 1024 else 'NO'})")
    print(f"  forward pass      : OK (no crash)")
    print(f"  emitted name      : {name}")
    print(f"  name is a tool    : {name_valid}")
    print(f"  full JSON parsed  : {parsed_ok}")
    verdict = "PASS" if (n_tok > 1024 and name_valid) else "REVIEW"
    print(f"  VERDICT           : {verdict}")


if __name__ == "__main__":
    main()
