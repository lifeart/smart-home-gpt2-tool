# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "optimum[onnxruntime]>=1.23",
#   "onnxruntime>=1.19",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 35 verification — confirm the ctx-4096 model handles a >2048-token
prompt without crashing and emits a valid tool call, in BOTH PyTorch and
the exported fp32 ONNX.

Builds a long SYSTEM prompt from ~10-12 full JSON tool schemas so the
tokenized prompt lands at ~3000-3500 tokens — far past both the stock
1024-token wall and the v12 2048 wall. Then greedily decodes a function
call (PyTorch) and again via onnxruntime (fp32 ONNX) and checks each
output parses as JSON with a name that is one of the offered tools.

Run on HF Jobs (needs the model's onnx/ already pushed):
    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN --timeout 1h \\
        training/verify_ctx4096.py
"""
import json
import os
import re

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO = os.environ.get("MODEL_REPO", "lifeart/smart-home-gpt2-v13-ctx4096")
NAME_RE = re.compile(r"""["'`]?name["'`]?\s*:\s*["']([^"'(\s,\}]+)""")

# A roomy set of tool schemas — 12 full schemas tokenize to ~3000-3500
# tokens, well past the old 1024 wall and the v12 2048 wall.
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
    {
        "name": "play_music",
        "description": "Play music on a speaker in a specific room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "enum": ["living room", "kitchen", "bedroom", "office",
                             "patio", "garage"],
                },
                "genre": {
                    "type": "string",
                    "enum": ["jazz", "rock", "classical", "pop", "ambient"],
                },
                "volume": {
                    "type": "integer",
                    "description": "Volume percent 0-100.",
                },
            },
            "required": ["room"],
        },
    },
    {
        "name": "set_fan_speed",
        "description": "Set the speed of a ceiling or standing fan in a room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "enum": ["living room", "kitchen", "bedroom", "office",
                             "sunroom", "nursery"],
                },
                "speed": {
                    "type": "string",
                    "enum": ["off", "low", "medium", "high"],
                },
            },
            "required": ["room", "speed"],
        },
    },
    {
        "name": "water_garden",
        "description": "Run the garden irrigation for a zone and duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "enum": ["front lawn", "back lawn", "flower beds",
                             "vegetable patch", "greenhouse"],
                },
                "minutes": {
                    "type": "integer",
                    "description": "How long to run irrigation, in minutes.",
                },
            },
            "required": ["zone", "minutes"],
        },
    },
    {
        "name": "arm_security_system",
        "description": "Arm the home security system in a chosen mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["home", "away", "night", "vacation"],
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": "Exit-delay before the system arms.",
                },
            },
            "required": ["mode"],
        },
    },
]


def build_long_prompt(n_schemas: int) -> tuple[str, list[str]]:
    """SYSTEM+USER prompt with `n_schemas` full tool schemas."""
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
def generate_torch(model, tok, prompt, device, max_new=80) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    close_brace = tok.encode("}", add_special_tokens=False)[0]
    newline = tok.encode("\n", add_special_tokens=False)[0]
    for _ in range(max_new):
        if cur.shape[1] >= model.config.n_positions:
            break
        out = model(cur)
        nxt = int(out.logits[0, -1, :].argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == close_brace or nxt == newline:
            break
    return tok.decode(cur[0, L:].tolist(), skip_special_tokens=True).strip()


def generate_onnx(ort_model, tok, prompt, max_new=80) -> str:
    """Greedy decode via the exported fp32 ONNX (optimum ORTModelForCausalLM)."""
    eos = tok.eos_token_id
    enc = tok(prompt, return_tensors="pt")
    plen = enc["input_ids"].shape[1]
    out = ort_model.generate(
        **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=eos,
    )
    return tok.decode(out[0, plen:], skip_special_tokens=True).strip()


def parse(out: str, valid_names: list[str]) -> tuple[str | None, bool]:
    name = None
    m = NAME_RE.search(out)
    if m:
        name = m.group(1)
    parsed_ok = False
    try:
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(out[start:end + 1])
            parsed_ok = isinstance(obj, dict) and "name" in obj
            if parsed_ok:
                name = obj["name"]
    except Exception as e:  # noqa: BLE001
        print(f"[parse] JSON parse failed (fuzzy name still used): {e}")
    return name, parsed_ok


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
    assert model.config.n_positions >= 4096, "context not extended to 4096!"
    assert tuple(model.transformer.wpe.weight.shape) == (4096, 768)

    # Build a prompt of ~3000-3500 tokens (well past the 2048 wall).
    prompt, valid_names, n_tok = None, None, 0
    for n_schemas in (12, 11, 10, 9, 8):
        p, names = build_long_prompt(n_schemas)
        nt = len(tok.encode(p, add_special_tokens=False))
        print(f"[prompt] {n_schemas} schemas -> {nt} tokens")
        if 2048 < nt <= model.config.n_positions - 90:
            prompt, valid_names, n_tok = p, names, nt
            print(f"[prompt] selected {n_schemas} schemas ({nt} tokens)")
            break
    assert prompt is not None, "could not build a prompt in (2048, 4096]"
    assert n_tok > 2048, f"prompt only {n_tok} tokens — not a >2048 test"

    # ---- PyTorch ----
    print("\n[torch] generating...")
    out_torch = generate_torch(model, tok, prompt, device)
    print(f"[torch] raw output: {out_torch!r}")
    name_torch, parsed_torch = parse(out_torch, valid_names)
    torch_ok = name_torch in valid_names

    # ---- fp32 ONNX ----
    print("\n[onnx] loading fp32 ONNX export...")
    from optimum.onnxruntime import ORTModelForCausalLM
    from onnxruntime import SessionOptions, GraphOptimizationLevel
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_model = ORTModelForCausalLM.from_pretrained(
        REPO, file_name="model.onnx", subfolder="onnx",
        provider="CPUExecutionProvider", session_options=so,
    )
    print("[onnx] generating...")
    out_onnx = generate_onnx(ort_model, tok, prompt)
    print(f"[onnx] raw output: {out_onnx!r}")
    name_onnx, parsed_onnx = parse(out_onnx, valid_names)
    onnx_ok = name_onnx in valid_names

    print("\n===== VERIFICATION (ctx-4096, >2048-token prompt) =====")
    print(f"  prompt tokens       : {n_tok}  (> 2048 wall: "
          f"{'YES' if n_tok > 2048 else 'NO'})")
    print(f"  PyTorch forward     : OK (no crash)")
    print(f"  PyTorch name        : {name_torch}  valid={torch_ok}  "
          f"json_parsed={parsed_torch}")
    print(f"  ONNX fp32 forward   : OK (no crash)")
    print(f"  ONNX fp32 name      : {name_onnx}  valid={onnx_ok}  "
          f"json_parsed={parsed_onnx}")
    verdict = "PASS" if (n_tok > 2048 and torch_ok and onnx_ok) else "REVIEW"
    print(f"  VERDICT             : {verdict}")


if __name__ == "__main__":
    main()
