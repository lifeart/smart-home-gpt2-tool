# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 23 — shared bench helpers.

Port of web/bench.js scoring (exact match: name + tolerant argsMatch) into Python
so HF-side benches can report the same exact-match number the browser uses.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
LOCAL_TEST = ROOT / "data" / "sh_test.json"
TEST_URL = (
    "https://raw.githubusercontent.com/barometech/smart-home-gpt2/master/"
    "data/sh_test.json"
)

ARGS_HINT_TMPL = (
    "Note: The function name will be: {name}. Output the arguments only.\n\n\n"
)


# ---------------- test loader ----------------

def load_test() -> list[dict]:
    if LOCAL_TEST.exists():
        return json.loads(LOCAL_TEST.read_text())
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            repo_id="lifeart/smart-home-sft-v2",
            filename="sh_test.json",
            repo_type="dataset",
        )
        return json.loads(Path(p).read_text())
    except Exception as e:
        print(f"[test] hub fetch failed: {e}; trying GitHub")
    with urllib.request.urlopen(TEST_URL) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------- parsers ----------------

NAME_RE = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"')


def extract_predicted_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_call(text: str) -> Optional[dict]:
    """Return {'name': str|None, 'arguments': dict} or None if unparseable."""
    js = extract_predicted_json(text)
    if not js:
        return None
    try:
        obj = json.loads(js)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "arguments": args}


def parse_args_only(text: str) -> Optional[dict]:
    """For args_only mode: gold is {'arguments': {...}}.

    Returns the arguments dict or None.
    """
    js = extract_predicted_json(text)
    if not js:
        return None
    try:
        obj = json.loads(js)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # v9 args_only emits {"arguments": {...}}; tolerate either shape
    if "arguments" in obj and isinstance(obj["arguments"], dict):
        return obj["arguments"]
    # Tolerate naked args dict (no wrapper)
    if "arguments" not in obj and "name" not in obj:
        return obj
    return None


def parse_gold(gold_str: str) -> dict:
    """Returns {'name': str|None, 'arguments': dict}."""
    if not isinstance(gold_str, str):
        return {"name": None, "arguments": {}}
    try:
        obj = json.loads(gold_str)
    except Exception:
        return {"name": None, "arguments": {}}
    if not isinstance(obj, dict):
        return {"name": None, "arguments": {}}
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "arguments": args}


# ---------------- scoring (mirrors web/bench.js) ----------------

def _normalize_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t and re.fullmatch(r"-?\d+(\.\d+)?", t):
            try:
                return float(t) if "." in t else int(t)
            except ValueError:
                return t.lower()
        return t.lower()
    return v


def args_match(pred: Any, gold: Any) -> bool:
    """Tolerant compare per web/bench.js argsMatch.

    JS: `Object.keys(a).filter(k => a[k] !== undefined)` — JSON.parse never
    produces undefined, so in Python this is just `a.keys()`. We keep nulls
    in both pred and gold because `{"k": null}` is a real annotation.
    """
    a = pred if isinstance(pred, dict) else {}
    b = gold if isinstance(gold, dict) else {}
    ak = sorted(a.keys())
    bk = sorted(b.keys())
    if ak != bk:
        return False
    for k in ak:
        av = a[k]
        bv = b[k]
        if isinstance(bv, list):
            if not isinstance(av, list):
                return False
            if len(av) != len(bv):
                return False
            as_ = sorted(_normalize_scalar(x) for x in av)
            bs_ = sorted(_normalize_scalar(x) for x in bv)
            try:
                if as_ != bs_:
                    return False
            except TypeError:
                # Mixed types — fall back to json compare
                if json.dumps(as_, sort_keys=True) != json.dumps(bs_, sort_keys=True):
                    return False
        elif isinstance(bv, dict):
            if not isinstance(av, dict):
                return False
            if not args_match(av, bv):
                return False
        else:
            an = _normalize_scalar(av)
            bn = _normalize_scalar(bv)
            if an is None and bn is None:
                continue
            if an is None or bn is None:
                return False
            if isinstance(an, (int, float)) and isinstance(bn, (int, float)):
                if an != bn:
                    return False
            else:
                if str(an) != str(bn):
                    return False
    return True


def score(pred_name: Optional[str], pred_args: Any, gold_str: str) -> dict:
    gold = parse_gold(gold_str)
    name_ok = pred_name == gold["name"] and gold["name"] is not None
    args_ok = args_match(pred_args, gold["arguments"])
    return {"name_ok": name_ok, "args_ok": args_ok, "exact_ok": name_ok and args_ok}


# ---------------- prompt helpers ----------------

def build_args_only_prompt(prompt: str, predicted_name: str) -> str:
    """Inject Granite args_only hint between USER and ASSISTANT.

    Mirrors training/relabel_granite.py exactly so v9 sees the same format it
    was trained on.
    """
    hint = ARGS_HINT_TMPL.format(name=predicted_name)
    marker = "\n\n\nASSISTANT:"
    if marker in prompt:
        head, tail = prompt.split(marker, 1)
        return head + "\n" + hint + "ASSISTANT:" + tail
    return prompt + " " + hint


# ---------------- per-domain aggregation ----------------

def aggregate(rows: list[dict], mode_key: str = "") -> dict:
    """Build summary with overall and per-domain breakdown.

    Each row should have {name_ok, args_ok, exact_ok, domain} (optionally
    prefixed with `${mode_key}_`).
    """
    p = (mode_key + "_") if mode_key else ""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    name_c = sum(1 for r in rows if r.get(p + "name_ok"))
    args_c = sum(1 for r in rows if r.get(p + "args_ok"))
    exact_c = sum(1 for r in rows if r.get(p + "exact_ok"))
    by_domain: dict[str, dict] = {}
    for r in rows:
        d = r.get("domain", "?")
        b = by_domain.setdefault(d, {"n": 0, "name": 0, "args": 0, "exact": 0})
        b["n"] += 1
        if r.get(p + "name_ok"):
            b["name"] += 1
        if r.get(p + "args_ok"):
            b["args"] += 1
        if r.get(p + "exact_ok"):
            b["exact"] += 1
    return {
        "n": n,
        "name_acc": name_c / n,
        "args_acc": args_c / n,
        "exact_acc": exact_c / n,
        "by_domain": by_domain,
    }


def print_summary(label: str, summ: dict) -> None:
    n = summ.get("n", 0)
    if not n:
        print(f"  {label}: n=0")
        return
    print(
        f"  {label}: name={summ['name_acc']*100:5.1f}% "
        f"args={summ['args_acc']*100:5.1f}% "
        f"exact={summ['exact_acc']*100:5.1f}% "
        f"(n={n})"
    )
    for d, b in sorted(summ.get("by_domain", {}).items()):
        print(
            f"    {d:<10} name={b['name']/b['n']*100:5.1f}% "
            f"args={b['args']/b['n']*100:5.1f}% "
            f"exact={b['exact']/b['n']*100:5.1f}% "
            f"({b['exact']}/{b['n']})"
        )


# ---------------- HF token ----------------

def load_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    p = Path("~/.cache/huggingface/token").expanduser()
    if p.exists():
        return p.read_text().strip()
    return None
