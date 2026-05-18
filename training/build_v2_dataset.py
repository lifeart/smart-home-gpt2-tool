# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
# ]
# ///
"""Build merged smart-home SFT v2 dataset.

Sources (in priority order):
  A. existing 1200 SH items (data/sh_train.json) x5 upsample = 6000
  B. MadeAgents/xlam-irrelevance-7.5k (cap 2500) - empty-call sentinel rows
  C. NousResearch/hermes-function-calling-v1 / func_calling_singleturn
       filtered to IoT-flavored rows (cap 500)
  D. Salesforce/xlam-function-calling-60k (gated, SKIPPED)

Empty-call sentinel: gold = {"name":"none","arguments":{}}, gold_name = "none".
"none" is not in tool_registry.json, so no real-tool collision. The model
learns "no function applies" as a literal output the JSON parser can handle.

Outputs data/sh_train_v2.json (list of {prompt, gold, gold_name, domain}).
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
TRAIN_IN = ROOT / "data" / "sh_train.json"
OUT_PATH = ROOT / "data" / "sh_train_v2.json"

UPSAMPLE = 5
CAP_IRRELEVANCE = 2500
CAP_HERMES = 500

PROMPT_TMPL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n{tools}\n\n\n"
    "USER: {query}\n\n\nASSISTANT: <functioncall> "
)

NONE_GOLD = '{"name":"none","arguments":{}}'
NONE_NAME = "none"

# IoT-relevant keyword filter (broad — better recall, we cap at 500)
IOT_KW = re.compile(
    r"\b(light|lamp|fan|thermostat|alarm|music|switch|door|scene|security|"
    r"vacuum|camera|blind|curtain|garage|sprinkler|irrigation|smart\s*home|"
    r"home_?automation|plug|outlet|appliance|window|temperature|climate|hvac|"
    r"heater|cooler|tv|television|speaker|sound|volume|brightness|lock|"
    r"unlock|garden|plant|kitchen|stove|oven|microwave|coffee|fridge|"
    r"refrigerator|cleaner|robot|hum(idi(fier|ty)|id)|air|purifier|sensor|"
    r"motion|window|shade|shutter)\b",
    re.IGNORECASE,
)


def load_base() -> list[dict]:
    with TRAIN_IN.open() as f:
        return json.load(f)


def adapt_irrelevance(cap: int) -> list[dict]:
    """xlam-irrelevance has: query (str), tools (json str), answers='[]'.

    We trust the dataset's irrelevance label — answers is always empty
    or near-empty. We surface a compact list-of-tool-names so the prompt
    matches our standard schema.
    """
    print(f"[B] loading xlam-irrelevance-7.5k …")
    ds = load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")
    print(f"[B] {len(ds)} rows total")
    out: list[dict] = []
    skipped = 0
    for row in ds:
        try:
            tools = json.loads(row["tools"])
        except Exception:
            skipped += 1
            continue
        # Surface short list of names + descriptions (truncated) — keeps
        # prompt length sane while preserving the "candidate tools" signal.
        names = []
        for t in tools[:10]:
            n = t.get("name") if isinstance(t, dict) else None
            if n:
                names.append(n)
        if not names:
            skipped += 1
            continue
        tool_block = json.dumps(names, indent=2)
        prompt = PROMPT_TMPL.format(tools=tool_block, query=row["query"])
        out.append({
            "prompt": prompt,
            "gold": NONE_GOLD,
            "gold_name": NONE_NAME,
            "domain": "irrelevant",
        })
        if len(out) >= cap:
            break
    print(f"[B] adapted {len(out)} rows (skipped {skipped})")
    return out


def _extract_first_tool_call(gpt_value: str) -> dict | None:
    """Hermes gpt response is a sequence of <tool_call>{json}</tool_call>.

    Return the first parsed call (dict with name+arguments) or None.
    """
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gpt_value, re.S)
    if not m:
        # Sometimes the JSON is bare without explicit closing tag
        m = re.search(r"<tool_call>\s*(\{.*\})", gpt_value, re.S)
        if not m:
            return None
    blob = m.group(1).strip()
    # Heuristic: keep cutting trailing junk until json.loads works
    for cut in range(len(blob), 0, -1):
        try:
            obj = json.loads(blob[:cut])
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except Exception:
            continue
    return None


def adapt_hermes(cap: int) -> list[dict]:
    print(f"[C] loading hermes-function-calling-v1 (singleturn) …")
    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        "func_calling_singleturn",
        split="train",
    )
    print(f"[C] {len(ds)} rows total")
    out: list[dict] = []
    skipped_no_iot = 0
    skipped_no_call = 0
    skipped_no_user = 0
    for row in ds:
        cat = (row.get("category") or "") + " " + (row.get("subcategory") or "")
        tools_str = row.get("tools") or ""
        haystack = cat + " " + tools_str[:2000]
        if not IOT_KW.search(haystack):
            skipped_no_iot += 1
            continue
        # Extract user query
        convs = row.get("conversations") or []
        user = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if not user or not gpt:
            skipped_no_user += 1
            continue
        call = _extract_first_tool_call(gpt)
        if not call:
            skipped_no_call += 1
            continue
        # Build short tool list (names only, like our v2 schema upper rows)
        try:
            tools = json.loads(tools_str)
        except Exception:
            skipped_no_call += 1
            continue
        names = []
        for t in tools:
            n = None
            if isinstance(t, dict):
                if "function" in t and isinstance(t["function"], dict):
                    n = t["function"].get("name")
                else:
                    n = t.get("name")
            if n:
                names.append(n)
        if not names:
            skipped_no_call += 1
            continue
        # Truncate user message if super long (the IoT prompts can be ~2KB)
        u = user.strip()
        if len(u) > 800:
            u = u[:800].rsplit(" ", 1)[0] + " …"
        prompt = PROMPT_TMPL.format(tools=json.dumps(names, indent=2), query=u)
        gold = json.dumps({"name": call["name"], "arguments": call.get("arguments", {})})
        out.append({
            "prompt": prompt,
            "gold": gold,
            "gold_name": call["name"],
            "domain": "iot_aux",
        })
        if len(out) >= cap:
            break
    print(
        f"[C] adapted {len(out)} rows "
        f"(skipped: no_iot={skipped_no_iot} no_call={skipped_no_call} "
        f"no_user={skipped_no_user})"
    )
    return out


def main() -> None:
    random.seed(42)

    base = load_base()
    print(f"[A] base SH items: {len(base)} × {UPSAMPLE} = {len(base) * UPSAMPLE}")
    sh_rows = base * UPSAMPLE

    irrel_rows = adapt_irrelevance(CAP_IRRELEVANCE)
    hermes_rows = adapt_hermes(CAP_HERMES)
    # xlam-60k is gated — skipping per plan
    print("[D] xlam-60k: SKIPPED (gated dataset, no access)")

    merged = sh_rows + irrel_rows + hermes_rows
    random.shuffle(merged)
    print(f"[merge] total = {len(merged)} rows")

    counts = {}
    for r in merged:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    print("[merge] domain breakdown:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}  {v}")

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[save] wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
