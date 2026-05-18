# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Build merged smart-home SFT v4 dataset.

Sources (per Iter 11.1 brief):
  A. existing v3 (sh_train_v3.json, 12134) — keep as base, BUT we will
     re-derive the noisy upsample factor: keep v2 portion (8651-noisy=5990 SH
     × 5 + 2500 xlam-irr + 151 hermes IoT) + Whisper-noisy (1161) × 5 instead
     of × 3 (so 4644 → 5805 noisy rows).
  B. xLAM-60k subset (via public mirror minpeter/xlam-function-calling-60k-parsed,
     CC-BY-4.0 origin) filtered to smart-home-relevant tools. Cap 3000.
  C. Expanded Hermes (func_calling_singleturn 1.89k) — relax IoT filter to
     audio/security/automation/scheduling. Cap 800.
  D. Twin-confusion contrastive pairs from SH base. Aim 600-1000 pairs.
  E. Numeric-argument focus: paraphrased numeric variants. Cap 2000.

Final target: 22-26k rows.

Run locally (CPU only — no model loads).
Output: data/sh_train_v4.json + push to lifeart/smart-home-sft-v2/sh_train_v4.json.
"""
import copy
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
TRAIN_V3_LOCAL = ROOT / "data" / "sh_train_v3.json"  # may not exist locally
TRAIN_NOISY_LOCAL = ROOT / "data" / "sh_train_noisy.json"
TRAIN_BASE_LOCAL = ROOT / "data" / "sh_train.json"
OUT_PATH = ROOT / "data" / "sh_train_v4.json"
DATA_REPO = "lifeart/smart-home-sft-v2"

PROMPT_TMPL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n{tools}\n\n\n"
    "USER: {query}\n\n\nASSISTANT: <functioncall> "
)

NONE_GOLD = '{"name":"none","arguments":{}}'

# Iter 11 spec: relaxed substring filter (smart-home + audio/security/automation/scheduling)
SH_KW = re.compile(
    r"\b(light|lamp|bulb|brightness|dim|"
    r"temperature|thermostat|hvac|climate|heater|cooler|radiator|"
    r"fan|ventilation|"
    r"music|media|tv|television|speaker|sound|volume|podcast|radio|"
    r"camera|doorbell|security|alarm|surveillance|motion|sensor|detector|"
    r"door|window|garage|lock|unlock|gate|"
    r"timer|reminder|schedule|cron|cronjob|"
    r"outlet|plug|switch|"
    r"scene|automation|routine|"
    r"sprinkler|irrigation|garden|plant|water_plant|"
    r"kitchen|coffee|microwave|oven|stove|fridge|refrigerator|"
    r"cleaning|vacuum|mop|robot_cleaner|"
    r"blind|curtain|shade|shutter|skylight|"
    r"smart_home|home_automation|iot|appliance|"
    r"audio|voice|playback|"
    r"pool|hot_tub|spa)",
    re.IGNORECASE,
)

# Twin-verb mapping for contrastive paraphrase
VERB_FLIPS_NAME = {
    "turn_on_": "turn_off_",
    "turn_off_": "turn_on_",
    "lock_": "unlock_",
    "unlock_": "lock_",
    "open_": "close_",
    "close_": "open_",
    "start_": "stop_",
    "stop_": "start_",
}

QUERY_FLIPS_EN = [
    # case-insensitive: (pattern, replacement) — keep capitalization style cheap.
    # ORDER MATTERS: more-specific patterns first.
    (re.compile(r"\bturn on\b", re.I), "turn off"),
    (re.compile(r"\bturn off\b", re.I), "turn on"),
    (re.compile(r"\bswitch on\b", re.I), "switch off"),
    (re.compile(r"\bswitch off\b", re.I), "switch on"),
    (re.compile(r"\bpower on\b", re.I), "power off"),
    (re.compile(r"\bpower off\b", re.I), "power on"),
    (re.compile(r"\bturn (.*?) on\b", re.I), r"turn \1 off"),
    (re.compile(r"\bturn (.*?) off\b", re.I), r"turn \1 on"),
    (re.compile(r"\bswitch (.*?) on\b", re.I), r"switch \1 off"),
    (re.compile(r"\bswitch (.*?) off\b", re.I), r"switch \1 on"),
    (re.compile(r"\bunlock\b", re.I), "lock"),
    (re.compile(r"\block\b", re.I), "unlock"),
    (re.compile(r"\bclose\b", re.I), "open"),
    (re.compile(r"\bopen\b", re.I), "close"),
    (re.compile(r"\bstart\b", re.I), "stop"),
    (re.compile(r"\bstop\b", re.I), "start"),
    (re.compile(r"\benable\b", re.I), "disable"),
    (re.compile(r"\bdisable\b", re.I), "enable"),
    (re.compile(r"\bbegin\b", re.I), "end"),
    (re.compile(r"\bend\b", re.I), "begin"),
    (re.compile(r"\bactivate\b", re.I), "deactivate"),
    (re.compile(r"\bdeactivate\b", re.I), "activate"),
    (re.compile(r"\barm\b", re.I), "disarm"),
    (re.compile(r"\bdisarm\b", re.I), "arm"),
    (re.compile(r"\bshut\b", re.I), "open"),
    (re.compile(r"\bshut off\b", re.I), "turn on"),
    (re.compile(r"\bkill\b", re.I), "turn on"),
    (re.compile(r"\braise\b", re.I), "lower"),
    (re.compile(r"\blower\b", re.I), "raise"),
    (re.compile(r"\bclosing\b", re.I), "opening"),
    (re.compile(r"\bopening\b", re.I), "closing"),
]


def load_v3() -> list[dict]:
    if TRAIN_V3_LOCAL.exists():
        print(f"[A] using local {TRAIN_V3_LOCAL}")
        with TRAIN_V3_LOCAL.open() as f:
            return json.load(f)
    print(f"[A] fetching {DATA_REPO}/sh_train_v3.json")
    p = hf_hub_download(repo_id=DATA_REPO, filename="sh_train_v3.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def load_noisy() -> list[dict]:
    if TRAIN_NOISY_LOCAL.exists():
        print(f"[noisy] using local {TRAIN_NOISY_LOCAL}")
        with TRAIN_NOISY_LOCAL.open() as f:
            return json.load(f)
    print(f"[noisy] fetching {DATA_REPO}/sh_train_noisy.json")
    p = hf_hub_download(repo_id=DATA_REPO, filename="sh_train_noisy.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def load_base() -> list[dict]:
    if TRAIN_BASE_LOCAL.exists():
        with TRAIN_BASE_LOCAL.open() as f:
            return json.load(f)
    print(f"[base] fetching {DATA_REPO}/sh_train.json")
    p = hf_hub_download(repo_id=DATA_REPO, filename="sh_train.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def adapt_xlam(cap: int) -> list[dict]:
    print(f"[B] loading minpeter/xlam-function-calling-60k-parsed …")
    ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    print(f"[B] {len(ds)} rows total")
    out: list[dict] = []
    skipped_no_match = 0
    skipped_parse = 0
    skipped_no_user = 0
    for row in ds:
        try:
            tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        except Exception:
            skipped_parse += 1
            continue
        # Extract names from xlam-parsed format: [{"type":"function","function":{"name":...}}]
        names = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function", t)
                n = fn.get("name") if isinstance(fn, dict) else None
                if n:
                    names.append(n)
        if not names:
            skipped_parse += 1
            continue
        haystack = " ".join(names) + " " + (row.get("tools", "")[:1000] if isinstance(row.get("tools"), str) else "")
        if not SH_KW.search(haystack):
            skipped_no_match += 1
            continue
        # messages = [{role:user,content},{role:assistant,tool_calls:[...]}]
        msgs = row["messages"]
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                skipped_parse += 1
                continue
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not user_msg or not asst_msg:
            skipped_no_user += 1
            continue
        user_text = user_msg.get("content") or ""
        tool_calls = asst_msg.get("tool_calls") or []
        if not tool_calls:
            skipped_no_user += 1
            continue
        first = tool_calls[0]
        fn = first.get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments")
        if not name:
            skipped_parse += 1
            continue
        # Args may be JSON-string or dict
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
        else:
            args = args_raw or {}
        gold = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        tool_block = json.dumps(names, indent=2)
        prompt = PROMPT_TMPL.format(tools=tool_block, query=user_text.strip())
        out.append({
            "prompt": prompt,
            "gold": gold,
            "gold_name": name,
            "domain": "xlam_sh",
        })
        if len(out) >= cap:
            break
    print(f"[B] adapted {len(out)} xLAM rows "
          f"(skipped: no_match={skipped_no_match} parse={skipped_parse} no_user={skipped_no_user})")
    return out


def _hermes_extract_first_call(gpt_value: str) -> dict | None:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gpt_value, re.S)
    if not m:
        m = re.search(r"<tool_call>\s*(\{.*\})", gpt_value, re.S)
        if not m:
            return None
    blob = m.group(1).strip()
    for cut in range(len(blob), 0, -1):
        try:
            obj = json.loads(blob[:cut])
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except Exception:
            continue
    return None


def adapt_hermes(cap: int) -> list[dict]:
    print(f"[C] loading hermes-function-calling-v1 (singleturn) with relaxed filter …")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")
    print(f"[C] {len(ds)} rows total")
    out: list[dict] = []
    skipped_no_iot = 0
    skipped_no_call = 0
    skipped_no_user = 0
    for row in ds:
        cat = (row.get("category") or "") + " " + (row.get("subcategory") or "")
        tools_str = row.get("tools") or ""
        haystack = cat + " " + tools_str[:2500]
        if not SH_KW.search(haystack):
            skipped_no_iot += 1
            continue
        convs = row.get("conversations") or []
        user = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if not user or not gpt:
            skipped_no_user += 1
            continue
        call = _hermes_extract_first_call(gpt)
        if not call:
            skipped_no_call += 1
            continue
        try:
            tools = json.loads(tools_str)
        except Exception:
            skipped_no_call += 1
            continue
        names = []
        for t in tools:
            n = None
            if isinstance(t, dict):
                fn = t.get("function") if "function" in t and isinstance(t["function"], dict) else t
                n = fn.get("name") if isinstance(fn, dict) else None
            if n:
                names.append(n)
        if not names:
            skipped_no_call += 1
            continue
        u = user.strip()
        if len(u) > 800:
            u = u[:800].rsplit(" ", 1)[0] + " …"
        prompt = PROMPT_TMPL.format(tools=json.dumps(names, indent=2), query=u)
        gold = json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False)
        out.append({
            "prompt": prompt,
            "gold": gold,
            "gold_name": call["name"],
            "domain": "iot_aux_v4",
        })
        if len(out) >= cap:
            break
    print(f"[C] adapted {len(out)} hermes rows "
          f"(skipped: no_iot={skipped_no_iot} no_call={skipped_no_call} no_user={skipped_no_user})")
    return out


def flip_query(q: str) -> str | None:
    """Apply twin-verb flip to a query string. Returns None if no flip applies.

    Apply only ONE flip per query (the first matching pattern) to avoid
    double-flipping "turn on" → "turn off" + "on" → "off" double-pass.
    """
    for pat, repl in QUERY_FLIPS_EN:
        if pat.search(q):
            return pat.sub(repl, q, count=1)
    return None


def flip_name(name: str) -> str | None:
    for prefix, opp in VERB_FLIPS_NAME.items():
        if name.startswith(prefix):
            return opp + name[len(prefix):]
    return None


def gen_twin_pairs(sh_base: list[dict], tool_registry: dict, cap: int) -> list[dict]:
    """For each SH item with twin-prone gold name, emit a flipped contrastive item.

    Constraint: the FLIPPED gold_name must exist in tool_registry, otherwise we'd
    teach the model a hallucinated name.
    """
    out: list[dict] = []
    valid_names = set(tool_registry.keys())
    seen_pairs: set[tuple[str, str]] = set()
    for row in sh_base:
        gold_name = row.get("gold_name", "")
        flipped_name = flip_name(gold_name)
        if not flipped_name or flipped_name not in valid_names:
            continue
        try:
            gold = json.loads(row["gold"])
        except Exception:
            continue
        # Pull the USER segment
        m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", row["prompt"], re.S)
        if not m:
            continue
        user_q = m.group(1).strip()
        new_q = flip_query(user_q)
        if not new_q or new_q == user_q:
            continue
        # Dedupe key
        key = (flipped_name, new_q[:120])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        new_gold = {"name": flipped_name, "arguments": gold.get("arguments", {})}
        new_gold_str = json.dumps(new_gold, ensure_ascii=False)
        # Replace USER segment in original prompt template
        new_prompt = re.sub(
            r"(USER:)\s*.*?(\s*\n\n\nASSISTANT:)",
            lambda mm: f"{mm.group(1)} {new_q}{mm.group(2)}",
            row["prompt"],
            count=1,
            flags=re.S,
        )
        out.append({
            "prompt": new_prompt,
            "gold": new_gold_str,
            "gold_name": flipped_name,
            "domain": row.get("domain", "twin") + "_twin",
        })
        if len(out) >= cap:
            break
    print(f"[D] twin pairs emitted: {len(out)} (cap={cap}, unique={len(seen_pairs)})")
    return out


# Numeric variants —
def _word_int(n: int) -> str:
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
        21: "twenty-one", 22: "twenty-two", 23: "twenty-three", 24: "twenty-four",
        25: "twenty-five", 30: "thirty", 40: "forty", 50: "fifty",
        60: "sixty", 70: "seventy", 75: "seventy-five", 80: "eighty", 90: "ninety", 100: "one hundred",
    }
    return words.get(n, str(n))


def gen_numeric_variants(sh_base: list[dict], cap: int) -> list[dict]:
    """For SH items whose gold args contain integers, emit 3-5 variants with
    perturbed values (matching also in the user-query if the original number
    appears verbatim there).
    """
    out: list[dict] = []
    rng = random.Random(7)
    for row in sh_base:
        try:
            gold = json.loads(row["gold"])
        except Exception:
            continue
        args = gold.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        numeric_keys = [(k, v) for k, v in args.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numeric_keys:
            continue
        m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", row["prompt"], re.S)
        if not m:
            continue
        user_q = m.group(1).strip()
        # Pick first numeric key for perturbation
        k, v = numeric_keys[0]
        # Only attempt if v is integer (or float looks integral) and is in [0, 100]
        v_int = int(v) if isinstance(v, (int, float)) else None
        if v_int is None or not isinstance(v, int) or v_int < 0 or v_int > 100:
            # Allow floats that are int-valued
            if isinstance(v, float) and v.is_integer() and 0 <= v <= 100:
                v_int = int(v)
            else:
                continue
        # Pick 3 perturbation deltas + 1 word-form variant
        deltas = sorted(set([-3, -2, +2, +3])) if v_int >= 5 else [+1, +2, +3, +5]
        variants = []
        for d in deltas[:3]:
            new_v = max(0, min(100, v_int + d))
            if new_v == v_int:
                continue
            variants.append(new_v)
        # word form
        word = _word_int(v_int)
        if word != str(v_int):
            variants.append((v_int, "word"))
        # cap 4 per source
        rng.shuffle(variants)
        variants = variants[:4]
        for variant in variants:
            if isinstance(variant, tuple) and variant[1] == "word":
                # Replace the numeric token in the user query with its word form, keep gold
                new_user = re.sub(rf"\b{v_int}\b", word, user_q, count=1)
                if new_user == user_q:
                    continue
                new_gold_args = dict(args)
                new_prompt = row["prompt"].replace(user_q, new_user, 1)
                out.append({
                    "prompt": new_prompt,
                    "gold": json.dumps({"name": gold["name"], "arguments": new_gold_args}, ensure_ascii=False),
                    "gold_name": gold["name"],
                    "domain": row.get("domain", "num") + "_num",
                })
            else:
                new_v = variant
                # Replace the numeric token in user query AND set new gold value
                new_user = re.sub(rf"\b{v_int}\b", str(new_v), user_q, count=1)
                if new_user == user_q:
                    continue
                new_gold_args = dict(args)
                new_gold_args[k] = new_v
                new_prompt = row["prompt"].replace(user_q, new_user, 1)
                out.append({
                    "prompt": new_prompt,
                    "gold": json.dumps({"name": gold["name"], "arguments": new_gold_args}, ensure_ascii=False),
                    "gold_name": gold["name"],
                    "domain": row.get("domain", "num") + "_num",
                })
            if len(out) >= cap:
                break
        if len(out) >= cap:
            break
    print(f"[E] numeric variants emitted: {len(out)} (cap={cap})")
    return out


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Dedupe by (gold_name, user_query). One copy per unique (name, query)."""
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for r in rows:
        m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", r["prompt"], re.S)
        user_q = (m.group(1).strip() if m else r["prompt"][:80]).lower()[:200]
        key = (r.get("gold_name", ""), user_q)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def dedupe_within(rows: list[dict]) -> list[dict]:
    """Dedupe within a single source bucket (used for additions like xlam)."""
    return dedupe_rows(rows)


def main() -> None:
    random.seed(42)

    # Load v3 (base — it already has v2 mix + noisy x3)
    v3 = load_v3()
    print(f"[A] v3 base: {len(v3)} rows")
    v3_counts = Counter(r["domain"] for r in v3)
    print(f"[A]   domain counts: {dict(v3_counts)}")

    # Step F: Stronger Whisper-noise loop — we want noisy x5 instead of noisy x3.
    # Split v3 into: noisy rows (already in v3 x3) + non-noisy.
    noisy_rows_in_v3 = [r for r in v3 if r["domain"].endswith("_noisy")]
    nonnoisy_v3 = [r for r in v3 if not r["domain"].endswith("_noisy")]
    print(f"[F] v3 noisy in-place: {len(noisy_rows_in_v3)} (x3 of ~{len(noisy_rows_in_v3)//3})")
    # Get unique noisy rows (deduped by (gold_name, user_query))
    unique_noisy = dedupe_rows(noisy_rows_in_v3)
    print(f"[F] unique noisy rows: {len(unique_noisy)}")
    # Re-upsample noisy by 5 (instead of x3)
    NOISY_UPSAMPLE = 5
    boosted_noisy = unique_noisy * NOISY_UPSAMPLE
    print(f"[F] boosted noisy x{NOISY_UPSAMPLE}: {len(boosted_noisy)}")

    # Base for everything we add: the v3 non-noisy items + boosted noisy
    v4_base = nonnoisy_v3 + boosted_noisy
    print(f"[merge-base] v4 base (non-noisy + noisy×5): {len(v4_base)}")

    # Load SH base for twin / numeric variants
    sh_base = load_base()
    print(f"[base] SH base: {len(sh_base)} items")

    # Load tool registry for twin name validation
    with (ROOT / "data" / "tool_registry.json").open() as f:
        registry_obj = json.load(f)
    print(f"[registry] {len(registry_obj)} entries")

    # B: xLAM-60k filtered subset
    CAP_XLAM = 3000
    xlam_rows = adapt_xlam(CAP_XLAM)

    # C: relaxed Hermes
    CAP_HERMES = 800
    hermes_rows = adapt_hermes(CAP_HERMES)

    # D: twin contrastive pairs
    CAP_TWIN = 1000
    twin_rows = gen_twin_pairs(sh_base, registry_obj, CAP_TWIN)

    # E: numeric variants
    CAP_NUMERIC = 2000
    numeric_rows = gen_numeric_variants(sh_base, CAP_NUMERIC)

    # Dedupe within each addition source (not against base — they have unique
    # (name, query) pairs anyway since SH base uses different verbs/IDs).
    xlam_rows = dedupe_within(xlam_rows)
    hermes_rows = dedupe_within(hermes_rows)
    twin_rows = dedupe_within(twin_rows)
    # Numeric variants: dedupe AGAINST the SH base to avoid teaching duplicates,
    # but preserve their multiplicity (each base item emits multiple unique
    # (variant_value) queries already).
    numeric_rows = dedupe_within(numeric_rows)

    additions = xlam_rows + hermes_rows + twin_rows + numeric_rows
    print(f"[merge] additions (deduped within): xlam={len(xlam_rows)} hermes={len(hermes_rows)} twin={len(twin_rows)} numeric={len(numeric_rows)}")

    # Now also numeric variants might collide with SH base — drop those collisions
    # by keeping the base copy. Build a set of base (name, user_q) keys.
    base_keys: set[tuple[str, str]] = set()
    for r in v4_base:
        m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", r["prompt"], re.S)
        user_q = (m.group(1).strip() if m else r["prompt"][:80]).lower()[:200]
        base_keys.add((r.get("gold_name", ""), user_q))
    cleaned_additions: list[dict] = []
    drop_collisions = 0
    for r in additions:
        m = re.search(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", r["prompt"], re.S)
        user_q = (m.group(1).strip() if m else r["prompt"][:80]).lower()[:200]
        key = (r.get("gold_name", ""), user_q)
        if key in base_keys:
            drop_collisions += 1
            continue
        cleaned_additions.append(r)
    print(f"[merge] addition-vs-base collisions dropped: {drop_collisions}")

    merged = v4_base + cleaned_additions
    print(f"[merge] total v4 rows: {len(merged)} (base {len(v4_base)} + additions {len(cleaned_additions)})")

    random.shuffle(merged)

    counts = Counter(r["domain"] for r in merged)
    print("\n=== final v4 dataset (by domain) ===")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")
    print(f"\n=== TOTAL {len(merged)} ===")

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[save] wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # Push to HF dataset repo
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True:
        api = HfApi()
        print(f"[push] uploading {OUT_PATH} -> {DATA_REPO}/sh_train_v4.json")
        api.upload_file(
            path_or_fileobj=str(OUT_PATH),
            path_in_repo="sh_train_v4.json",
            repo_id=DATA_REPO,
            repo_type="dataset",
            commit_message=f"v4 dataset: {len(merged)} rows (+xlam +hermes-relaxed +twin +numeric, noisy×5)",
        )
        print("[push] done")


if __name__ == "__main__":
    main()
