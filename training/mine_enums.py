"""
Iteration 8.1 — mine enum value sets from training data.

For each function in data/tool_registry.json, for each *string-typed* parameter,
scan data/sh_train_v2.json and data/sh_train.json for all gold-JSON entries
with that function name. Collect the observed values of that parameter.

Rules (per PLAN.md):
- ≥3 distinct training items must have observed the key (else noise)
- frequency floor: keep values that appear ≥2 times in the gold set; drop singletons
- cap enum size at 20 (else drop — open vocab)
- preserve casing as-observed
- multi-word values kept as-is

Output:
- Augment data/tool_registry.json with `"enums": {param: [v1, v2, ...]}` per function.
- Mirror to web/public/eval/tool_registry.json.
- Print a per-function table of newly added enums (top 10).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "data" / "tool_registry.json"
WEB_REGISTRY_PATH = ROOT / "web" / "public" / "eval" / "tool_registry.json"
TRAIN_PATHS = [
    ROOT / "data" / "sh_train_v2.json",
    ROOT / "data" / "sh_train.json",
]

MAX_ENUM = 20
MIN_DISTINCT_ITEMS = 3   # ≥3 distinct training items observing this key
MIN_VALUE_FREQ = 2       # value must appear ≥2 times


def iter_gold_calls(train_paths):
    """Yield (fn_name, args_dict) tuples parsed from gold JSONs."""
    n_items = 0
    n_parse_err = 0
    for p in train_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            n_items += 1
            gold_str = item.get("gold", "")
            # Handle possible multi-call concatenation — split on </functioncall>
            chunks = gold_str.split("</functioncall>")
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Strip leading <functioncall> if present
                if chunk.startswith("<functioncall>"):
                    chunk = chunk[len("<functioncall>"):].strip()
                try:
                    g = json.loads(chunk)
                except Exception:
                    n_parse_err += 1
                    continue
                name = g.get("name")
                args = g.get("arguments")
                if not name or not isinstance(args, dict):
                    continue
                yield name, args
    print(f"[mine] scanned {n_items} items; {n_parse_err} parse errors")


def main():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Build a set of (fn, param_name) for string-typed params
    string_params = {}  # fn -> [param_name, ...]
    for fn, spec in registry.items():
        sps = []
        for p, t in spec.get("params", {}).items():
            if t == "string":
                sps.append(p)
        if sps:
            string_params[fn] = sps

    print(f"[mine] {sum(len(v) for v in string_params.values())} string params across "
          f"{len(string_params)} functions are candidates for enum mining")

    # Counters: (fn, param) -> Counter(value -> count)
    # And track distinct ITEM occurrences (deduped within a single gold)
    value_counters = defaultdict(Counter)
    item_seen = defaultdict(set)  # (fn, param) -> set of value-strings observed (one per item)

    item_idx = 0
    for fn, args in iter_gold_calls(TRAIN_PATHS):
        item_idx += 1
        if fn not in string_params:
            continue
        for p in string_params[fn]:
            if p not in args:
                continue
            v = args[p]
            if not isinstance(v, str):
                continue
            v = v.strip()
            if not v:
                continue
            key = (fn, p)
            value_counters[key][v] += 1
            item_seen[key].add(item_idx)

    # Apply filters
    new_enums = defaultdict(dict)  # fn -> {param: [v1, ...]}
    table_rows = []  # (fn, param, enum_size, sample_str)
    for (fn, p), counter in value_counters.items():
        n_distinct_items = len(item_seen[(fn, p)])
        if n_distinct_items < MIN_DISTINCT_ITEMS:
            continue
        # Filter values by freq floor
        kept = [v for v, c in counter.items() if c >= MIN_VALUE_FREQ]
        if not kept:
            continue
        # Sort by count desc, then by value
        kept.sort(key=lambda v: (-counter[v], v.lower()))
        if len(kept) > MAX_ENUM:
            # Too open-vocab — skip
            continue
        new_enums[fn][p] = kept
        sample = ", ".join(kept[:8])
        if len(kept) > 8:
            sample += ", ..."
        table_rows.append((fn, p, len(kept), sample))

    print(f"[mine] enums added: {sum(len(v) for v in new_enums.values())} params "
          f"across {len(new_enums)} functions")
    print()
    print("function_name  param  enum_size  sample_values")
    print("-" * 100)
    # Sort by function name then param
    table_rows.sort(key=lambda r: (r[0], r[1]))
    for fn, p, sz, sample in table_rows[:30]:
        print(f"  {fn:30s} {p:18s} {sz:3d}   {sample}")
    if len(table_rows) > 30:
        print(f"... ({len(table_rows) - 30} more) ...")

    # Sanity-check: print the top-10 most-enriched functions
    print()
    print("Top 10 most-enriched functions:")
    top = sorted(new_enums.items(), key=lambda kv: -sum(len(v) for v in kv[1].values()))[:10]
    for fn, eparams in top:
        for p, vals in eparams.items():
            print(f"  {fn}.{p}  ({len(vals)}): {vals[:8]}")

    # Specific sanity: how does `room` look across functions?
    print()
    print("Sanity check — `room` enum across all functions:")
    room_count = 0
    for fn, eparams in new_enums.items():
        if "room" in eparams:
            room_count += 1
            print(f"  {fn}.room ({len(eparams['room'])}): {eparams['room']}")
    print(f"  (room enum in {room_count} functions)")

    # Write the registry back with new field `enums`
    for fn, eparams in new_enums.items():
        # Keep existing fields; add 'enums'
        registry[fn]["enums"] = eparams

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    # Mirror
    WEB_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print()
    print(f"[mine] wrote {REGISTRY_PATH}")
    print(f"[mine] wrote {WEB_REGISTRY_PATH}")


if __name__ == "__main__":
    main()
