"""
Iteration 8.1 — mine enum value sets from training data.

Two-pass strategy:
  (1) Pool values per param-name *globally* across all functions that declare
      that param as string. Many params (room, door, scene, station, ...) are
      semantically the same across functions, and the test set draws from a
      shared vocabulary — so per-function mining is too narrow.
  (2) For each (function, param), the enum = the global pool restricted to
      string-typed params, capped at MAX_ENUM size.

Filters (per PLAN.md, adapted):
- value must appear >=2 times globally (across all functions × items)
- enum size <=20 (else open-vocab — drop)
- preserve casing as-observed (multi-word kept as-is)

Output:
- Augment data/tool_registry.json with `"enums": {param: [v1, v2, ...]}` per
  function. Mirror to web/public/eval/tool_registry.json.
- Print sample table.
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

MAX_ENUM = 30            # bumped from 20 — closed sets like `room` (21 vals) are
                         # still legitimate enums; only true open-vocab (song,
                         # time at 36/24+ values) should be dropped.
MIN_VALUE_FREQ = 2       # value must appear ≥2 times globally


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
    string_param_names = set()  # set of all param names that are string-typed somewhere
    for fn, spec in registry.items():
        sps = []
        for p, t in spec.get("params", {}).items():
            if t == "string":
                sps.append(p)
                string_param_names.add(p)
        if sps:
            string_params[fn] = sps

    print(f"[mine] {sum(len(v) for v in string_params.values())} string params across "
          f"{len(string_params)} functions are candidates for enum mining")
    print(f"[mine] {len(string_param_names)} distinct string-param NAMES (global pool keys)")

    # PASS 1: Pool values per param-NAME globally (across all functions).
    # A value is admissible for param-name P if it was observed as the value of
    # P in ANY function call. This captures shared vocabularies (room, door,
    # scene, station, mode, ...). It also captures cases where a less-frequent
    # variant of room ("attic", "porch") appears mostly in some functions but
    # is still valid for any function that takes a room.
    global_pool = defaultdict(Counter)  # param_name -> Counter(value -> count)
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
            global_pool[p][v] += 1

    # Apply frequency + size filters globally
    global_enums = {}  # param_name -> [v1, ...]
    for p, counter in global_pool.items():
        kept = [v for v, c in counter.items() if c >= MIN_VALUE_FREQ]
        if not kept:
            continue
        kept.sort(key=lambda v: (-counter[v], v.lower()))
        if len(kept) > MAX_ENUM:
            # Too open-vocab globally — drop
            continue
        global_enums[p] = kept

    print()
    print(f"[mine] {len(global_enums)} param-names pass global filter (pool size <= {MAX_ENUM})")
    print()
    print("global param-name pools (top 20 by enum size):")
    print("-" * 100)
    top_pools = sorted(global_enums.items(), key=lambda kv: -len(kv[1]))[:20]
    for p, vals in top_pools:
        sample = ", ".join(vals[:8])
        if len(vals) > 8:
            sample += ", ..."
        print(f"  {p:18s} {len(vals):3d}   {sample}")
    print()

    # Dropped pools (>20 entries past freq filter — open-vocab)
    dropped = []
    for p, counter in global_pool.items():
        kept = [v for v, c in counter.items() if c >= MIN_VALUE_FREQ]
        if len(kept) > MAX_ENUM:
            dropped.append((p, len(kept)))
    if dropped:
        print(f"[mine] dropped {len(dropped)} param-names as open-vocab (>{MAX_ENUM} distinct values):")
        for p, n in sorted(dropped, key=lambda x: -x[1])[:10]:
            print(f"  {p}: {n} distinct values (open-vocab)")
        print()

    # PASS 2: For each (function, param), assign the global pool enum.
    new_enums = defaultdict(dict)  # fn -> {param: [v1, ...]}
    table_rows = []  # (fn, param, enum_size, sample_str)
    for fn, params in string_params.items():
        for p in params:
            if p in global_enums:
                new_enums[fn][p] = global_enums[p]
                sample = ", ".join(global_enums[p][:8])
                if len(global_enums[p]) > 8:
                    sample += ", ..."
                table_rows.append((fn, p, len(global_enums[p]), sample))

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
