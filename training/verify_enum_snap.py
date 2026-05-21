# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Iter 38 — verify enum value-snapping on the cached synth-v2 BEST run.

The accuracy report's idea B1: snap predicted argument values to their
`tool_registry.json` enum member ("gym" -> "basement gym", "living_room"
-> "living room"). This re-scores the cached synthesis predictions from
`results/iter32_synth2_BEST.json` with and without `canon.snap_enums`, so
the gain is measured deterministically — no model, no API, no cost.

Run: python training/verify_enum_snap.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bench_common import args_match, parse_gold
from canon import canonicalize_args, snap_enums

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "iter32_synth2_BEST.json"
REGISTRY = ROOT / "data" / "tool_registry.json"


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    rows = json.loads(RESULTS.read_text())["rows"]
    n = len(rows)

    base = snap = regress = 0
    gained: list = []
    for r in rows:
        s = r.get("synth") or {}
        pname, pargs = s.get("name"), s.get("arguments") or {}
        gold = parse_gold(r["gold"])
        gname, gargs = gold["name"], gold["arguments"]
        name_ok = pname is not None and pname == gname

        # baseline: value canonicalization on both sides (the shipped scorer)
        b_ok = name_ok and args_match(
            canonicalize_args(pargs), canonicalize_args(gargs)
        )
        # +enum-snap: snap to registry enums, then canonicalize, both sides
        s_ok = name_ok and args_match(
            canonicalize_args(snap_enums(pname, pargs, registry)),
            canonicalize_args(snap_enums(gname, gargs, registry)),
        )
        base += b_ok
        snap += s_ok
        if s_ok and not b_ok:
            gained.append(r.get("i"))
        if b_ok and not s_ok:
            regress += 1

    print(f"baseline synth exact  : {base}/{n} = {base / n * 100:.2f}%")
    print(f"+enum-snap synth exact: {snap}/{n} = {snap / n * 100:.2f}%"
          f"  ({(snap - base) / n * 100:+.2f} pp)")
    print(f"rows gained: {len(gained)}   regressions: {regress}")
    print(f"gained row ids: {gained}")
    assert regress == 0, "enum-snap caused a regression — investigate"
    assert snap >= base, "enum-snap did not help"
    print("\nVERDICT: enum-snap is a safe, positive post-process.")


if __name__ == "__main__":
    main()
