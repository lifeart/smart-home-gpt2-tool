#!/usr/bin/env python3
"""Iter 10.1 — Audit which gold names are missing from descriptions / registry.

For each unique gold_name in `data/sh_test.json` and `data/sh_train.json`:
- Check presence in `web/public/eval/function_descriptions.json` (retrieval index source)
- Check presence in `data/tool_registry.json` (== `web/public/eval/tool_registry.json`)

Print per-source tables.
"""

import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(rel: str):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def domain_of(item, registry_domains):
    if "domain" in item and item["domain"]:
        return item["domain"]
    return registry_domains.get(item.get("gold_name", ""), "?")


def main():
    test = load_json("data/sh_test.json")
    train = load_json("data/sh_train.json")
    descs = load_json("web/public/eval/function_descriptions.json")
    reg_data = load_json("data/tool_registry.json")
    reg_web = load_json("web/public/eval/tool_registry.json")

    assert reg_data == reg_web, "data/ and web/public/eval/ tool_registry.json differ!"

    desc_names = set(descs.keys())
    reg_names = set(reg_data.keys())

    test_names = Counter()
    test_domains_of = {}
    for it in test:
        n = it.get("gold_name", "")
        test_names[n] += 1
        test_domains_of[n] = it.get("domain", "?")

    train_names = Counter()
    for it in train:
        n = it.get("gold_name", "")
        train_names[n] += 1

    print(f"# Iter 10.1 audit")
    print(f"function_descriptions.json: {len(desc_names)} entries")
    print(f"tool_registry.json:         {len(reg_names)} entries")
    print(f"unique gold names in test (n={len(test)}):   {len(test_names)}")
    print(f"unique gold names in train (n={len(train)}): {len(train_names)}")
    print()

    test_missing_desc = sorted(
        [(n, c, test_domains_of.get(n, "?")) for n, c in test_names.items() if n not in desc_names],
        key=lambda x: (-x[1], x[0]),
    )
    test_missing_reg = sorted(
        [(n, c, test_domains_of.get(n, "?")) for n, c in test_names.items() if n not in reg_names],
        key=lambda x: (-x[1], x[0]),
    )
    train_missing_desc = sorted(
        [(n, c) for n, c in train_names.items() if n not in desc_names],
        key=lambda x: (-x[1], x[0]),
    )
    train_missing_reg = sorted(
        [(n, c) for n, c in train_names.items() if n not in reg_names],
        key=lambda x: (-x[1], x[0]),
    )

    print(f"## TEST: gold names absent from function_descriptions.json: {len(test_missing_desc)}")
    by_dom = defaultdict(list)
    for n, c, d in test_missing_desc:
        by_dom[d].append((n, c))
    for d in sorted(by_dom):
        names = by_dom[d]
        print(f"  [{d}] ({len(names)}): {', '.join(f'{n}×{c}' for n,c in names)}")
    print()

    print(f"## TEST: gold names absent from tool_registry.json: {len(test_missing_reg)}")
    by_dom = defaultdict(list)
    for n, c, d in test_missing_reg:
        by_dom[d].append((n, c))
    for d in sorted(by_dom):
        names = by_dom[d]
        print(f"  [{d}] ({len(names)}): {', '.join(f'{n}×{c}' for n,c in names)}")
    print()

    print(f"## TRAIN: gold names absent from function_descriptions.json: {len(train_missing_desc)}")
    print(f"  {', '.join(f'{n}×{c}' for n,c in train_missing_desc[:40])}")
    if len(train_missing_desc) > 40:
        print(f"  ... +{len(train_missing_desc)-40} more")
    print()

    print(f"## TRAIN: gold names absent from tool_registry.json: {len(train_missing_reg)}")
    print(f"  {', '.join(f'{n}×{c}' for n,c in train_missing_reg[:40])}")
    if len(train_missing_reg) > 40:
        print(f"  ... +{len(train_missing_reg)-40} more")
    print()

    # Registry present but missing in descriptions (function lacks examples for retrieval)
    reg_missing_desc = sorted(reg_names - desc_names)
    print(f"## REGISTRY entries with no description: {len(reg_missing_desc)}")
    if reg_missing_desc:
        print(f"  {', '.join(reg_missing_desc)}")
    print()

    # Description-only entries (in desc but not in registry; not necessarily a bug)
    desc_only = sorted(desc_names - reg_names)
    print(f"## DESCRIPTIONS not in registry: {len(desc_only)}")
    if desc_only:
        print(f"  {', '.join(desc_only)}")
    print()

    # Union: names anywhere known
    all_known = desc_names | reg_names
    test_unknown_anywhere = sorted([(n, c, test_domains_of.get(n, "?")) for n, c in test_names.items() if n not in all_known])
    print(f"## TEST: gold names unknown to BOTH descriptions and registry: {len(test_unknown_anywhere)}")
    for n, c, d in test_unknown_anywhere:
        print(f"  [{d}] {n} ×{c}")
    print()

    # For each test-missing-from-desc name, find sample queries (in train first, then test)
    print(f"## SAMPLE QUERIES for test-missing-from-desc names (for description writing):")
    for n, c, d in test_missing_desc:
        train_examples = [it for it in train if it.get("gold_name") == n][:3]
        test_examples = [it for it in test if it.get("gold_name") == n][:3]
        print(f"\n  ### {n} (domain={d}, test×{c}, train×{train_names.get(n,0)})")
        if train_examples:
            for ex in train_examples:
                prompt = ex.get("prompt", "")
                user = prompt.split("USER:")[-1].split("ASSISTANT:")[0].strip()
                print(f"    [train] {user!r}")
                print(f"            gold={ex.get('gold')!r}")
        elif test_examples:
            for ex in test_examples:
                prompt = ex.get("prompt", "")
                user = prompt.split("USER:")[-1].split("ASSISTANT:")[0].strip()
                print(f"    [test ] {user!r}")
                print(f"            gold={ex.get('gold')!r}")
        else:
            print(f"    (no examples found)")


if __name__ == "__main__":
    main()
