"""Combine 10 sh_*.json files into one SFT corpus.
Multi-tool format: SYSTEM has 3-5 candidate function specs, model must pick correct one.
Split 1200 train / 300 test (held-out).
"""
import json, random
from pathlib import Path

random.seed(42)
DIR = Path(__file__).resolve().parent
FILES = ["sh_lighting.json", "sh_climate.json", "sh_security.json", "sh_media.json", "sh_kitchen.json",
         "sh_garden.json", "sh_blinds.json", "sh_cleaning.json", "sh_timers.json", "sh_sensors.json"]


def to_sft(item):
    """Multi-tool prompt: SYSTEM with all 3-5 function specs, model picks correct."""
    funcs = item["function"]
    if isinstance(funcs, dict):
        funcs = [funcs]
    fn_json = json.dumps(funcs, indent=2)[:1200]
    prompt = (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        f"{fn_json}\n\n\n"
        f"USER: {item['prompt'][:300]}\n\n\nASSISTANT: <functioncall> "
    )
    gold = json.dumps({"name": item["gold_name"], "arguments": item.get("gold_args", {})}, separators=(",", ":"))
    iid = str(item.get("id", ""))
    domain = iid.split("_")[0] if "_" in iid else "misc"
    return {"prompt": prompt, "gold": gold, "gold_name": item["gold_name"], "domain": domain}


def main():
    all_items = []
    for f in FILES:
        p = DIR / f
        if not p.exists():
            print(f"MISSING: {f}"); continue
        arr = json.load(open(p, encoding='utf-8'))
        for it in arr:
            try:
                all_items.append(to_sft(it))
            except Exception as e:
                print(f"  skip {it.get('id','?')}: {e}")
    print(f"loaded {len(all_items)} items")
    random.shuffle(all_items)
    train = all_items[:1200]
    test = all_items[1200:1500]
    (DIR / "sh_train.json").write_text(json.dumps(train, indent=2, ensure_ascii=False))
    (DIR / "sh_test.json").write_text(json.dumps(test, indent=2, ensure_ascii=False))
    print(f"train: {len(train)}, test: {len(test)}")
    by_dom = {}
    for it in train:
        d = it["domain"]
        by_dom[d] = by_dom.get(d, 0) + 1
    print("train distribution:")
    for k, v in sorted(by_dom.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
