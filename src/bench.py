"""Bench smart_home_v2.pt on 300 held-out multi-tool items."""
import os, sys, json, re, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrated_gpt2_torch import GPT2, encode, decode

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

W = Path(__file__).resolve().parent.parent / "weights" / "smart_home_v2.pt"
TEST = Path(__file__).resolve().parent.parent / "data" / "sh_test.json"


def parse_name(text):
    m = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    return m.group(1) if m else None


@torch.no_grad()
def gen(model, prompt, max_new=60):
    ids = encode(prompt)
    ids = ids[-900:] if len(ids) > 900 else list(ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= 1024: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits, _ = model(ii)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0]: break
    return decode(ids[L:]).strip()


def main():
    items = json.load(open(TEST, encoding='utf-8'))
    print(f"[Smart-home v2 bench] {len(items)} held-out multi-tool items")
    model = GPT2()
    model.load_state_dict(torch.load(str(W), map_location='cpu'))
    model.to(DEVICE); model.eval()

    correct = 0; t0 = time.time()
    by_domain = {}
    for i, s in enumerate(items):
        out = gen(model, s["prompt"])
        pred = parse_name(out)
        ok = (pred == s["gold_name"])
        if ok: correct += 1
        d = s.get("domain", "?")
        if d not in by_domain: by_domain[d] = [0, 0]
        by_domain[d][1] += 1
        if ok: by_domain[d][0] += 1
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{len(items)}] acc={correct/(i+1)*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)

    acc = correct / len(items)
    print(f"\n=== Smart-Home v2 — multi-tool held-out ===")
    print(f"  Accuracy: {correct}/{len(items)} = {acc*100:.1f}%")
    print(f"  Time: {time.time()-t0:.0f}s  ({(time.time()-t0)/len(items):.1f}s/query)")
    print(f"  By domain:")
    for d, (c, n) in sorted(by_domain.items()):
        print(f"    {d:<10} {c}/{n} = {c/n*100:.1f}%")
    out = {"acc": acc, "n": len(items), "correct": correct, "by_domain": by_domain}
    Path(__file__).resolve().parent.joinpath("bench_v2_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
