"""SFT v2: 1200 multi-tool smart-home items.
Resume from gpt2_ft_final.pt, train 1 epoch with PAD=1024 (multi-tool prompts longer).
"""
import os, sys, json, time, random
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrated_gpt2_torch import GPT2, encode

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

BASE = Path(__file__).resolve().parent.parent / "weights" / "smart_home_v2_base.pt"  # fallback: train from openai-community/gpt2
DATA = Path(__file__).resolve().parent.parent / "data" / "sh_train.json"
OUT = Path(__file__).resolve().parent.parent / "weights" / "smart_home_v2.pt"

PAD = 1024
LR = 1e-5
BATCH = 1
GRAD_ACCUM = 4


def make_batch(samples, indices):
    B = len(indices)
    input_ids = torch.zeros((B, PAD), dtype=torch.long)
    labels = torch.full((B, PAD), -100, dtype=torch.long)
    for i, idx in enumerate(indices):
        s = samples[idx]
        prompt_ids = encode(s["prompt"])
        gold_ids = encode(s["gold"])[:80]
        max_prompt = PAD - len(gold_ids)
        prompt_ids = prompt_ids[-max_prompt:]
        seq = prompt_ids + gold_ids
        T = len(seq)
        v = T - len(gold_ids)
        input_ids[i, :T] = torch.tensor(seq, dtype=torch.long)
        labels[i, v:T] = torch.tensor(gold_ids, dtype=torch.long)
    return input_ids.to(DEVICE), labels.to(DEVICE)


def main():
    print("[smart-home SFT v2 — multi-tool]")
    pairs = json.load(open(DATA, encoding='utf-8'))
    print(f"  loaded {len(pairs)} pairs")
    model = GPT2()
    if BASE.exists():
        print(f"  resume from {BASE.name}")
        model.load_state_dict(torch.load(str(BASE), map_location='cpu'))
    else:
        print(f"  no checkpoint at {BASE} -> training from raw openai-community/gpt2")
        from integrated_gpt2_torch import load_gpt2_torch_weights
        load_gpt2_torch_weights(model)
    model.to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    idx = list(range(len(pairs)))
    random.Random(42).shuffle(idx)
    t0 = time.time()
    step = accum = 0
    running = 0.0
    opt.zero_grad()
    for i in range(0, len(idx), BATCH):
        ii, lb = make_batch(pairs, idx[i:i+BATCH])
        logits, _ = model(ii)
        sl = logits[:, :-1, :].contiguous(); slb = lb[:, 1:].contiguous()
        if not (slb != -100).any(): continue
        _, _, V = sl.shape
        loss = F.cross_entropy(sl.reshape(-1, V), slb.reshape(-1), ignore_index=-100)
        (loss / GRAD_ACCUM).backward()
        running += loss.item(); accum += 1
        if accum == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  step {step}/{len(idx)//(GRAD_ACCUM*BATCH)}  loss={loss.item():.3f}  avg20={running/(accum*20):.3f}  t={time.time()-t0:.0f}s", flush=True)
                running = 0.0
            accum = 0
    torch.save(model.state_dict(), OUT)
    print(f"DONE saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
