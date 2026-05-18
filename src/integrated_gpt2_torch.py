import os
"""ПРАВИЛЬНАЯ интеграция GPT-2 + adapter в ОДНОМ torch.nn.Module.

Принцип: один forward, один backward, без кэша, без autograd-numpy границы.
- GPT-2 — torch frozen (requires_grad=False)
- Adapter — torch trainable, side-channel из layer 6
- forward(input_ids) → (text_logits, adapter_outputs)
- Gradient идёт ТОЛЬКО через adapter (GPT-2 заморожена) — это в одном графе

Веса GPT-2 загружаются из safetensors прямо в torch.Tensor.
Adapter веса можно инициализировать с нуля или загружать из существующего .npz.
"""
import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

GPT2_DIR = Path(os.environ.get("BASE_GPT2", str(Path(__file__).resolve().parent.parent / "weights" / "base_gpt2")))


# ============================================================
# GPT-2 в torch (как nn.Module). Frozen базовая модель.
# ============================================================
class GPT2Block(nn.Module):
    def __init__(self, d_model=768, n_head=12):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        # GPT-2 packs Q,K,V into one matrix: (d, 3*d)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.mlp_fc = nn.Linear(d_model, 4 * d_model)
        self.mlp_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        # x: (B, T, d)
        B, T, _ = x.shape
        # Self-attention
        h = self.ln_1(x)
        qkv = self.c_attn(h)                                # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.c_proj(ctx)
        # MLP (GeLU as in GPT-2)
        h = self.ln_2(x)
        h = self.mlp_fc(h)
        h = 0.5 * h * (1 + torch.tanh(math.sqrt(2/math.pi) * (h + 0.044715 * h**3)))
        x = x + self.mlp_proj(h)
        return x


class GPT2(nn.Module):
    def __init__(self, vocab=50257, d_model=768, n_head=12, n_layer=12, max_pos=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([GPT2Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, hidden_at=None):
        """input_ids: (B, T). Returns (logits (B,T,V), [hidden states by layer if hidden_at]).
        hidden_at: list of layer indices to capture (e.g. [6])."""
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        captured = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if hidden_at is not None and i in hidden_at:
                captured[i] = x   # SAME GRAPH — это просто ссылка на тензор
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T   # tied embeddings
        return logits, captured


def _ensure_base_gpt2(gpt2_dir: Path):
    """Auto-download openai-community/gpt2 to gpt2_dir if missing."""
    if (gpt2_dir / "model.safetensors").exists() and (gpt2_dir / "tokenizer.json").exists():
        return
    print(f"[load_gpt2_torch_weights] base GPT-2 not found at {gpt2_dir}, downloading from HF...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="openai-community/gpt2", local_dir=str(gpt2_dir),
                          allow_patterns=["*.json", "*.safetensors"])
        print(f"[load_gpt2_torch_weights] downloaded to {gpt2_dir}")
    except Exception as e:
        raise RuntimeError(
            f"Base GPT-2 weights missing at {gpt2_dir} and auto-download failed: {e}\n"
            f"Manual fix: pip install huggingface-hub && "
            f"huggingface-cli download openai-community/gpt2 --local-dir {gpt2_dir}"
        ) from e


def load_gpt2_torch_weights(model: GPT2, gpt2_dir=GPT2_DIR):
    """Load HF-format weights into our torch GPT-2. Auto-downloads base if missing."""
    _ensure_base_gpt2(gpt2_dir)
    from safetensors import safe_open
    w = {}
    with safe_open(gpt2_dir / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k)
    sd = model.state_dict()
    # GPT-2 HF mapping
    sd['wte.weight'].copy_(w['wte.weight'])
    sd['wpe.weight'].copy_(w['wpe.weight'])
    sd['ln_f.weight'].copy_(w['ln_f.weight'])
    sd['ln_f.bias'].copy_(w['ln_f.bias'])
    for L in range(len(model.layers)):
        p = f"h.{L}."
        # HF c_attn.weight is (d, 3d) but in torch.Linear it's (out, in)
        # HF stores Conv1D-style weight (in, out), so we need transpose
        sd[f'layers.{L}.ln_1.weight'].copy_(w[p+'ln_1.weight'])
        sd[f'layers.{L}.ln_1.bias'].copy_(w[p+'ln_1.bias'])
        sd[f'layers.{L}.ln_2.weight'].copy_(w[p+'ln_2.weight'])
        sd[f'layers.{L}.ln_2.bias'].copy_(w[p+'ln_2.bias'])
        # Conv1D: weight (in, out) → torch.Linear (out, in) — TRANSPOSE
        sd[f'layers.{L}.c_attn.weight'].copy_(w[p+'attn.c_attn.weight'].T)
        sd[f'layers.{L}.c_attn.bias'].copy_(w[p+'attn.c_attn.bias'])
        sd[f'layers.{L}.c_proj.weight'].copy_(w[p+'attn.c_proj.weight'].T)
        sd[f'layers.{L}.c_proj.bias'].copy_(w[p+'attn.c_proj.bias'])
        sd[f'layers.{L}.mlp_fc.weight'].copy_(w[p+'mlp.c_fc.weight'].T)
        sd[f'layers.{L}.mlp_fc.bias'].copy_(w[p+'mlp.c_fc.bias'])
        sd[f'layers.{L}.mlp_proj.weight'].copy_(w[p+'mlp.c_proj.weight'].T)
        sd[f'layers.{L}.mlp_proj.bias'].copy_(w[p+'mlp.c_proj.bias'])
    return model


# ============================================================
# Adapter (torch) — side-channel из middle layer
# ============================================================
class AdapterV5(nn.Module):
    def __init__(self, d_hidden=768, d_bottle=192, d_out=96,
                 n_action=53, n_scope=6, n_format=7, n_spec=4, n_target=16):
        super().__init__()
        self.W1 = nn.Linear(d_hidden, d_bottle)
        self.ln1 = nn.LayerNorm(d_bottle)
        self.W2 = nn.Linear(d_bottle, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        # Heads
        self.h_action = nn.Linear(d_out, n_action)
        self.h_scope  = nn.Linear(d_out, n_scope)
        self.h_format = nn.Linear(d_out, n_format)
        self.h_spec   = nn.Linear(d_out, n_spec)
        self.h_target = nn.Linear(d_out, n_target)
        self.ptr_s    = nn.Linear(d_out, 1)
        self.ptr_e    = nn.Linear(d_out, 1)
        self.gate     = nn.Linear(d_out, 1)

    def forward(self, hidden, mask):
        """hidden: (B, T, d_hidden), mask: (B, T) float 0/1."""
        h1 = F.gelu(self.ln1(self.W1(hidden)))               # (B, T, d_bottle)
        h2 = F.gelu(self.ln2(self.W2(h1)))                    # (B, T, d_out)
        # Mean-pool с маской
        m = mask.unsqueeze(-1)                                # (B, T, 1)
        pooled = (h2 * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        # Heads
        out = {
            'action': self.h_action(pooled),
            'scope':  self.h_scope(pooled),
            'format': self.h_format(pooled),
            'spec':   self.h_spec(pooled),
            'target': self.h_target(pooled),
        }
        ps = self.ptr_s(h2).squeeze(-1)                       # (B, T)
        pe = self.ptr_e(h2).squeeze(-1)
        ps = ps.masked_fill(mask < 0.5, -1e9)
        pe = pe.masked_fill(mask < 0.5, -1e9)
        out['ptr_s'] = ps
        out['ptr_e'] = pe
        out['gate']  = torch.sigmoid(self.gate(pooled).squeeze(-1))
        return out


# ============================================================
# IntegratedGPT2 — ОДИН nn.Module: GPT-2 frozen + adapter в side-channel
# ============================================================
class IntegratedGPT2(nn.Module):
    def __init__(self, adapter_layer=6):
        super().__init__()
        self.gpt = GPT2()
        self.adapter = AdapterV5()
        self.adapter_layer = adapter_layer

    def freeze_gpt(self):
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.gpt.eval()  # отключить dropout etc

    def forward(self, input_ids, mask):
        """input_ids: (B, T), mask: (B, T) float 0/1.
        Возвращает (gpt_logits, adapter_out) — оба в ОДНОМ графе."""
        gpt_logits, captured = self.gpt(input_ids, hidden_at=[self.adapter_layer])
        h = captured[self.adapter_layer]   # (B, T, 768) — тот же tensor, не копия
        adapter_out = self.adapter(h, mask)
        return gpt_logits, adapter_out


# ============================================================
# Загрузка adapter весов из .npz (autograd numpy → torch)
# ============================================================
def load_adapter_npz(adapter: AdapterV5, npz_path):
    w = dict(np.load(npz_path))
    sd = adapter.state_dict()
    # Маппинг npz keys → torch keys
    # autograd np: W1 (768, 192), torch.Linear (192, 768) — TRANSPOSE
    sd['W1.weight'].copy_(torch.from_numpy(w['W1'].T.astype(np.float32)))
    sd['W1.bias'].copy_(torch.from_numpy(w['b1'].astype(np.float32)))
    sd['ln1.weight'].copy_(torch.from_numpy(w['ln1_g'].astype(np.float32)))
    sd['ln1.bias'].copy_(torch.from_numpy(w['ln1_b'].astype(np.float32)))
    sd['W2.weight'].copy_(torch.from_numpy(w['W2'].T.astype(np.float32)))
    sd['W2.bias'].copy_(torch.from_numpy(w['b2'].astype(np.float32)))
    sd['ln2.weight'].copy_(torch.from_numpy(w['ln2_g'].astype(np.float32)))
    sd['ln2.bias'].copy_(torch.from_numpy(w['ln2_b'].astype(np.float32)))
    # Heads — тоже transposed
    head_map = [('action', 'h_action'), ('scope', 'h_scope'),
                ('format', 'h_format'), ('specificity', 'h_spec'),
                ('target_kind', 'h_target')]
    for npz_name, torch_name in head_map:
        sd[f'{torch_name}.weight'].copy_(torch.from_numpy(w[f'h_{npz_name}_W'].T.astype(np.float32)))
        sd[f'{torch_name}.bias'].copy_(torch.from_numpy(w[f'h_{npz_name}_b'].astype(np.float32)))
    # ptr — (96, 1) → torch.Linear(96 → 1) so weight shape (1, 96), TRANSPOSE
    sd['ptr_s.weight'].copy_(torch.from_numpy(w['ptr_s_W'].T.astype(np.float32)))
    sd['ptr_s.bias'].copy_(torch.zeros(1))
    sd['ptr_e.weight'].copy_(torch.from_numpy(w['ptr_e_W'].T.astype(np.float32)))
    sd['ptr_e.bias'].copy_(torch.zeros(1))
    # gate
    sd['gate.weight'].copy_(torch.from_numpy(w['gate_W'].T.astype(np.float32)))
    sd['gate.bias'].copy_(torch.from_numpy(w['gate_b'].astype(np.float32)))


# ============================================================
# Tokenizer (через tokenizers HF — без изменений)
# ============================================================
from tokenizers import Tokenizer as HFTokenizer
_TOK = None
def get_tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = HFTokenizer.from_file(str(GPT2_DIR / "tokenizer.json"))
    return _TOK


def encode(text):
    return get_tokenizer().encode(text).ids


def decode(ids):
    return get_tokenizer().decode([int(x) for x in ids])


# ============================================================
# Sanity test
# ============================================================
if __name__ == "__main__":
    print("Building IntegratedGPT2 (torch)...")
    model = IntegratedGPT2(adapter_layer=6)
    print(f"  GPT-2 params: {sum(p.numel() for p in model.gpt.parameters()):,}")
    print(f"  adapter params: {sum(p.numel() for p in model.adapter.parameters()):,}")

    print("\nLoading GPT-2 weights from safetensors...")
    load_gpt2_torch_weights(model.gpt)
    print("  done")

    print("\nLoading adapter weights from npz...")
    load_adapter_npz(model.adapter, "weights/adapter_v89_mt.npz")
    print("  done")

    print("\nFreezing GPT-2...")
    model.freeze_gpt()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,} (только adapter)")

    print("\nForward test 'read src/auth.py':")
    ids = encode("read src/auth.py")[:80]
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones((1, len(ids)), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        gpt_logits, adapter_out = model(input_ids, mask)
    print(f"  gpt_logits: {tuple(gpt_logits.shape)}")
    print(f"  adapter action logits: {tuple(adapter_out['action'].shape)}")
    print(f"  adapter gate: {float(adapter_out['gate'].item()):.3f}")

    sys.path.insert(0, "code")
    from modes_spec_v5 import ACTIONS
    pred_action = ACTIONS[int(adapter_out['action'].argmax(dim=1).item())]
    print(f"  predicted action: {pred_action}")
    next_token = int(gpt_logits[0, -1].argmax().item())
    print(f"  GPT-2 next token: '{decode([next_token])}'")
    print("\n=== ОБА выхода из ОДНОГО forward'а: ✓ ===")
