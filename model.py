"""
Transformer decoder-only language model from scratch for lafontaine-gpt.

Architecture (GPT-style):
    token embedding + positional embedding
    → N x (masked multi-head self-attention + feed-forward)
    → layer norm
    → linear projection to vocab

Usage:
    model = GPT(GPTConfig(vocab_size=5000))
    logits, loss = model(x, targets=y)
    ids = model.generate(prompt, max_new_tokens=200)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size  : int   = 5000   # size of the vocabulary
    block_size  : int   = 128    # maximum sequence length (context window)
    n_layer     : int   = 4      # number of transformer blocks
    n_head      : int   = 4      # number of attention heads
    n_embd      : int   = 256    # embedding dimension
    dropout     : float = 0.1    # dropout probability


# ── Building blocks ───────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Masked multi-head self-attention.
    Each token can only attend to previous tokens (causal mask),
    which is what makes the model autoregressive.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"

        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Expands the embedding to 4x its size, applies GELU, then projects back.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block: LayerNorm → Attention → residual
                         + LayerNorm → FeedForward → residual
    Pre-norm formulation (more stable than post-norm).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.ff   = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── Model ─────────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Args:
        config : GPTConfig with all hyperparameters
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "token_emb": nn.Embedding(config.vocab_size, config.n_embd),
            "pos_emb"  : nn.Embedding(config.block_size, config.n_embd),
            "drop"     : nn.Dropout(config.dropout),
            "blocks"   : nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            "ln_f"     : nn.LayerNorm(config.n_embd),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["token_emb"].weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized ===> {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        device  = idx.device
        pos     = torch.arange(T, device=device)

        tok_emb = self.transformer["token_emb"](idx)
        pos_emb = self.transformer["pos_emb"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)

        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)

        return idx


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tokenizer import WordTokenizer

    tokenizer = WordTokenizer.load("tokenizer.json")

    config = GPTConfig(
        vocab_size = len(tokenizer),
        block_size = 128,
        n_layer    = 4,
        n_head     = 4,
        n_embd     = 256,
        dropout    = 0.1,
    )

    model = GPT(config)

    x = torch.randint(0, len(tokenizer), (2, 128))
    y = torch.randint(0, len(tokenizer), (2, 128))
    logits, loss = model(x, targets=y)

    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Loss         : {loss.item():.4f}")

    prompt = torch.tensor([[tokenizer.bos_id]])
    output = model.generate(prompt, max_new_tokens=50, temperature=1.0, top_k=40)
    print(f"Generated    : {tokenizer.decode(output[0].tolist())!r}")