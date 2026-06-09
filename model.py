"""
Transformer decoder-only language model from scratch for lafontaine-gpt.

Architecture (GPT-style):
    token embedding + positional embedding
    → N x (masked multi-head self-attention + feed-forward)
    → layer norm
    → linear projection to vocab

Usage:
    model = GPT(GPTConfig(vocab_size=500))
    logits, loss = model(x, targets=y)
    ids = model.generate(prompt, max_new_tokens=200)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from tokenizer import BPETokenizer


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size  : int   = 500    # size of the BPE vocabulary
    block_size  : int   = 128    # maximum sequence length (context window)
    n_layer     : int   = 4      # number of transformer blocks
    n_head      : int   = 4      # number of attention heads
    n_embd      : int   = 128    # embedding dimension
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

        # Q, K, V projections packed in one linear layer for efficiency
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of ones
        # registered as a buffer so it moves to the right device automatically
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_size) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_size)
        attn  = (q @ k.transpose(-2, -1)) * scale          # (B, n_head, T, T)
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v                                       # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
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
        x = x + self.attn(self.ln1(x))   # attention with residual connection
        x = x + self.ff(self.ln2(x))     # feed-forward with residual connection
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

        # Final linear layer: projects embeddings to vocab logits
        # No bias, and we tie weights with token embedding (standard practice)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["token_emb"].weight = self.lm_head.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized — {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier-style initialization for Linear and Embedding layers."""
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
        """
        Forward pass.

        Args:
            idx     : token ids, shape (B, T)
            targets : target token ids, shape (B, T) — optional, for training

        Returns:
            logits  : shape (B, T, vocab_size)
            loss    : cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        device = idx.device
        pos    = torch.arange(T, device=device)  # (T,)

        # Token + positional embeddings
        tok_emb = self.transformer["token_emb"](idx)    # (B, T, n_embd)
        pos_emb = self.transformer["pos_emb"](pos)      # (T, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        # Final layer norm
        x = self.transformer["ln_f"](x)

        # Project to vocab
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten (B, T, vocab_size) → (B*T, vocab_size) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,   # ignore <pad> token (id=0)
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
        """
        Autoregressively generates new tokens given a prompt.

        Args:
            idx            : prompt token ids, shape (1, T)
            max_new_tokens : number of tokens to generate
            temperature    : >1 = more random, <1 = more focused
            top_k          : if set, only sample from the top-k most likely tokens

        Returns:
            tensor of shape (1, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Take logits of the last token
            logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from the distribution
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)   # (1, 1)
            idx       = torch.cat([idx, idx_next], dim=1)

        return idx


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = BPETokenizer.load("tokenizer.json")

    config = GPTConfig(
        vocab_size = len(tokenizer),
        block_size = 128,
        n_layer    = 4,
        n_head     = 4,
        n_embd     = 128,
        dropout    = 0.1,
    )

    model = GPT(config)

    # Dummy forward pass
    x = torch.randint(0, len(tokenizer), (2, 128))  # batch of 2 sequences
    y = torch.randint(0, len(tokenizer), (2, 128))
    logits, loss = model(x, targets=y)

    print(f"Logits shape : {tuple(logits.shape)}")  # (2, 128, vocab_size)
    print(f"Loss         : {loss.item():.4f}")

    # Dummy generation
    prompt = torch.tensor([[tokenizer.bos_id]])
    output = model.generate(prompt, max_new_tokens=50, temperature=1.0, top_k=40)
    print(f"Generated    : {tokenizer.decode(output[0].tolist())!r}")