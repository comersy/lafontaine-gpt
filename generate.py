"""
Generate La Fontaine fables from a trained checkpoint.

Usage:
    python generate.py
    python generate.py --prompt "Le loup" --tokens 300 --temperature 0.8 --top_k 40
"""

import os
import argparse
import torch

from tokenizer import BPETokenizer
from model     import GPT, GPTConfig

# ── Defaults ──────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join("checkpoints", "finetune.pt")
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.8
TOP_K           = 40


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint ===> {checkpoint_path}")
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer  = BPETokenizer.load("tokenizer.json")
    model      = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded ===> iter {checkpoint['iter']} ===> val loss {checkpoint['val_loss']:.4f}\n")
    return model, tokenizer


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(prompt, model, tokenizer, device, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    ids = tokenizer.encode(prompt, add_special=False)
    if not ids:
        ids = [tokenizer.bos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(output[0].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default="Le",           help="Prompt")
    parser.add_argument("--tokens",      type=int,   default=MAX_NEW_TOKENS, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,    help="Sampling temperature")
    parser.add_argument("--top_k",       type=int,   default=TOP_K,          help="Top-k sampling")
    parser.add_argument("--checkpoint",  type=str,   default=CHECKPOINT_PATH,help="Checkpoint path")
    parser.add_argument("--n",           type=int,   default=1,              help="Number of fables")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device ===> {device}\n")

    model, tokenizer = load_model(args.checkpoint, device)

    for i in range(args.n):
        if args.n > 1:
            print(f"{'='*60}\n  Fable {i + 1}\n{'='*60}")
        text = generate(args.prompt, model, tokenizer, device, args.tokens, args.temperature, args.top_k)
        print(text)
        print()


if __name__ == "__main__":
    main()