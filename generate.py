"""
Generate La Fontaine fables from a trained checkpoint.

Usage:
    python generate.py
    python generate.py --prompt "Le loup" --tokens 200 --temperature 0.8 --top_k 40
"""

import os
import argparse
import torch

from tokenizer import WordTokenizer
from model     import GPT, GPTConfig


# ── Defaults ──────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.9    # >1 more random, <1 more focused
TOP_K           = 40     # only sample from the top-k tokens


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str) -> tuple[GPT, WordTokenizer]:
    print(f"Loading checkpoint ===> {checkpoint_path}")
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = WordTokenizer.load("tokenizer.json")

    config = checkpoint["config"]
    model  = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded ===> iter {checkpoint['iter']} ===> val loss {checkpoint['val_loss']:.4f}\n")
    return model, tokenizer


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    prompt        : str,
    model         : GPT,
    tokenizer     : WordTokenizer,
    device        : str,
    max_new_tokens: int   = MAX_NEW_TOKENS,
    temperature   : float = TEMPERATURE,
    top_k         : int   = TOP_K,
) -> str:
    # Encode prompt
    ids = tokenizer.encode(prompt, add_special=False)
    if not ids:
        ids = [tokenizer.bos_id]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
        )

    return tokenizer.decode(output[0].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate La Fontaine fables")
    parser.add_argument("--prompt",      type=str,   default="Le",  help="Text prompt to start generation")
    parser.add_argument("--tokens",      type=int,   default=MAX_NEW_TOKENS, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,    help="Sampling temperature")
    parser.add_argument("--top_k",       type=int,   default=TOP_K,          help="Top-k sampling")
    parser.add_argument("--checkpoint",  type=str,   default=CHECKPOINT_PATH,help="Path to checkpoint")
    parser.add_argument("--n",           type=int,   default=1,              help="Number of fables to generate")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()  else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device ===> {device}\n")

    model, tokenizer = load_model(args.checkpoint, device)

    for i in range(args.n):
        if args.n > 1:
            print(f"{'='*60}")
            print(f"  Fable {i + 1}")
            print(f"{'='*60}")

        text = generate(
            prompt         = args.prompt,
            model          = model,
            tokenizer      = tokenizer,
            device         = device,
            max_new_tokens = args.tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )
        print(text)
        print()


if __name__ == "__main__":
    main()