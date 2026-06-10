"""
Training loop for lafontaine-gpt.

Two phases:
    pretrain  — trains on Data - French (Moliere, Bossuet, Corneille, La Bruyere...)
    finetune  — fine-tunes on Data - Fables starting from a pretrain checkpoint

Usage:
    python train.py --phase pretrain
    python train.py --phase finetune
"""

import os
import json
import math
import time
import argparse
import torch
from torch.utils.data import DataLoader

from tokenizer import BPETokenizer
from dataset   import build_loaders, BLOCK_SIZE, BATCH_SIZE
from model     import GPT, GPTConfig


# ── Hyperparameters ───────────────────────────────────────────────────────────

PRETRAIN_CONFIG = {
    "n_layer"    : 4,
    "n_head"     : 4,
    "n_embd"     : 256,
    "dropout"    : 0.1,
    "max_iters"  : 20000,
    "lr"         : 3e-4,
    "min_lr"     : 3e-5,
    "warmup"     : 500,
    "batch_size" : 32,
    "block_size" : BLOCK_SIZE,
    "eval_every" : 500,
    "eval_iters" : 50,
    "checkpoint" : "checkpoints/pretrain.pt",
    "log_file"   : "pretrain_log.json",
}

FINETUNE_CONFIG = {
    "n_layer"    : 4,
    "n_head"     : 4,
    "n_embd"     : 256,
    "dropout"    : 0.3,   # higher dropout to avoid overfitting on small fables corpus
    "max_iters"  : 5000,
    "lr"         : 1e-4,  # lower lr for finetuning
    "min_lr"     : 1e-5,
    "warmup"     : 100,
    "batch_size" : 32,
    "block_size" : BLOCK_SIZE,
    "eval_every" : 250,
    "eval_iters" : 50,
    "checkpoint" : "checkpoints/finetune.pt",
    "log_file"   : "finetune_log.json",
}

DEVICE = (
    "cuda" if torch.cuda.is_available()  else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: dict) -> float:
    """Cosine decay with linear warmup."""
    if it < cfg["warmup"]:
        return cfg["lr"] * it / cfg["warmup"]
    if it > cfg["max_iters"]:
        return cfg["min_lr"]
    decay = (it - cfg["warmup"]) / (cfg["max_iters"] - cfg["warmup"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model: GPT, train_loader: DataLoader, val_loader: DataLoader, eval_iters: int) -> dict:
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total = 0.0
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, targets=y)
            total += loss.item()
        losses[split] = total / min(eval_iters, len(loader))
    model.train()
    return losses


# ── Training ──────────────────────────────────────────────────────────────────

def train(phase: str) -> None:

    assert phase in ("pretrain", "finetune"), "phase must be 'pretrain' or 'finetune'"
    cfg = PRETRAIN_CONFIG if phase == "pretrain" else FINETUNE_CONFIG

    os.makedirs("checkpoints", exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = BPETokenizer.load("tokenizer.json")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        mode        = phase,
        tokenizer   = tokenizer,
        block_size  = cfg["block_size"],
        batch_size  = cfg["batch_size"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size = len(tokenizer),
        block_size = cfg["block_size"],
        n_layer    = cfg["n_layer"],
        n_head     = cfg["n_head"],
        n_embd     = cfg["n_embd"],
        dropout    = cfg["dropout"],
    )

    model = GPT(model_cfg).to(DEVICE)

    # For finetuning, load pretrain checkpoint
    if phase == "finetune":
        pretrain_path = PRETRAIN_CONFIG["checkpoint"]
        if os.path.exists(pretrain_path):
            print(f"Loading pretrain checkpoint ===> {pretrain_path}")
            torch.serialization.add_safe_globals([GPTConfig])
            ckpt = torch.load(pretrain_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded ===> iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}\n")
        else:
            print(f"Warning: no pretrain checkpoint found at {pretrain_path}, finetuning from scratch.\n")

    # Update dropout for finetuning
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = cfg["dropout"]

    print(f"Device ===> {DEVICE}")
    print(f"Phase  ===> {phase}")
    print(f"Params ===> {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": 1e-2},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"])

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_iter    = iter(train_loader)
    t_start       = time.time()

    log = {
        "phase"  : phase,
        "config" : {**cfg, "vocab_size": len(tokenizer)},
        "evals"  : [],
        "best_val_loss" : None,
        "best_iter"     : None,
        "total_time_sec": None,
    }

    print(f"Starting {phase} for {cfg['max_iters']} iterations\n")

    for it in range(cfg["max_iters"]):

        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % cfg["eval_every"] == 0 or it == cfg["max_iters"] - 1:
            losses  = estimate_loss(model, train_loader, val_loader, cfg["eval_iters"])
            elapsed = time.time() - t_start

            print(
                f"[{phase} | iter {it:6d}] "
                f"train loss: {losses['train']:.4f} ===> "
                f"val loss: {losses['val']:.4f} ===> "
                f"{elapsed:.0f}s"
            )

            log["evals"].append({
                "iter"       : it,
                "train_loss" : round(losses["train"], 4),
                "val_loss"   : round(losses["val"],   4),
                "lr"         : round(lr, 6),
                "elapsed_sec": round(elapsed, 1),
            })

            if losses["val"] < best_val_loss:
                best_val_loss        = losses["val"]
                log["best_val_loss"] = round(best_val_loss, 4)
                log["best_iter"]     = it
                torch.save({
                    "model_state": model.state_dict(),
                    "config"     : model_cfg,
                    "iter"       : it,
                    "val_loss"   : best_val_loss,
                    "optimizer"  : optimizer.state_dict(),
                }, cfg["checkpoint"])
                print(f"  Checkpoint saved ===> {cfg['checkpoint']} (val loss: {best_val_loss:.4f})\n")

            log["total_time_sec"] = round(time.time() - t_start, 1)
            with open(cfg["log_file"], "w") as f:
                json.dump(log, f, indent=2)

    print(f"\n{phase} complete ===> log saved to {cfg['log_file']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="pretrain", choices=["pretrain", "finetune"])
    args = parser.parse_args()
    train(args.phase)