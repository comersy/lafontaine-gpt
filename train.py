"""
Training loop for lafontaine-gpt.

Trains the GPT model on La Fontaine fables using the Word tokenizer
and the FablesDataset. Saves checkpoints and logs train/val loss.

Usage:
    python train.py
"""

import os
import json
import math
import time
import torch
from torch.utils.data import DataLoader

from tokenizer import WordTokenizer
from dataset   import FablesDataset, build_loaders, BLOCK_SIZE, BATCH_SIZE
from model     import GPT, GPTConfig


# ── Hyperparameters ───────────────────────────────────────────────────────────

# Model
BLOCK_SIZE  = BLOCK_SIZE   # imported from dataset.py
N_LAYER     = 2
N_HEAD      = 2
N_EMBD      = 64
DROPOUT     = 0.4

# Training
MAX_ITERS       = 5000     # total number of training iterations
EVAL_INTERVAL   = 250      # evaluate on val set every N iters
EVAL_ITERS      = 50       # number of val batches to average for eval loss
LOG_INTERVAL    = 50       # print train loss every N iters

# Optimizer
LEARNING_RATE   = 3e-4     # peak learning rate
WEIGHT_DECAY    = 1e-2     # L2 regularization
GRAD_CLIP       = 1.0      # gradient clipping max norm

# Learning rate schedule (cosine decay with warmup)
WARMUP_ITERS    = 200      # linear warmup over first N iters
LR_DECAY_ITERS  = MAX_ITERS
MIN_LR          = 3e-5     # minimum learning rate (= 10% of peak)

# Checkpoints
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_NAME = "best_model.pt"
LOG_FILE        = "training_log.json"

# Device
DEVICE = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(it: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    - Linear warmup from 0 to LEARNING_RATE over WARMUP_ITERS steps
    - Cosine decay from LEARNING_RATE to MIN_LR over LR_DECAY_ITERS steps
    - Constant MIN_LR after LR_DECAY_ITERS
    """
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, float]:
    """
    Estimates average loss on train and val sets over EVAL_ITERS batches.
    Switches model to eval mode, then back to train mode.
    """
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total = 0.0
        for i, (x, y) in enumerate(loader):
            if i >= EVAL_ITERS:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, targets=y)
            total += loss.item()
        losses[split] = total / min(EVAL_ITERS, len(loader))
    model.train()
    return losses


# ── Training loop ─────────────────────────────────────────────────────────────

def train() -> None:

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = WordTokenizer.load("tokenizer.json")

    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        data_dir   = "Data - Fables",
        tokenizer  = tokenizer,
        block_size = BLOCK_SIZE,
        batch_size = BATCH_SIZE,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    config = GPTConfig(
        vocab_size = len(tokenizer),
        block_size = BLOCK_SIZE,
        n_layer    = N_LAYER,
        n_head     = N_HEAD,
        n_embd     = N_EMBD,
        dropout    = DROPOUT,
    )
    model = GPT(config).to(DEVICE)
    print(f"Device: {DEVICE}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate parameters: apply weight decay only to weight matrices,
    # not to biases or LayerNorm parameters
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=LEARNING_RATE)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_iter    = iter(train_loader)
    t0            = time.time()
    t_start       = time.time()

    # Training log written to disk after every eval
    log = {
        "config": {
            "vocab_size" : len(tokenizer),
            "block_size" : BLOCK_SIZE,
            "n_layer"    : N_LAYER,
            "n_head"     : N_HEAD,
            "n_embd"     : N_EMBD,
            "dropout"    : DROPOUT,
            "max_iters"  : MAX_ITERS,
            "lr"         : LEARNING_RATE,
            "batch_size" : BATCH_SIZE,
        },
        "evals"          : [],   # one entry per eval step
        "best_val_loss"  : None,
        "best_iter"      : None,
        "total_time_sec" : None,
    }

    print(f"\nStarting training for {MAX_ITERS} iterations\n")

    for it in range(MAX_ITERS):

        # Update learning rate
        lr = get_lr(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Get next batch (cycle through the loader)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        # Forward + backward
        logits, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # ── Evaluation + checkpoint ───────────────────────────────────────────
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_loader, val_loader)
            elapsed = time.time() - t_start
            print(
                f"\n[Eval iter {it}] "
                f"train loss: {losses['train']:.4f} ===> "
                f"val loss: {losses['val']:.4f}\n"
            )

            # Append to log
            log["evals"].append({
                "iter"       : it,
                "train_loss" : round(losses["train"], 4),
                "val_loss"   : round(losses["val"],   4),
                "lr"         : round(lr, 6),
                "elapsed_sec": round(elapsed, 1),
            })

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                log["best_val_loss"] = round(best_val_loss, 4)
                log["best_iter"]     = it
                checkpoint = {
                    "model_state": model.state_dict(),
                    "config"     : config,
                    "iter"       : it,
                    "val_loss"   : best_val_loss,
                    "optimizer"  : optimizer.state_dict(),
                }
                path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
                torch.save(checkpoint, path)
                print(f"  Checkpoint saved ===> {path} ===> val loss: {best_val_loss:.4f}\n")

            # Write log to disk after every eval
            log["total_time_sec"] = round(time.time() - t_start, 1)
            with open(LOG_FILE, "w") as f:
                json.dump(log, f, indent=2)

    print(f"Training complete. Log saved ===> {LOG_FILE}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()