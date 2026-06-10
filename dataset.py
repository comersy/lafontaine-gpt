"""
Dataset for lafontaine-gpt.

Two modes:
    pretrain  — loads Data - French (filtered by PRETRAIN_AUTHORS)
    finetune  — loads Data - Fables only

Usage:
    train_loader, val_loader = build_loaders(mode="pretrain", tokenizer=tokenizer)
    train_loader, val_loader = build_loaders(mode="finetune", tokenizer=tokenizer)
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer import BPETokenizer

# ── Hyperparameters ───────────────────────────────────────────────────────────

BLOCK_SIZE   = 128
BATCH_SIZE   = 32
TRAIN_SPLIT  = 0.9

FABLES_DIR   = "Data - Fables"
FRENCH_DIR   = "Data - French"
PRETRAIN_AUTHORS = ["Moliere", "Bossuet", "Corneille", "La Bruyere", "Racine"]


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_pretrain_corpus() -> str:
    """Loads all French authors in PRETRAIN_AUTHORS."""
    corpus = ""
    count  = 0
    for author in PRETRAIN_AUTHORS:
        author_dir = os.path.join(FRENCH_DIR, author)
        if not os.path.exists(author_dir):
            print(f"  Warning: {author_dir} not found, skipping.")
            continue
        files = sorted(glob.glob(os.path.join(author_dir, "*.txt")))
        for path in tqdm(files, desc=f"  Loading {author}", unit="file"):
            with open(path, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n"
        count += len(files)
    print(f"Pretrain corpus ===> {count} files, {len(corpus):,} characters")
    return corpus


def load_finetune_corpus() -> str:
    """Loads all La Fontaine fables, wrapping each with <bos>/<eos>."""
    files  = sorted(glob.glob(os.path.join(FABLES_DIR, "**", "*.txt"), recursive=True))
    corpus = ""
    for path in tqdm(files, desc="  Loading fables", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        corpus += f"<bos> {text} <eos>\n"
    print(f"Finetune corpus ===> {len(files)} fables, {len(corpus):,} characters")
    return corpus


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_corpus(corpus: str, tokenizer: BPETokenizer, chunk_size: int = 10000) -> list[int]:
    """
    Encodes a large corpus in chunks with a tqdm progress bar.
    """
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]
    ids    = []
    for chunk in tqdm(chunks, desc="  Encoding", unit="chunk"):
        ids += tokenizer.encode(chunk)
    print(f"  Encoded ===> {len(ids):,} tokens")
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Generic autoregressive dataset.
    Returns (x, y) pairs where y = x shifted by one token.
    """

    def __init__(
        self,
        ids         : list[int],
        block_size  : int   = BLOCK_SIZE,
        split       : str   = "train",
        train_split : float = TRAIN_SPLIT,
    ):
        assert split in ("train", "val")
        self.block_size = block_size

        split_idx = int(len(ids) * train_split)
        ids = ids[:split_idx] if split == "train" else ids[split_idx:]

        self.ids = torch.tensor(ids, dtype=torch.long)
        print(f"  {split} ===> {len(self.ids):,} tokens, {len(self):,} sequences")

    def __len__(self) -> int:
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        x = self.ids[start     : start + self.block_size]
        y = self.ids[start + 1 : start + self.block_size + 1]
        return x, y


# ── Builder ───────────────────────────────────────────────────────────────────

def build_loaders(
    mode        : str,
    tokenizer   : BPETokenizer,
    block_size  : int   = BLOCK_SIZE,
    batch_size  : int   = BATCH_SIZE,
    train_split : float = TRAIN_SPLIT,
) -> tuple[DataLoader, DataLoader]:
    """
    Builds train and val DataLoaders for the given mode.
    """
    assert mode in ("pretrain", "finetune")

    print(f"\nBuilding {mode} loaders...")

    corpus = load_pretrain_corpus() if mode == "pretrain" else load_finetune_corpus()
    ids    = encode_corpus(corpus, tokenizer)

    train_dataset = TextDataset(ids, block_size, split="train", train_split=train_split)
    val_dataset   = TextDataset(ids, block_size, split="val",   train_split=train_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = BPETokenizer.load("tokenizer.json")

    for mode in ("pretrain", "finetune"):
        train_loader, val_loader = build_loaders(mode=mode, tokenizer=tokenizer)
        x, y = next(iter(train_loader))
        print(f"\n[{mode}] batch ===> x={tuple(x.shape)}, y={tuple(y.shape)}")
        print(f"  Decoded x[0] : {tokenizer.decode(x[0].tolist())!r}\n")