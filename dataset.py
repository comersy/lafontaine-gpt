"""
Dataset for lafontaine-gpt.

Loads all fables, encodes them with the Word tokenizer,
and returns (x, y) pairs for autoregressive training.

Usage:
    dataset = FablesDataset(data_dir="Data - Fables", tokenizer=tokenizer, block_size=128)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, y in loader:
        # x: (batch_size, block_size)  — input token ids
        # y: (batch_size, block_size)  — target token ids (x shifted by 1)
        ...
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import WordTokenizer

# ── Hyperparameters ───────────────────────────────────────────────────────────
BLOCK_SIZE = 128   # number of tokens per training sequence
BATCH_SIZE = 32    # number of sequences per batch
TRAIN_SPLIT = 0.9  # 90% train, 10% validation


# ── Dataset ───────────────────────────────────────────────────────────────────

class FablesDataset(Dataset):
    """
    PyTorch Dataset for La Fontaine fables.

    Each item is a pair (x, y) of token id tensors of length block_size,
    where y = x shifted by one position (next-token prediction).

    Args:
        data_dir   : root folder containing the fable .txt files
        tokenizer  : trained WordTokenizer
        block_size : length of each training sequence
        split      : "train" or "val"
        train_split: fraction of data used for training (default 0.9)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: WordTokenizer,
        block_size: int = BLOCK_SIZE,
        split: str = "train",
        train_split: float = TRAIN_SPLIT,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.block_size = block_size
        self.tokenizer  = tokenizer

        # ── Load and encode the corpus ────────────────────────────────────────
        corpus = self._load_corpus(data_dir)
        ids    = tokenizer.encode(corpus)

        print(f"Corpus: {len(corpus):,} characters → {len(ids):,} tokens")

        # ── Train / val split ─────────────────────────────────────────────────
        split_idx = int(len(ids) * train_split)
        if split == "train":
            self.ids = ids[:split_idx]
        else:
            self.ids = ids[split_idx:]

        self.ids = torch.tensor(self.ids, dtype=torch.long)
        print(f"  {split} split: {len(self.ids):,} tokens, {len(self):,} sequences")

    # ── Corpus loading ────────────────────────────────────────────────────────

    def _load_corpus(self, data_dir: str) -> str:
        """
        Reads all .txt fable files recursively and concatenates them.
        Each fable is separated by a <eos> token to signal boundaries.
        """
        fable_files = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
        )
        if not fable_files:
            raise FileNotFoundError(f"No .txt files found in {data_dir!r}")

        print(f"Loading {len(fable_files)} fables from {data_dir!r}")

        corpus = ""
        eos = self.tokenizer.EOS_TOKEN
        bos = self.tokenizer.BOS_TOKEN
        for path in fable_files:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            # Wrap each fable with <bos> ... <eos> so the model learns
            # where fables start and end
            corpus += f"{bos} {text} {eos}\n"

        return corpus

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        # Number of non-overlapping sequences of length block_size
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x, y) where:
            x = ids[idx * block_size : (idx+1) * block_size]
            y = ids[idx * block_size + 1 : (idx+1) * block_size + 1]
        """
        start = idx * self.block_size
        x = self.ids[start     : start + self.block_size]
        y = self.ids[start + 1 : start + self.block_size + 1]
        return x, y


# ── Helper: build train and val loaders ───────────────────────────────────────

def build_loaders(
    data_dir: str,
    tokenizer: WordTokenizer,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
) -> tuple[DataLoader, DataLoader]:
    """
    Builds and returns (train_loader, val_loader).

    Args:
        data_dir    : root folder with fable .txt files
        tokenizer   : trained WordTokenizer
        block_size  : sequence length
        batch_size  : number of sequences per batch
        train_split : fraction of data for training
    """
    train_dataset = FablesDataset(data_dir, tokenizer, block_size, split="train", train_split=train_split)
    val_dataset   = FablesDataset(data_dir, tokenizer, block_size, split="val",   train_split=train_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = WordTokenizer.load("tokenizer.json")

    # Build loaders
    train_loader, val_loader = build_loaders(
        data_dir   = "Data - Fables",
        tokenizer  = tokenizer,
        block_size = BLOCK_SIZE,
        batch_size = BATCH_SIZE,
    )

    # Inspect one batch
    x, y = next(iter(train_loader))
    print(f"\nBatch shape  : x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"x[0]         : {x[0].tolist()}")
    print(f"y[0]         : {y[0].tolist()}")
    print(f"Decoded x[0] : {tokenizer.decode(x[0].tolist())!r}")