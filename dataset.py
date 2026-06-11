"""
Dataset for lafontaine-gpt.

Two modes:
    pretrain  — loads all of Data - French
    finetune  — loads Data - Fables only
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer import BPETokenizer

# ── Hyperparameters ───────────────────────────────────────────────────────────

BLOCK_SIZE  = 256
BATCH_SIZE  = 32
TRAIN_SPLIT = 0.9

FABLES_DIR  = "Data - Fables"
FRENCH_DIR  = "Data - French"


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_pretrain_corpus() -> str:
    files  = sorted(glob.glob(os.path.join(FRENCH_DIR, "**", "*.txt"), recursive=True))
    corpus = ""
    for path in tqdm(files, desc="Loading French corpus", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            corpus += f.read() + "\n"
    print(f"Pretrain ===> {len(files)} files, {len(corpus):,} characters")
    return corpus


def load_finetune_corpus(tokenizer: BPETokenizer) -> str:
    files  = sorted(glob.glob(os.path.join(FABLES_DIR, "**", "*.txt"), recursive=True))
    corpus = ""
    bos    = tokenizer.BOS_TOKEN
    eos    = tokenizer.EOS_TOKEN
    for path in tqdm(files, desc="Loading fables", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        corpus += f"{bos} {text} {eos}\n"
    print(f"Finetune ===> {len(files)} fables, {len(corpus):,} characters")
    return corpus


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_corpus(corpus: str, tokenizer: BPETokenizer, chunk_size: int = 10000) -> list[int]:
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]
    ids    = []
    for chunk in tqdm(chunks, desc="Encoding", unit="chunk"):
        ids += tokenizer.encode(chunk)
    print(f"Encoded ===> {len(ids):,} tokens")
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, ids, block_size=BLOCK_SIZE, split="train", train_split=TRAIN_SPLIT):
        assert split in ("train", "val")
        self.block_size = block_size
        split_idx = int(len(ids) * train_split)
        ids = ids[:split_idx] if split == "train" else ids[split_idx:]
        self.ids = torch.tensor(ids, dtype=torch.long)
        print(f"  {split} ===> {len(self.ids):,} tokens, {len(self):,} sequences")

    def __len__(self):
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.ids[start     : start + self.block_size]
        y = self.ids[start + 1 : start + self.block_size + 1]
        return x, y


# ── Builder ───────────────────────────────────────────────────────────────────

def build_loaders(mode, tokenizer, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT):
    assert mode in ("pretrain", "finetune")
    print(f"\nBuilding {mode} loaders...")

    corpus = load_pretrain_corpus() if mode == "pretrain" else load_finetune_corpus(tokenizer)
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
        print(f"\n[{mode}] x={tuple(x.shape)}, y={tuple(y.shape)}")
        print(f"  Decoded x[0] : {tokenizer.decode(x[0].tolist())!r}\n")