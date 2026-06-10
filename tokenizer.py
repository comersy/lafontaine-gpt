"""
BPE Tokenizer from scratch for lafontaine-gpt.

Trained on the full corpus (Data - French + Data - Fables) so that
the vocabulary covers both the pretraining language and the finetuning style.

Usage:
    python tokenizer.py
    python tokenizer.py --vocab_size 5000 --min_freq 2

    tokenizer = BPETokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Le corbeau et le renard")
    text = tokenizer.decode(ids)
"""

import re
import json
import argparse
import glob
import os
from collections import defaultdict


# ── Hyperparameters ───────────────────────────────────────────────────────────

VOCAB_SIZE = 5000   # number of tokens in the final vocabulary
MIN_FREQ   = 2      # ignore words appearing less than N times in the corpus

# Directories
FABLES_DIR = "Data - Fables"
FRENCH_DIR = "Data - French"

# Authors to include in pretraining corpus (ignores Dumas, Montaigne, etc.)
PRETRAIN_AUTHORS = ["Moliere", "Bossuet", "Corneille", "La Bruyere", "Racine"]


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(include_fables: bool = True, include_french: bool = True) -> str:
    """
    Loads and concatenates all .txt files from the selected directories.

    Args:
        include_fables : include Data - Fables
        include_french : include Data - French (filtered by PRETRAIN_AUTHORS)
    """
    corpus = ""

    if include_fables:
        files = sorted(glob.glob(os.path.join(FABLES_DIR, "**", "*.txt"), recursive=True))
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n"
        print(f"Fables ===> {len(files)} files loaded")

    if include_french:
        count = 0
        for author in PRETRAIN_AUTHORS:
            author_dir = os.path.join(FRENCH_DIR, author)
            if not os.path.exists(author_dir):
                continue
            files = sorted(glob.glob(os.path.join(author_dir, "*.txt")))
            for path in files:
                with open(path, "r", encoding="utf-8") as f:
                    corpus += f.read() + "\n"
            count += len(files)
        print(f"French corpus ===> {count} files loaded from {PRETRAIN_AUTHORS}")

    print(f"Total corpus ===> {len(corpus):,} characters\n")
    return corpus


# ── BPE Helpers ───────────────────────────────────────────────────────────────

def get_word_freqs(text: str, min_freq: int = MIN_FREQ) -> dict:
    """
    Splits text into words, returns character-level tokenized words with frequencies.
    Words appearing less than min_freq times are discarded.

    Ex: "le renard" -> {"l e </w>": 12, "r e n a r d </w>": 8}
    """
    raw_freq = defaultdict(int)
    words = re.findall(r"[a-zA-ZÀ-ÿ''\-]+", text.lower())
    for word in words:
        raw_freq[word] += 1

    word_freqs = {}
    for word, freq in raw_freq.items():
        if freq >= min_freq:
            tokenized = " ".join(list(word)) + " </w>"
            word_freqs[tokenized] = freq

    return word_freqs


def get_pairs(word_freqs: dict) -> dict:
    """Counts all adjacent symbol pairs across the vocabulary."""
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_pair(pair: tuple, word_freqs: dict) -> dict:
    """Merges the most frequent pair across all words."""
    new_word_freqs = {}
    bigram  = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged  = "".join(pair)
    for word, freq in word_freqs.items():
        new_word_freqs[pattern.sub(merged, word)] = freq
    return new_word_freqs


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer trained from scratch.

    Attributes:
        vocab_size : target vocabulary size
        min_freq   : minimum word frequency to enter the vocabulary
        merges     : ordered list of learned merge rules [(a, b), ...]
        vocab      : dict {token_string: token_id}
        vocab_inv  : dict {token_id: token_string}
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    WORD_END  = "</w>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = VOCAB_SIZE, min_freq: int = MIN_FREQ):
        self.vocab_size = vocab_size
        self.min_freq   = min_freq
        self.merges     : list[tuple] = []
        self.vocab      : dict[str, int] = {}
        self.vocab_inv  : dict[int, str] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """Learns BPE merge rules from raw text."""
        print(f"BPE training ===> target: {self.vocab_size} tokens, min_freq: {self.min_freq}")

        word_freqs = get_word_freqs(text, self.min_freq)

        # Base vocabulary: special tokens + all unique characters
        base_chars: set[str] = set()
        for word in word_freqs:
            for symbol in word.split():
                base_chars.add(symbol)

        vocab_tokens = self.SPECIAL_TOKENS + sorted(base_chars)
        n_merges     = self.vocab_size - len(vocab_tokens)

        if n_merges <= 0:
            raise ValueError(f"vocab_size={self.vocab_size} too small, need at least {len(vocab_tokens)}")

        print(f"  {len(vocab_tokens)} base tokens ===> {n_merges} merges to learn")

        for i in range(n_merges):
            pairs = get_pairs(word_freqs)
            if not pairs:
                print(f"  No more pairs after {i} merges.")
                break

            best       = max(pairs, key=lambda p: pairs[p])
            word_freqs = merge_pair(best, word_freqs)
            self.merges.append(best)
            vocab_tokens.append("".join(best))

            if (i + 1) % 500 == 0:
                print(f"  Merge {i + 1}/{n_merges} ===> {(''.join(best))!r}")

        self._build_vocab(vocab_tokens)
        print(f"Vocabulary ===> {len(self.vocab)} tokens\n")

    def _build_vocab(self, tokens: list) -> None:
        self.vocab     = {tok: idx for idx, tok in enumerate(tokens)}
        self.vocab_inv = {idx: tok for tok, idx in self.vocab.items()}

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """Applies learned merge rules to a single word."""
        symbols = list(word) + [self.WORD_END]
        for pair in self.merges:
            i, new_symbols = 0, []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Encodes raw text into token ids."""
        ids = []
        if add_special:
            ids.append(self.vocab[self.BOS_TOKEN])

        for token in re.findall(r"[a-zA-ZÀ-ÿ''\-]+|\S", text.lower()):
            if re.match(r"[a-zA-ZÀ-ÿ''\-]+", token):
                for tok in self._tokenize_word(token):
                    ids.append(self.vocab.get(tok, self.vocab[self.UNK_TOKEN]))
            else:
                ids.append(self.vocab.get(token, self.vocab[self.UNK_TOKEN]))

        if add_special:
            ids.append(self.vocab[self.EOS_TOKEN])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decodes token ids back to text."""
        tokens = [self.vocab_inv.get(i, self.UNK_TOKEN) for i in ids]
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        text   = " ".join(tokens)
        text   = text.replace(" " + self.WORD_END, " ").replace(self.WORD_END, "")
        return text.strip()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saves the tokenizer to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size" : self.vocab_size,
                "min_freq"   : self.min_freq,
                "merges"     : self.merges,
                "vocab"      : self.vocab,
            }, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved ===> {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Loads a tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok            = cls(vocab_size=data["vocab_size"], min_freq=data["min_freq"])
        tok.merges     = [tuple(p) for p in data["merges"]]
        tok.vocab      = data["vocab"]
        tok.vocab_inv  = {v: k for k, v in tok.vocab.items()}
        print(f"Tokenizer loaded ===> {path} ({len(tok.vocab)} tokens)")
        return tok

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int: return self.vocab[self.PAD_TOKEN]
    @property
    def unk_id(self) -> int: return self.vocab[self.UNK_TOKEN]
    @property
    def bos_id(self) -> int: return self.vocab[self.BOS_TOKEN]
    @property
    def eos_id(self) -> int: return self.vocab[self.EOS_TOKEN]

    def __len__(self)  -> int: return len(self.vocab)
    def __repr__(self) -> str: return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min_freq",   type=int, default=MIN_FREQ)
    args = parser.parse_args()

    # Train on both corpora so the vocab covers pretraining + finetuning
    corpus = load_corpus(include_fables=True, include_french=True)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.train(corpus)

    # Quick test
    sample  = "Le corbeau et le renard"
    ids     = tokenizer.encode(sample, add_special=True)
    decoded = tokenizer.decode(ids)
    print(f"Original : {sample!r}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded!r}\n")

    tokenizer.save("tokenizer.json")