"""
BPE Tokenizer from scratch for lafontaine-gpt.

Trained on the full corpus (Data - French + Data - Fables).
Vocabulary size is fixed at VOCAB_SIZE tokens.

Usage:
    python tokenizer.py
    python tokenizer.py --vocab_size 32000 --min_freq 2

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
from tqdm import tqdm


# ── Hyperparameters ───────────────────────────────────────────────────────────

VOCAB_SIZE = 32000
MIN_FREQ   = 2

FABLES_DIR = "Data - Fables"
FRENCH_DIR = "Data - French"


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(include_fables: bool = True, include_french: bool = True) -> str:
    corpus = ""

    if include_fables:
        files = sorted(glob.glob(os.path.join(FABLES_DIR, "**", "*.txt"), recursive=True))
        for path in tqdm(files, desc="Loading fables", unit="file"):
            with open(path, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n"
        print(f"Fables ===> {len(files)} files")

    if include_french:
        files = sorted(glob.glob(os.path.join(FRENCH_DIR, "**", "*.txt"), recursive=True))
        for path in tqdm(files, desc="Loading French corpus", unit="file"):
            with open(path, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n"
        print(f"French corpus ===> {len(files)} files")

    print(f"Total ===> {len(corpus):,} characters\n")
    return corpus


# ── BPE Helpers ───────────────────────────────────────────────────────────────

def get_word_freqs(text: str, min_freq: int) -> dict:
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
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_pair(pair: tuple, word_freqs: dict) -> dict:
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
    BPE tokenizer trained from scratch.

    Attributes:
        vocab_size : target vocabulary size
        min_freq   : minimum word frequency
        merges     : ordered list of learned merge rules
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
        print(f"BPE training ===> target: {self.vocab_size} tokens, min_freq: {self.min_freq}")

        word_freqs = get_word_freqs(text, self.min_freq)

        base_chars: set[str] = set()
        for word in word_freqs:
            for symbol in word.split():
                base_chars.add(symbol)

        vocab_tokens = self.SPECIAL_TOKENS + sorted(base_chars)
        n_merges     = self.vocab_size - len(vocab_tokens)

        if n_merges <= 0:
            raise ValueError(f"vocab_size={self.vocab_size} too small")

        print(f"  {len(vocab_tokens)} base tokens ===> {n_merges} merges to learn\n")

        for i in tqdm(range(n_merges), desc="BPE merges", unit="merge"):
            pairs = get_pairs(word_freqs)
            if not pairs:
                print(f"No more pairs after {i} merges.")
                break
            best       = max(pairs, key=lambda p: pairs[p])
            word_freqs = merge_pair(best, word_freqs)
            self.merges.append(best)
            vocab_tokens.append("".join(best))

        self._build_vocab(vocab_tokens)
        print(f"\nVocabulary ===> {len(self.vocab):,} tokens\n")

    def _build_vocab(self, tokens: list) -> None:
        self.vocab     = {tok: idx for idx, tok in enumerate(tokens)}
        self.vocab_inv = {idx: tok for tok, idx in self.vocab.items()}

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
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
        tokens = [self.vocab_inv.get(i, self.UNK_TOKEN) for i in ids]
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        text = ""
        for tok in tokens:
            if tok.endswith(self.WORD_END):
                # Word-final token: append word (without </w>) then a space
                text += tok[:-len(self.WORD_END)] + " "
            else:
                # Sub-word token: append directly, no space
                text += tok

        return text.strip()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size" : self.vocab_size,
                "min_freq"   : self.min_freq,
                "merges"     : self.merges,
                "vocab"      : self.vocab,
            }, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved ===> {path} ({len(self.vocab):,} tokens)")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok           = cls(vocab_size=data["vocab_size"], min_freq=data["min_freq"])
        tok.merges    = [tuple(p) for p in data["merges"]]
        tok.vocab     = data["vocab"]
        tok.vocab_inv = {v: k for k, v in tok.vocab.items()}
        print(f"Tokenizer loaded ===> {path} ({len(tok.vocab):,} tokens)")
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

    corpus    = load_corpus(include_fables=True, include_french=True)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.train(corpus)

    sample  = "Le corbeau et le renard"
    ids     = tokenizer.encode(sample, add_special=True)
    decoded = tokenizer.decode(ids)
    print(f"Original : {sample!r}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded!r}\n")

    tokenizer.save("tokenizer.json")