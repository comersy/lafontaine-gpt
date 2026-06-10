"""
Word-level tokenizer from scratch for lafontaine-gpt.

Trained on the full corpus (Data - French + Data - Fables).
Each unique word becomes a token. Vocabulary size is automatic.

Usage:
    python tokenizer.py
    python tokenizer.py --min_freq 2

    tokenizer = WordTokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Le corbeau et le renard")
    text = tokenizer.decode(ids)
"""

import re
import json
import argparse
import glob
import os
from collections import Counter
from tqdm import tqdm


# ── Hyperparameters ───────────────────────────────────────────────────────────

MIN_FREQ   = 5      # ignore words appearing less than N times

FABLES_DIR = "Data - Fables"
FRENCH_DIR = "Data - French"
PRETRAIN_AUTHORS = ["Moliere", "Bossuet", "Corneille", "La Bruyere", "Racine"]


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
        count = 0
        for author in PRETRAIN_AUTHORS:
            author_dir = os.path.join(FRENCH_DIR, author)
            if not os.path.exists(author_dir):
                continue
            files = sorted(glob.glob(os.path.join(author_dir, "*.txt")))
            for path in tqdm(files, desc=f"Loading {author}", unit="file"):
                with open(path, "r", encoding="utf-8") as f:
                    corpus += f.read() + "\n"
            count += len(files)
        print(f"French corpus ===> {count} files")

    print(f"Total ===> {len(corpus):,} characters\n")
    return corpus


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer.
    Each unique word in the corpus becomes a token.
    Words below min_freq are mapped to <unk>.

    Attributes:
        vocab     : dict {token_string: token_id}
        vocab_inv : dict {token_id: token_string}
        min_freq  : minimum frequency for a word to enter the vocab
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq  = min_freq
        self.vocab     : dict[str, int] = {}
        self.vocab_inv : dict[int, str] = {}

    # ── Splitting ─────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """
        Splits text into word and punctuation tokens.
        Keeps French accents and apostrophes.
        Ex: "Le loup, dit-il" -> ["le", "loup", ",", "dit", "-", "il"]
        """
        return re.findall(r"[a-zA-ZÀ-ÿ]+|[^\s]", text.lower())

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """
        Builds vocabulary from raw text.
        Special tokens first, then words sorted by frequency.
        """
        print(f"Training word tokenizer ===> min_freq={self.min_freq}")
        tokens = self._split(text)
        freq   = Counter(tokens)

        print(f"  {len(tokens):,} tokens, {len(freq):,} unique words")

        words = [w for w, c in freq.most_common() if c >= self.min_freq]
        print(f"  Keeping {len(words):,} words with freq >= {self.min_freq}")

        vocab_tokens = self.SPECIAL_TOKENS + words
        self._build_vocab(vocab_tokens)
        print(f"Vocabulary ===> {len(self.vocab):,} tokens\n")

    def _build_vocab(self, tokens: list[str]) -> None:
        self.vocab     = {tok: idx for idx, tok in enumerate(tokens)}
        self.vocab_inv = {idx: tok for tok, idx in self.vocab.items()}

    # ── Encoding / Decoding ───────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Encodes raw text into token ids."""
        ids = []
        if add_special:
            ids.append(self.vocab[self.BOS_TOKEN])
        for tok in self._split(text):
            ids.append(self.vocab.get(tok, self.vocab[self.UNK_TOKEN]))
        if add_special:
            ids.append(self.vocab[self.EOS_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decodes token ids back to text with smart spacing."""
        tokens = [self.vocab_inv.get(i, self.UNK_TOKEN) for i in ids]
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        no_space_before = set(".,;:!?)»\"-")
        no_space_after  = set("(«\"")

        text = ""
        for i, tok in enumerate(tokens):
            if i == 0:
                text += tok
            elif tok in no_space_before:
                text += tok
            elif tokens[i - 1] in no_space_after:
                text += tok
            elif tok == "'" or (i > 0 and tokens[i - 1] == "'"):
                text += tok
            else:
                text += " " + tok

        return text.strip()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "vocab": self.vocab}, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved ===> {path} ({len(self.vocab):,} tokens)")

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok            = cls(min_freq=data["min_freq"])
        tok.vocab      = data["vocab"]
        tok.vocab_inv  = {v: k for k, v in tok.vocab.items()}
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
    def __repr__(self) -> str: return f"WordTokenizer(vocab_size={len(self.vocab)}, min_freq={self.min_freq})"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=MIN_FREQ)
    args = parser.parse_args()

    corpus    = load_corpus(include_fables=True, include_french=True)
    tokenizer = WordTokenizer(min_freq=args.min_freq)
    tokenizer.train(corpus)

    sample  = "Le corbeau et le renard"
    ids     = tokenizer.encode(sample, add_special=True)
    decoded = tokenizer.decode(ids)
    print(f"Original : {sample!r}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded!r}\n")

    tokenizer.save("tokenizer.json")