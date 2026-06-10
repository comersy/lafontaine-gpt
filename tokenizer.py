"""
Word-level tokenizer from scratch for lafontaine-gpt.

Each unique word in the corpus becomes a token.
Vocabulary size is determined automatically from the corpus.

Usage:
    tokenizer = WordTokenizer()
    tokenizer.train(text)
    tokenizer.save("tokenizer.json")

    tokenizer = WordTokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Le corbeau et le renard")
    text = tokenizer.decode(ids)
"""

import re
import json
from collections import Counter


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer.

    Splits text into words and punctuation, builds a vocabulary
    from the corpus, and maps each token to an integer id.

    Attributes:
        vocab     : dict {token_string: token_id}
        vocab_inv : dict {token_id: token_string}
        min_freq  : minimum frequency for a word to be included in the vocab
    """

    # Reserved special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, min_freq: int = 1):
        """
        Args:
            min_freq : minimum number of occurrences for a word to enter the vocab.
                       Set to 1 to keep every word (default for small corpora).
        """
        self.min_freq  = min_freq
        self.vocab     : dict[str, int] = {}
        self.vocab_inv : dict[int, str] = {}

    # ── Splitting ─────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """
        Splits text into words and punctuation tokens.
        Keeps accents and classical French apostrophes.

        Ex: "Le loup, dit-il" -> ["le", "loup", ",", "dit", "-", "il"]
        """
        # Match words (with accents) or any single non-space character
        tokens = re.findall(r"[a-zA-ZÀ-ÿ]+|[^\s]", text.lower())
        return tokens

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """
        Builds the vocabulary from raw text.

        1. Split text into word tokens
        2. Count frequencies
        3. Keep words with freq >= min_freq
        4. Build vocab: special tokens first, then words sorted by frequency
        """
        tokens = self._split(text)
        freq   = Counter(tokens)

        print(f"Corpus ===> {len(tokens):,} tokens, {len(freq):,} unique words")

        # Filter by minimum frequency
        words = [w for w, c in freq.most_common() if c >= self.min_freq]

        print(f"Vocabulary ===> {len(self.SPECIAL_TOKENS)} special tokens + {len(words)} words = {len(self.SPECIAL_TOKENS) + len(words)} total")

        self._build_vocab(self.SPECIAL_TOKENS + words)

    def _build_vocab(self, tokens: list[str]) -> None:
        self.vocab     = {tok: idx for idx, tok in enumerate(tokens)}
        self.vocab_inv = {idx: tok for tok, idx in self.vocab.items()}

    # ── Encoding / Decoding ───────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encodes raw text into a list of token ids.

        Args:
            text        : raw text
            add_special : if True, prepends <bos> and appends <eos>
        """
        ids = []
        if add_special:
            ids.append(self.vocab[self.BOS_TOKEN])

        for tok in self._split(text):
            ids.append(self.vocab.get(tok, self.vocab[self.UNK_TOKEN]))

        if add_special:
            ids.append(self.vocab[self.EOS_TOKEN])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token ids back to text.
        Tries to reconstruct natural spacing around punctuation.
        """
        tokens = [self.vocab_inv.get(i, self.UNK_TOKEN) for i in ids]

        # Filter special tokens
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        # Reconstruct text with smart spacing
        text = ""
        no_space_before = set(".,;:!?)»\"-")
        no_space_after  = set("(«\"")

        for i, tok in enumerate(tokens):
            if i == 0:
                text += tok
            elif tok in no_space_before:
                text += tok
            elif tokens[i - 1] in no_space_after:
                text += tok
            elif tok == "'":
                text += tok
            elif i > 0 and tokens[i - 1] == "'":
                text += tok
            else:
                text += " " + tok

        return text.strip()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saves the tokenizer to a JSON file."""
        data = {
            "min_freq" : self.min_freq,
            "vocab"    : self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved ===> {path}")

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        """Loads a tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer          = cls(min_freq=data["min_freq"])
        tokenizer.vocab    = data["vocab"]
        tokenizer.vocab_inv = {v: k for k, v in tokenizer.vocab.items()}
        print(f"Tokenizer loaded ===> {path} ({len(tokenizer.vocab)} tokens)")
        return tokenizer

    # ── Useful properties ─────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={len(self.vocab)}, min_freq={self.min_freq})"


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import glob

    # Load all fables
    fable_files = sorted(glob.glob(os.path.join("Data - Fables", "**", "*.txt"), recursive=True))
    corpus = ""
    for path in sorted(fable_files):
        with open(path, "r", encoding="utf-8") as f:
            corpus += f.read() + "\n"

    print(f"Corpus loaded ===> {len(corpus):,} characters, {len(fable_files)} fables\n")

    # Train
    tokenizer = WordTokenizer(min_freq=1)
    tokenizer.train(corpus)

    # Test encode / decode
    sample  = "Le corbeau et le renard"
    ids     = tokenizer.encode(sample, add_special=True)
    decoded = tokenizer.decode(ids)

    print(f"\nOriginal : {sample!r}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded!r}")

    # Save
    tokenizer.save("tokenizer.json")