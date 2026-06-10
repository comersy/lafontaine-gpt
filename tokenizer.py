"""
BPE Tokenizer from scratch for lafontaine-gpt.

Usage:
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(text)
    tokenizer.save("tokenizer.json")

    tokenizer = BPETokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Le corbeau et le renard")
    text = tokenizer.decode(ids)
"""

import re
import json
from collections import defaultdict

# ── Hyperparameter ────────────────────────────────────────────────────────────
VOCAB_SIZE = 2000   # number of tokens in the final vocabulary


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_word_freqs(text: str) -> dict[str, int]:
    """
    Splits the text into words and returns their frequencies.
    Each word is represented as a space-separated sequence of characters,
    with a special </w> token at the end to mark word boundaries.

    Ex: "le renard" -> {"l e </w>": 1, "r e n a r d </w>": 1}
    """
    word_freqs: dict[str, int] = defaultdict(int)
    # Keep accents, apostrophes and hyphens specific to classical French
    words = re.findall(r"[a-zA-ZÀ-ÿ''\-]+", text.lower())
    for word in words:
        tokenized = " ".join(list(word)) + " </w>"
        word_freqs[tokenized] += 1
    return dict(word_freqs)


def get_pairs(word_freqs: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Counts all adjacent symbol pairs in the current vocabulary.

    Ex: {"l e </w>": 3} -> {("l","e"): 3, ("e","</w>"): 3}
    """
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_pair(
    pair: tuple[str, str],
    word_freqs: dict[str, int]
) -> dict[str, int]:
    """
    Merges a pair across all words in the current vocabulary.

    Ex: merge ("l","e") on {"l e </w>": 3} -> {"le </w>": 3}
    """
    new_word_freqs: dict[str, int] = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(pair)
    for word, freq in word_freqs.items():
        new_word = pattern.sub(merged, word)
        new_word_freqs[new_word] = freq
    return new_word_freqs


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer trained from scratch.

    Attributes:
        vocab_size  : target vocabulary size (main hyperparameter)
        merges      : ordered list of learned merge rules [(a, b), ...]
        vocab       : dict {token_string: token_id}
        vocab_inv   : dict {token_id: token_string}
    """

    # Reserved special tokens
    PAD_TOKEN   = "<pad>"
    UNK_TOKEN   = "<unk>"
    BOS_TOKEN   = "<bos>"   # beginning of sequence
    EOS_TOKEN   = "<eos>"   # end of sequence
    WORD_END    = "</w>"    # word boundary marker (internal to BPE)

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.vocab_inv: dict[int, str] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """
        Learns BPE merge rules from raw text.

        1. Initialize the vocab with all unique characters
        2. Iterate: find the most frequent pair, merge it, add to vocab
        3. Repeat until vocab_size is reached
        """
        print(f"BPE training — target: {self.vocab_size} tokens")

        # Word frequencies with character-level tokenization
        word_freqs = get_word_freqs(text)

        # Initial vocab = special tokens + all unique characters
        base_chars: set[str] = set()
        for word in word_freqs:
            for symbol in word.split():
                base_chars.add(symbol)

        vocab_tokens = self.SPECIAL_TOKENS + sorted(base_chars)
        n_merges = self.vocab_size - len(vocab_tokens)

        if n_merges <= 0:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small: "
                f"{len(vocab_tokens)} base tokens are already required."
            )

        print(f"  {len(vocab_tokens)} base tokens, {n_merges} merges to learn")

        # Learn merge rules
        for i in range(n_merges):
            pairs = get_pairs(word_freqs)
            if not pairs:
                print(f"  No more pairs available after {i} merges.")
                break

            best_pair = max(pairs, key=lambda p: pairs[p])
            word_freqs = merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            vocab_tokens.append("".join(best_pair))

            if (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{n_merges} — best pair: {''.join(best_pair)!r}")

        # Build the final vocabulary
        self._build_vocab(vocab_tokens)
        print(f"Final vocabulary: {len(self.vocab)} tokens")

    def _build_vocab(self, tokens: list[str]) -> None:
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.vocab_inv = {idx: tok for tok, idx in self.vocab.items()}

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """
        Applies the learned merge rules to a single word.
        """
        symbols = list(word) + [self.WORD_END]
        # Apply merges in the order they were learned
        for pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encodes a text into a list of token ids.

        Args:
            text        : raw text
            add_special : if True, prepends <bos> and appends <eos>
        """
        ids: list[int] = []
        if add_special:
            ids.append(self.vocab[self.BOS_TOKEN])

        words = re.findall(r"[a-zA-ZÀ-ÿ''\-]+|\S", text.lower())
        for word in words:
            if re.match(r"[a-zA-ZÀ-ÿ''\-]+", word):
                tokens = self._tokenize_word(word)
            else:
                tokens = [word]  # punctuation, digits...

            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab[self.UNK_TOKEN]))

        if add_special:
            ids.append(self.vocab[self.EOS_TOKEN])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token ids back to raw text.
        Strips special tokens and reconstructs words via </w>.
        """
        tokens = [self.vocab_inv.get(i, self.UNK_TOKEN) for i in ids]
        # Filter out special tokens
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        text = " ".join(tokens)
        # </w> marks word boundaries: replace " </w>" with a space
        text = text.replace(" " + self.WORD_END, " ")
        text = text.replace(self.WORD_END, "")
        return text.strip()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saves the tokenizer to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved: {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Loads a tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.merges = [tuple(p) for p in data["merges"]]
        tokenizer.vocab = data["vocab"]
        tokenizer.vocab_inv = {v: k for k, v in tokenizer.vocab.items()}
        print(f"Tokenizer loaded: {path} ({len(tokenizer.vocab)} tokens)")
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
        return f"BPETokenizer(vocab_size={self.vocab_size}, trained={len(self.merges)} merges)"


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import glob

    # Load all fables
    fable_files = glob.glob(os.path.join("Data - Fables", "**", "*.txt"), recursive=True)
    corpus = ""
    for path in sorted(fable_files):
        with open(path, "r", encoding="utf-8") as f:
            corpus += f.read() + "\n"

    print(f"Corpus loaded: {len(corpus)} characters, {len(fable_files)} fables\n")

    # Train
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(corpus)

    # Test encode/decode
    sample = "Le corbeau et le renard"
    ids = tokenizer.encode(sample, add_special=True)
    decoded = tokenizer.decode(ids)
    print(f"\nOriginal : {sample!r}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded!r}")

    # Save
    tokenizer.save("tokenizer.json")