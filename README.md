# lafontaine-gpt

A language model built from scratch to generate La Fontaine fables in classical French.

---

## What this project is

A GPT-style transformer trained entirely from scratch — no pretrained weights, no external embeddings, no shortcuts. Just PyTorch, math, and La Fontaine.

The goal is simple: generate new fables in the style of Jean de La Fontaine.

---

## Starting point: fables only

The obvious first attempt was to train directly on the 242 fables (~95k tokens). A transformer has to learn everything at once from that — French grammar, syntax, vocabulary, narrative structure, poetic style — which is too much to ask from too little data.

Best result from that attempt:

```
le loup, dit- t'en ne sommes, l'elle en peut- vous me suis point de moi?
le jour par leur plus fait le plus que- nous vous vous ne n'ai de les biens
```

Recognizable characters (loup, renard, lion), zero coherent sentence structure.

The fix is the standard one used by real LLMs: pretrain on a large general corpus first, then finetune on the target domain.

---

## Pretraining corpus

Three sources, all extracted via Hugging Face datasets and assembled into raw `.txt` files:

| Source | Size |
|---|---|
| French Wikipedia | 7.38 GB |
| Common Corpus (French subset) | 9.76 GB |
| Project Gutenberg (French books) | 2.12 GB |
| **Total** | **~19.3 GB** |

Wikipedia gives broad general-language coverage. Common Corpus adds variety (open government text, books, periodicals — all public domain or open-licensed). Gutenberg adds classical French literature, closer in register to La Fontaine than Wikipedia's encyclopedic tone.

---

## Turning raw text into training data: the tokenizer

Going from 19 GB of raw text to a trained vocabulary and encoded token files turned out to be the hardest engineering problem in this project, harder than the model itself.

**BPE training.** The first version was a naive Python BPE trainer: recompute every pair frequency from scratch after every merge. On a corpus this size it would have taken days. We rewrote the trainer in Rust, then optimized it twice further:
- replaced full recomputation with incremental pair-count updates (only update the pairs touched by each merge)
- added an inverted index (pair → which words contain it) so each merge only rescans the words it actually affects, instead of the whole corpus
- replaced linear-scan "find the most frequent pair" with a lazy `BinaryHeap`, turning an O(n) lookup into O(log n)

Result: training a 32,000-token BPE vocabulary on the full corpus now takes about 10 minutes, down from a multi-day estimate in Python.

**Encoding.** Applying ~19,000 learned merge rules to every word in 19 GB of text is its own bottleneck. We parallelized encoding across all CPU cores with `rayon` (splitting each file into per-thread chunks), added a `dashmap`-backed cache so each unique word is only run through the merge rules once, and replaced sequential rule-by-rule application with a best-first merge selector (find the single highest-priority applicable merge per pass instead of trying all ~19,000 rules in order). Encoding the full corpus now finishes in about a minute.

The Rust binary (`bpe_tokenizer/`, built with `cargo`) handles both `train` (learn the BPE vocabulary) and `encode` (corpus to binary token ids). Python (`tokenizer.py`) only handles loading the trained vocabulary and encode/decode for inference. `dataset.py` calls the Rust binary automatically if the encoded `.bin` file doesn't exist yet.

Current state:

```
Loading pretrain_ids.bin...
  5,238,112,733 tokens total
```

5.2 billion tokens from 19.3 GB of raw text, vocabulary of 31,907 BPE tokens.

---

## Model

A standard decoder-only GPT architecture — nothing novel here, just conventional choices (causal self-attention, learned positional embeddings, pre-norm transformer blocks, weight-tied embedding/output projection) with hyperparameters sized to what a single GPU can train in reasonable time:

```
GPT initialized ===> 41,781,760 parameters
```

8 layers, 8 attention heads, 512-dim embeddings, 512-token context window. Small relative to GPT-2 (117M) or any real LLM, but the focus so far has been on getting the data pipeline right rather than scaling the architecture.

---

## Two-phase training

**Phase 1 — pretraining.** Train on the full 19.3 GB French corpus so the model learns grammar, vocabulary, and general sentence structure before ever seeing a fable. Best validation loss so far: 2.83.

**Phase 2 — finetuning.** Starting from the pretrained checkpoint, finetune exclusively on the 242 La Fontaine fables, no other data mixed in. This is where the model is expected to pick up fable-specific structure: animal characters, the closing moral, dialogue, verse rhythm.

With only ~95k tokens of finetuning data and a 41M-parameter model, overfitting is the dominant challenge here: training loss drops quickly while validation loss rises almost immediately. Mitigations tried so far: training on fables as individual padded sequences instead of concatenating and slicing them arbitrarily (preventing the end of one fable from bleeding into the start of another), plus freezing the first N transformer blocks during finetuning so only the later layers adapt, reducing the number of trainable parameters without touching the pretrained French-language base.

---

## Current generation quality

Not there yet, but improving. Recent samples from the finetuned checkpoint already show recognizable La Fontaine vocabulary and structure (classical spelling, animal characters, moral framing) but still suffer from repetition loops and occasional word fragmentation from the tokenizer.

Sample (prompt: "Le lion"):

```
le lion pour moi est un chien qui se plaît à la chasse il est sage et sage je le
crois mais il ne faut pas qu'on le sache je crois que la chasse de cette espèce
est plus douce que la nôtre [...] il étoit bon de ne pas faire il avoit fait son
métier et il avoit fait son métier il étoit beau il avoit fait son métier [...]
```

Classical syntax and tone are there early on, but the model falls into a repetition loop after a few dozen tokens — a common failure mode when the model loses track of context.

Next steps:
- repetition penalty during sampling, to break out of these loops
- add punctuation as base vocabulary tokens (the current tokenizer drops most punctuation, which likely contributes to run-on, comma-less generations)
- close the finetuning data gap (95k tokens vs 41M parameters) without diluting the La Fontaine-only style

---

## Project structure

```
lafontaine-gpt/
├── Data - Fables/             # 242 fables, 12 books
├── Data - French/
│   ├── Wikipedia/              # 7.38 GB
│   ├── CommonCorpus/           # 9.76 GB
│   └── Gutenberg/               # 2.12 GB
├── data_downloader/            # scripts to pull each source via Hugging Face
├── bpe_tokenizer/               # Rust BPE trainer + encoder (cargo project)
│   └── src/main.rs
├── tokenizer.py                  # Python encode/decode using the trained vocab
├── dataset.py                     # dataloaders; auto-encodes via the Rust binary
├── model.py                        # GPT architecture
├── train.py                         # pretrain / finetune loop, with resume support
├── generate.py                       # sample fables from a checkpoint
├── tokenizer.json                     # trained vocabulary (31,907 tokens)
├── pretrain_ids.bin / finetune_ids.bin
└── checkpoints/
    ├── pretrain.pt
    └── finetune.pt
```

---

## Quickstart

**1. Build the Rust tokenizer**
```bash
cd bpe_tokenizer
cargo build --release
cd ..
```

**2. Train the BPE vocabulary**
```bash
.\bpe_tokenizer\target\release\tokenizer_train.exe train --vocab_size 32000 --min_freq 2
```

**3. Train the model** (corpus encoding happens automatically on first run)
```bash
python train.py --phase pretrain
python train.py --phase finetune
```

**4. Generate fables**
```bash
python generate.py --checkpoint checkpoints/finetune.pt --prompt "Le loup" --tokens 300 --temperature 0.8
```