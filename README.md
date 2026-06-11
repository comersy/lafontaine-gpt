# lafontaine-gpt

A language model built from scratch to generate La Fontaine fables in classical French.

---

## What this project is

A GPT-style transformer trained entirely from scratch — no pretrained weights, no external embeddings, no shortcuts. Just PyTorch, math, and La Fontaine.

The goal is simple: generate new fables in the style of Jean de La Fontaine. Getting there turned out to be less simple.

---

## Iteration 1 — fables only

The obvious starting point: train directly on the 242 fables (~150k tokens).

A transformer trained on 150k tokens has to learn everything at once — French grammar, syntax, vocabulary, narrative structure, poetic style. That is too much to ask from too little data.

Best result:

```
le loup, dit- t'en ne sommes, l'elle en peut- vous me suis point de moi?
le jour par leur plus fait le plus que- nous vous vous ne n'ai de les biens
```

Recognizable characters (loup, renard, lion), zero coherent structure.

---

## Iteration 2 — adding classical French authors

The model needs to learn French before it learns fables. We added contemporaries of La Fontaine: Molière, Bossuet, Corneille, La Bruyère. Same epoch, same language, same classical style.

Better, but still too little data. The gap between pretraining and finetuning was too large, and the model kept mixing Molière's theatre style with La Fontaine's fable structure.

---

## Iteration 3 — French Wikipedia

To give the model a real foundation in French, we added the full French Wikipedia dump: 7.9 GB of text, covering all the language the model will ever need. Combined with the classical authors, the pretraining corpus is now large enough to actually learn French before touching a single fable.

**Total pretraining corpus: 7.9 GB across 20 files.**
**Finetuning corpus: 242 fables.**

---

## The tokenizer problem

A vocabulary of 32,000 BPE tokens (same size as LLaMA) requires learning ~32,000 merge rules across a 7.9 GB corpus. The Python implementation would have taken several days.

We rewrote the BPE trainer in Rust (`tokenizer.rs`) with an incremental pair-count update: instead of recomputing all pair frequencies from scratch after each merge (O(n × m)), we update only the pairs affected by the merge (O(n)). Training the full vocabulary now takes around 10 minutes.

The Rust binary outputs a `tokenizer.json` compatible with the Python encode/decode pipeline. The encoding of the 7.9 GB corpus to binary token ids is also done in Rust, with the Python training loop calling the binary automatically if the encoded file does not exist yet.

---

## Two-phase training

**Phase 1 — pretraining**

Train on the full French corpus (Wikipedia + classical authors). The model learns French: grammar, syntax, vocabulary, narrative flow. No fables yet.

**Phase 2 — finetuning**

Fine-tune on the 242 La Fontaine fables only. The model already speaks French. Now it learns the specific style: animal characters, moral at the end, alexandrine verse, dialogue structure.

This mirrors how real LLMs are built: massive general pretraining, then domain-specific finetuning.

---

## Architecture

- Decoder-only transformer (GPT-style)
- Causal masked multi-head self-attention
- BPE tokenizer, 32,000 tokens, trained from scratch in Rust
- Positional embeddings
- Cosine learning rate schedule with linear warmup
- AdamW optimizer with weight decay and gradient clipping

---

## Project structure

```
lafontaine-gpt/
├── Data - Fables/          # 242 fables across 12 books
├── Data - French/          # Wikipedia + classical authors (7.9 GB)
├── tokenizer.rs            # BPE tokenizer trainer in Rust
├── tokenizer.py            # BPE encode/decode in Python
├── dataset.py              # dataloader, calls Rust encoder if needed
├── model.py                # GPT transformer from scratch
├── train.py                # two-phase training loop
├── generate.py             # generate fables from a checkpoint
├── tokenizer.json          # trained vocabulary (32k tokens)
├── pretrain_ids.bin        # encoded pretraining corpus
├── finetune_ids.bin        # encoded finetuning corpus
└── checkpoints/
    ├── pretrain.pt
    └── finetune.pt
```

---

## Quickstart

**1. Compile the Rust tokenizer**
```bash
rustup run stable-x86_64-pc-windows-gnu rustc -O tokenizer.rs -o tokenizer_train.exe
```

**2. Train the BPE vocabulary**
```bash
.\tokenizer_train.exe train --vocab_size 32000 --min_freq 2
```

**3. Train the model**
```bash
python train.py --phase pretrain
python train.py --phase finetune
```
The corpus encoding happens automatically on first run.

**4. Generate fables**
```bash
python generate.py --prompt "Le loup" --tokens 300 --temperature 0.8
```

---

## Results

| Phase | corpus | vocab | best val loss |
|-------|--------|-------|---------------|
| Fables only (BPE 500) | 150k tokens | 500 | 4.05 |
| Fables only (word-level) | 150k tokens | 2,732 | 4.54 |
| Pretrain + finetune (in progress) | 7.9 GB + 242 fables | 32,000 | — |