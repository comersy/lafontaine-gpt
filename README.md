# lafontaine-gpt

A language model built from scratch to generate La Fontaine fables in classical French.

---

## What this project is

This is a GPT-style transformer trained entirely from scratch — no pretrained weights, no external embeddings, no shortcuts. Just PyTorch, math, and La Fontaine.

The goal is to generate new fables in the style of Jean de La Fontaine, using only his 239 original fables as the target domain.

---

## The problem with training from scratch on fables only

A transformer trained solely on 239 fables (~150k tokens) has to learn everything at once: French grammar, syntax, vocabulary, narrative structure, poetic style. That is too much to ask from too little data.

The best result we got training on fables only:

```
le loup, dit- t'en ne sommes, l'elle en peut- vous me suis point de moi?
le jour par leur plus fait le plus que- nous vous vous ne n'ai de les biens
```

Recognizable words, some La Fontaine characters (loup, renard, lion), but no coherent structure.

---

## The solution: two-phase training

Inspired by how real LLMs are built, we split training into two phases.

**Phase 1 — Pretraining**

Train on a large corpus of 17th century French literature: Molière, Racine, La Bruyère, Bossuet, and other contemporaries of La Fontaine. The model learns French — grammar, syntax, classical vocabulary, narrative structure. It does not need to learn fables yet.

**Phase 2 — Fine-tuning**

Fine-tune exclusively on the 239 La Fontaine fables. The model already knows how to speak classical French. Now it learns the specific style: the moral at the end, the animal characters, the alexandrine verse, the dialogue structure.

This is exactly how GPT, BERT, and other LLMs are built: massive pretraining on general text, then fine-tuning on a specific domain.

---

## Architecture

- Decoder-only transformer (GPT-style)
- Causal masked self-attention
- Word-level tokenizer trained on the corpus
- Cosine learning rate schedule with linear warmup
- AdamW optimizer with weight decay

---

## Project structure

```
lafontaine-gpt/
├── Data - Fables/
│   ├── Livre 1/
│   │   └── *.txt
│   └── ...
├── tokenizer.py      # word-level tokenizer trained from scratch
├── dataset.py        # fables dataloader with train/val split
├── model.py          # GPT transformer from scratch
├── train.py          # training loop with checkpoint and logging
├── generate.py       # generate fables from a trained checkpoint
├── tokenizer.json    # saved tokenizer vocabulary
├── training_log.json # loss curves saved during training
└── checkpoints/
    └── best_model.pt
```

---

## Quickstart

**1. Train the tokenizer**
```bash
python tokenizer.py
```

**2. Train the model**
```bash
python train.py
```

**3. Generate fables**
```bash
python generate.py --prompt "Le loup" --tokens 300 --temperature 0.8
```

---

## Results

| Phase | vocab size | n_layer | n_embd | best val loss |
|-------|-----------|---------|--------|---------------|
| Fables only (BPE 500) | 500 | 4 | 128 | 4.05 |
| Fables only (word-level) | 2732 | 4 | 128 | 4.54 |
| Pretrain + finetune | coming soon | | | |

---

## Next steps

- Collect 17th century French corpus for pretraining (Molière, Racine, Gutenberg)
- Implement two-phase training in `train.py` with `--phase pretrain` and `--phase finetune`
- Evaluate generation quality after fine-tuning