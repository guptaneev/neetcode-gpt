# neetcode-gpt

A decoder-only Transformer built from scratch — no `nn.Transformer`, no Hugging Face.

This project implements tokenization, embeddings, scaled dot-product attention,
multi-head attention, positional encoding, layer normalization, and a full
training loop, assembled from my NeetCode ML course submissions into one
end-to-end working model.

## What's here

- `tokenizer.py` — Builds/loads vocabulary and converts text ↔ token IDs.
- `model.py` — Core GPT architecture:
  - token + positional embeddings
  - masked multi-head self-attention
  - feed-forward network
  - residual connections + layer norm
- `attention.py` — Scaled dot-product attention and causal masking logic.
- `train.py` — Data pipeline, batching, optimization loop, checkpointing.
- `generate.py` — Autoregressive text generation from a prompt.
- `config.py` — Hyperparameters (context length, heads, layers, LR, etc.).
- `utils.py` — Helper functions (mask creation, seed setup, logging).

> If your file names differ, keep this section structure and rename entries to match your repo.

## Run it

```bash
python train.py
```

(Optional generation step after training)

```bash
python generate.py --prompt "Once upon a time"
```
