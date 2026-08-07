# neetcode-gpt

An educational, decoder-only Transformer/GPT implementation built from scratch.

## Overview

`neetcode-gpt` consolidates NeetCode ML course exercises into a single repository that walks through core GPT building blocks without relying on `nn.Transformer` or Hugging Face abstractions. The focus is learning how each component works and how they connect into an end-to-end training and text-generation workflow.

## Features

Implemented components in this repository include:

- Tokenization utilities (character vocab + merge/token counting helpers)
- Embedding lookup
- Scaled dot-product (causal) attention
- Multi-head self-attention
- Positional encoding
- Layer normalization variants
- Decoder-style Transformer blocks
- GPT model assembly and projection to vocabulary logits
- Training loop with AdamW + cross-entropy
- Autoregressive token generation

## Requirements

This repository does not currently include a dependency lockfile. The code uses:

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [torchtyping](https://github.com/patrick-kidger/torchtyping)

Example install:

```bash
pip install torch numpy torchtyping
```

## Usage

There is no packaged CLI yet; modules are designed to be imported and run from Python scripts or notebooks.

### Train a model

```python
import torch
from model.gpt import GPT
from train import Solution as TrainSolution

# Example only: replace with your encoded token tensor
data = torch.randint(0, 65, (5000,))

model = GPT(
    vocab_size=65,
    context_length=32,
    model_dim=64,
    num_blocks=2,
    num_heads=4,
)

trainer = TrainSolution()
final_loss = trainer.train(
    model=model,
    data=data,
    epochs=200,
    context_length=32,
    batch_size=16,
    lr=3e-4,
)
print("Final loss:", final_loss)
```

### Generate text

```python
import torch
from generate import Solution as GenerateSolution

# int_to_char should match your training vocabulary
int_to_char = {i: chr(65 + i % 26) for i in range(65)}
context = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

generator = GenerateSolution()
out = generator.generate(
    model=model,
    new_chars=40,
    context=context,
    context_length=32,
    int_to_char=int_to_char,
)
print(out)
```

## Project Structure

- `data/` — tokenization, vocab, preprocessing, and batch/dataset helpers
- `model/` — attention, normalization, positional encoding, Transformer, and GPT modules
- `train.py` — training loop implementation
- `generate.py` — autoregressive generation loop implementation
- `foundations/` — supporting ML exercises and building-block implementations

## Status / Future Work

Current status: core educational GPT components are implemented and organized in one repository.

Potential next steps:

- Add a reproducible end-to-end training script/CLI
- Add dataset preparation examples and checkpoints
- Add automated tests and benchmark/evaluation utilities
