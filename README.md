# neetcode-gpt [In Progress]

A decoder-only Transformer built from scratch — no `nn.Transformer`, no Hugging Face.

This project implements tokenization, embeddings, scaled dot-product attention,
multi-head attention, positional encoding, layer normalization, and a full
training loop, assembled from my NeetCode ML course submissions into one
end-to-end working model.

## Run it

```bash
python train.py
```

(Optional generation step after training)

```bash
python generate.py --prompt "Once upon a time"
```
