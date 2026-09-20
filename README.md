# Transformer Language Model From Scratch

A portfolio project for implementing the core components of a decoder-only Transformer language model in **PyTorch** — from tokenization and embeddings through causal self-attention, Transformer blocks, training, and text generation.

> **Status:** In development. The repository is being built incrementally so that each architectural component can be understood, tested, and evaluated independently.

## Why This Project

Modern LLM APIs make powerful models easy to use, but understanding what happens beneath the API layer is valuable for AI engineering. This project focuses on implementing the fundamental Transformer pipeline rather than wrapping an existing hosted LLM.

## Target Architecture

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embeddings + Positional Embeddings
   ↓
Masked Multi-Head Self-Attention
   ↓
Feed-Forward Network
   ↓
Transformer Blocks
   ↓
Layer Normalization
   ↓
Linear Language-Model Head
   ↓
Next-Token Probabilities
   ↓
Autoregressive Text Generation
```

## Planned Implementation

- Text preprocessing and vocabulary construction
- Tokenization and sequence batching
- Token and positional embeddings
- Scaled dot-product attention
- Causal masking
- Multi-head self-attention
- Feed-forward network
- Residual connections and layer normalization
- Stacked Transformer blocks
- Next-token prediction objective
- Training and validation loops
- Autoregressive text generation
- Checkpoint save/load
- Basic evaluation and training visualizations

## Tech Stack

**Language:** Python  
**Deep Learning:** PyTorch  
**Data:** NumPy / Python text processing  
**Visualization:** Matplotlib  
**Environment:** Jupyter / Google Colab or local Python

## Intended Repository Structure

```text
LLM_Creation/
├── README.md
├── src/
│   ├── tokenizer.py
│   ├── attention.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── notebooks/
│   └── transformer-from-scratch.ipynb
├── tests/
│   ├── test_tokenizer.py
│   └── test_attention.py
├── data/
│   └── README.md
├── requirements.txt
└── .gitignore
```

The structure above represents the target organization as implementation is added.

## Engineering Goals

The finished project should demonstrate more than a working notebook. The goal is to show understanding of:

1. **Tensor shapes and data flow** through a Transformer.
2. **Why causal masking is required** for autoregressive language modeling.
3. **How attention weights are computed** and combined across heads.
4. **How residual connections, normalization, and feed-forward layers interact.**
5. **How a language model is trained** using next-token prediction.
6. **How inference differs from training** during autoregressive generation.
7. **How model configuration affects parameter count and compute requirements.**

## Evaluation Plan

Training progress will be evaluated using training/validation loss and perplexity where appropriate. Generated samples will be used as qualitative checks, while unit tests will validate critical components such as masking and tensor dimensions.

## Scope

This is intentionally a small educational Transformer rather than an attempt to reproduce a production-scale foundation model. Training will use a manageable text corpus and model size suitable for experimentation on consumer hardware or a hosted notebook environment.

## Roadmap

- [ ] Implement tokenizer and dataset pipeline
- [ ] Implement embeddings and positional encoding
- [ ] Implement causal self-attention
- [ ] Implement multi-head attention
- [ ] Build Transformer block
- [ ] Assemble decoder-only language model
- [ ] Add training and validation pipeline
- [ ] Add text generation
- [ ] Add unit tests
- [ ] Add architecture visualization
- [ ] Document experiment results

## Portfolio Context

This project complements my applied AI work in RAG, AI agents, MCP-based systems, conversational memory, and AI-assisted analytics by exploring the model architecture underneath modern generative-AI applications.
