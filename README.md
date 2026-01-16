# GPT Implementation (NanoGPT)

This repository contains a character-level Generative Pre-trained Transformer (GPT) built from scratch in PyTorch.

This project is an educational implementation following Andrej Karpathy's lecture: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Project Goal

The primary objective of this project was to understand the internal mechanics of Large Language Models (LLMs) by building one from the ground up. Rather than using high-level abstractions (like Hugging Face `transformers`), this implementation manually constructs the entire architecture to explore how self-attention and transformers actually work.

## Architecture

The model is a **Decoder-only Transformer** (similar to the original GPT-2 architecture) trained on the **Tiny Shakespeare** dataset.

Key components implemented from scratch include:
* **Token & Positional Embeddings:** Mapping characters to vectors and encoding position.
* **Self-Attention Mechanism:** Implementing the Query, Key, and Value interactions with masking.
* **Multi-Head Attention:** Parallelizing attention to capture different data relationships.
* **Feed-Forward Networks:** Position-wise processing layers.
* **Residual Connections & LayerNorm:** To stabilize deep network training.

## Dataset

* **Source:** Tiny Shakespeare
* **Type:** Character-level text generation
* **Vocabulary:** The unique characters found in the text

## Getting Started

### Prerequisites
* Python 3.x
* PyTorch (with CUDA support if available)

### Part 2: Usage and Credits

### Usage

To train the model and generate text, run the script:
python GPT.py

### Sample Output

The model starts with random gibberish, but after training (approx. 5000 iterations), it begins to generate structure resembling Shakespearean plays:

    Constrary:
    Kie, Rome, Sir John, Wedam, you in an joy,
    The boy?

    KING HENRY VI:
    Go, go the king; they seal alexan-tail.

    CLARENCE:
    So slances they not now their grace wits fruit.

    QUEEN MARGARET:
    O, dear; be my sovereign, as, take it.
    Ah, hearts as dead, a good my husband,
    And kill'd not the camevel: that he rosts sad it is.

### Acknowledgements
All credit for the lesson plan goes to Andrej Karpathy.

    Video: Let's build GPT: from scratch, in code, spelled out.

    Paper: Attention Is All You Need (Vaswani et al., 2017)
