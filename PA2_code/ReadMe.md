# CSE 256 – PA2: Transformer Implementation

This project implements a Transformer-based model for:

- **Part 1**: Encoder + Classifier (Speech Classification)
- **Part 2**: Decoder Language Model (Next Word Prediction)
- **Part 3**: Positional Encoding Exploration

---

## 1. Project Structure

- `main.py` — Entry point for running all experiments  
- `transformer.py` — Implementation of Encoder and Decoder  
- `dataset.py` — Dataset classes  
- `tokenizer.py` — Simple tokenizer  
- `utilities.py` — Sanity check utilities  
- `speechesdataset/` — Dataset directory  

---

## 2. Running the Code

All experiments are controlled via `main.py`.

---

## Part 1: Encoder + Classifier

Train the encoder and classifier jointly:

```bash
python main.py --run part1
```

Run attention sanity check:

```bash
python main.py --run part1 --sanity_check
```

This will:
- Train for 15 epochs
- Report training and test accuracy
- Print encoder parameter count

---

## Part 2: Decoder Language Model

Run language model pretraining:

```bash
python main.py --run part2
```

Run decoder sanity check:

```bash
python main.py --run part2 --sanity_check
```

This will:
- Train for 500 iterations (default)
- Print perplexity every 100 iterations
- Report final perplexity on:
  - Training set
  - Obama test set
  - W. Bush test set
  - H. Bush test set

---

## Change Training Iterations

To run longer training (e.g., 2000 iterations):

```bash
python main.py --run part2 --max_iters 2000
```

---

## Part 3: Positional Encoding Exploration

Baseline (learned positional encoding):

```bash
python main.py --run part2 --pos_encoding learned
```

No positional encoding:

```bash
python main.py --run part2 --pos_encoding none
```

AliBi positional encoding:

```bash
python main.py --run part2 --pos_encoding alibi
```

---

## 3. Default Hyperparameters

- `batch_size = 16`
- `block_size = 32`
- `n_embd = 64`
- `n_head = 2`
- `n_layer = 4`
- `ffn_hidden = 100`
- `dropout = 0.1`

You may override them via command-line arguments.

Example:

```bash
python main.py --run part2 --batch_size 32 --lr 5e-4
```

---

## 4. Outputs

During training, the program prints:

- Loss
- Accuracy (Part 1)
- Perplexity (Part 2 & 3)
- Parameter counts

When `--sanity_check` is enabled, attention heatmaps will be generated for visualization.

---

## P.S. 
The .zip file includes screenshots of the results of a particular run for each part.