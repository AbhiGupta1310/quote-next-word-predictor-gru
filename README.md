# Next-Word Predictor (Quote Generator)

A compact, reproducible project that trains a GRU-based next-word / quote generator from a quotes dataset and provides a Streamlit demo for inference.

This README summarizes the notebook implementation, the `app.py` inference logic, how i handled challenges during development, and how to run or reproduce the work locally.

---

## Project summary

The model is an Embedding → GRU → Dense (softmax) architecture trained to predict the next token given a seed sequence. The notebook (`next-word-predictor.ipynb`) contains the full data cleaning, tokenizer preparation, sequence generation, training, and artifact saving steps. The Streamlit app (`app.py`) loads the saved artifacts and exposes a simple UI to generate text using top-k + temperature sampling.

---

## Files & layout (current)

- `next-word-predictor.ipynb` — training pipeline (data cleaning → tokenizer → sequences → model → train → save)
- `app.py` — Streamlit inference app (loads artifacts from `model/` and `json/`)
- `data/Quote_data.csv` — source quotes dataset used by the notebook
- `model/quote_generator_model_3.keras` — trained model for inference
- `model/best_quote_model.keras` — checkpoint saved during training
- `model/tokenizer_3.pkl` — saved Tokenizer
- `json/model_params_3.json` — saved parameters (e.g., `max_len`)
- `json/training_history_3.json` — training metrics
- `requirements.txt` — pinned dependencies for running the project

---

## Quickstart (run demo locally)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run app.py
```

4. In the browser UI, enter a seed phrase and press **Predict Next Word of Quote**.

Note: `app.py` expects artifacts under `model/` and `json/` (for example `model/tokenizer_3.pkl`, `model/quote_generator_model_3.keras`, `json/model_params_3.json`).

---

## How the notebook works (high level, by section)

- Data load & cleaning (cell 2 → cell ~15): load `data/Quote_data.csv`, drop nulls, lowercase, remove punctuation, filter quotes by length (keep 4–35 words) to reduce noise.
- Tokenizer & vocabulary (cells ~16–22): fit a `Tokenizer` with `oov_token='<UNK>'`, limit vocab to `vocab_size` (8k), create sequences.
- Supervised pairs (cells ~23–30): build (input_seq → next_token) pairs and pad inputs to `max_len`.
- Model (cells ~31–34): Embedding + GRU(256) → Dropout → Dense with softmax. Compiled with Adam and `sparse_categorical_crossentropy`.
- Callbacks & training (cells ~34–37): `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`; training on GPU recommended (notebook notes targeted P100).
- Save artifacts (cell ~37): save model, tokenizer, `model_params_3.json` (stores `max_len`), and training history.

---

## Inference details (the logic in `app.py`)

- Load artifacts: model, tokenizer, and `max_len` from `json/`.
- Reverse index: create `reverse_word_index` from `tokenizer.word_index`.
- Generation loop:
  - Tokenize seed and pad to `max_len`.
  - Get model probabilities and zero out reserved tokens (`<PAD>` and `<UNK>`).
  - Renormalize probabilities and select top-k indices.
  - Apply temperature to top-k probabilities and sample the next token.
  - Append the sampled token and repeat for `next_words` steps.

Sampling parameters you can tune programmatically: `next_words`, `temperature` (0.5 safe → 1.5 creative), `top_k`.

---

## Challenges i faced and how i tackled them

- Noisy quotes and huge length variance — mitigated by filtering quotes to keep only those with 4–35 words and resetting index, reducing outliers and training noise.
- Variable-length sequences — handled by creating many incremental (prefix → next token) pairs and padding inputs to a single `max_len` used at inference (padding='pre').
- Out-of-vocabulary tokens — used a dedicated `oov_token='<UNK>'` in the `Tokenizer` and filtered `<UNK>` at inference time.
- Repetitive / low-quality sampling from argmax — improved generation quality using top-k sampling plus temperature scaling instead of greedy selection.
- Overfitting / unstable training — added `Dropout`, `ModelCheckpoint`, `EarlyStopping` with `restore_best_weights=True`, and `ReduceLROnPlateau` to stabilize training and retain the best model.
- Long training time and GPU requirement — noted in the notebook (P100 target). For CPU-only runs, reduce batch size/model size or train on a subset for fast experimentation.
- Model / TF version mismatch risk — saved `model_params_3.json` and artifacts; recommended using TF 2.x and matching training environment when loading models.

These measures collectively improved robustness and inference quality while keeping the pipeline reproducible.

---

## Reproduce training (step-by-step)

1. Open `next-word-predictor.ipynb` and run cells in order. Key phases: data cleaning → tokenizer → sequences → model definition → callbacks → train → save.
2. Ensure GPU if available (training time will be significantly longer on CPU).
3. After training, confirm saved artifacts appear under `model/` and `json/`.

---

## Troubleshooting quick list

- Missing files on startup: confirm `model/tokenizer_3.pkl`, `model/quote_generator_model_3.keras`, and `json/model_params_3.json` exist. Update paths in `app.py` if you store artifacts elsewhere.
- TensorFlow/Keras load errors: try a matching TF 2.x version to that used during training or recreate the model architecture and load weights instead.
- Poor generation quality: reduce `top_k` and lower `temperature` for safer outputs; retrain with more data or stronger regularization if needed.
- Slow inference on CPU: reduce model size or deploy the model to a GPU-backed server.

---

## Requirements

Install from the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

---
