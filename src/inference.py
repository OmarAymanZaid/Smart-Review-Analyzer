# src/inference.py
# ─────────────────────────────────────────────
# Loads trained models from disk once at import time.
# Exposes a single predict() function used by app.py.
# ─────────────────────────────────────────────

import pickle

import numpy as np
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from src.config import (
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    DEVICE,
    MODELS_DIR,
)
from src.preprocessing import preprocess_text


# ── Load models once ──────────────────────────────────────────────────────────

def _load_pickle(filename: str):
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


# TF-IDF
tfidf_vectorizer = _load_pickle("tfidf_vectorizer.pkl")
lr_tfidf         = _load_pickle("lr_tfidf.pkl")

# Word2Vec
import gensim
w2v_model = gensim.models.Word2Vec.load(str(MODELS_DIR / "w2v_model.gensim"))
lr_w2v    = _load_pickle("lr_w2v.pkl")

# DistilBERT
_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
_bert      = DistilBertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=2
)
_bert.load_state_dict(
    torch.load(MODELS_DIR / "bert_best.pt", map_location=DEVICE)
)
_bert = _bert.to(DEVICE)
_bert.eval()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sentence_vector(words: list[str]) -> np.ndarray:
    """Average Word2Vec embeddings; return zero vector for OOV sentences."""
    valid = [w for w in words if w in w2v_model.wv]
    if valid:
        return np.mean(w2v_model.wv[valid], axis=0)
    return np.zeros(w2v_model.vector_size)


def _predict_bert(cleaned_text: str) -> int:
    encoding = _tokenizer(
        cleaned_text,
        max_length=BERT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = _bert(input_ids=input_ids, attention_mask=attention_mask).logits
    return int(torch.argmax(logits, dim=1).item())


def _predict_tfidf(cleaned_text: str) -> int:
    features = tfidf_vectorizer.transform([cleaned_text])
    return int(lr_tfidf.predict(features)[0])


def _predict_w2v(cleaned_text: str) -> int:
    vec      = _get_sentence_vector(cleaned_text.split())
    features = vec.reshape(1, -1)
    return int(lr_w2v.predict(features)[0])


# ── Public API ────────────────────────────────────────────────────────────────

MODEL_KEYS = {
    "BERT":             _predict_bert,
    "Linear (TF-IDF)":  _predict_tfidf,
    "Linear (W2V)":     _predict_w2v,
}


def predict(raw_text: str, model_choice: str) -> int:
    """
    Classify a raw review string.

    Parameters
    ----------
    raw_text     : original (uncleaned) review text
    model_choice : one of 'BERT', 'Linear (TF-IDF)', 'Linear (W2V)'

    Returns
    -------
    0 (Negative) or 1 (Positive)
    """
    if model_choice not in MODEL_KEYS:
        raise ValueError(f"Unknown model: {model_choice!r}. "
                         f"Choose from {list(MODEL_KEYS)}")
    cleaned = preprocess_text(raw_text)
    return MODEL_KEYS[model_choice](cleaned)


def predict_batch(texts: list[str], model_choice: str) -> list[int]:
    """Run predict() over a list of raw texts."""
    return [predict(t, model_choice) for t in texts]
