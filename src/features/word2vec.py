"""
src/features/word2vec.py
-------------------------
Word2Vec embedding feature extraction for the Smart Review Analyzer.

بيمثل كل مراجعة كـ متوسط متجهات كلماتها (Averaged Word Embeddings).
"""

import os
import numpy as np
from gensim.models import Word2Vec


def build_word2vec(X_train,
                   vector_size=100,
                   window=5,
                   min_count=2,
                   workers=4,
                   sg=1,
                   epochs=10,
                   seed=42):
    """
    Train a Word2Vec model on the training corpus ONLY.

    بيتدرب على X_train بس عشان منعملش data leakage.

    Args:
        X_train     : iterable of cleaned text strings (training set).
        vector_size : number of dimensions per word vector.
        window      : context window size (words before and after).
        min_count   : ignore words appearing fewer than this many times.
        workers     : number of parallel threads.
        sg          : 1 = Skip-gram | 0 = CBOW.
        epochs      : number of training passes over the corpus.
        seed        : random seed for reproducibility.

    Returns:
        Trained gensim Word2Vec model.
    """
    train_tokens = [text.split() for text in X_train]

    model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        seed=seed,
    )

    print(f"[Word2Vec] Vocabulary size     : {len(model.wv):,} words")
    print(f"[Word2Vec] Embedding dimensions: {model.vector_size}")

    return model


def review_to_vec(tokens, model, vector_size=100):
    """
    Convert a list of word tokens to a single averaged vector.

    الكلمات اللي مش موجودة في الـ vocabulary بيتم تجاهلها.
    لو المراجعة كلها كلمات غير موجودة بيرجع zero vector.

    Args:
        tokens      : list of word strings.
        model       : trained Word2Vec model.
        vector_size : fallback vector size for OOV reviews.

    Returns:
        numpy array of shape (vector_size,).
    """
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)


def transform_word2vec(X, model):
    """
    Transform a text split into averaged Word2Vec embedding matrix.

    بتحوّل كل مراجعة لـ vector واحد بعمل متوسط لكل كلماتها.

    Args:
        X     : iterable of cleaned text strings.
        model : trained Word2Vec model.

    Returns:
        numpy array of shape (n_samples, vector_size).
    """
    size = model.vector_size
    embeddings = np.array([review_to_vec(text.split(), model, size) for text in X])
    print(f"[Word2Vec] Embedding matrix: {embeddings.shape}")
    return embeddings


def save_word2vec(model, X_train_w2v, X_test_w2v,
                  out_dir="outputs/features"):
    """
    Save Word2Vec model and embedding arrays to disk.

    Args:
        model       : trained Word2Vec model.
        X_train_w2v : numpy array of training embeddings.
        X_test_w2v  : numpy array of test embeddings.
        out_dir     : directory to save files.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.save(os.path.join(out_dir, "word2vec.model"))
    np.save(os.path.join(out_dir, "X_train_w2v.npy"), X_train_w2v)
    np.save(os.path.join(out_dir, "X_test_w2v.npy"),  X_test_w2v)

    print(f"[Word2Vec] Model and embeddings saved to: {out_dir}")


def load_word2vec(out_dir="outputs/features"):
    """
    Load a previously saved Word2Vec model and embedding arrays.

    Returns:
        model       : trained Word2Vec model.
        X_train_w2v : numpy array of training embeddings.
        X_test_w2v  : numpy array of test embeddings.
    """
    model       = Word2Vec.load(os.path.join(out_dir, "word2vec.model"))
    X_train_w2v = np.load(os.path.join(out_dir, "X_train_w2v.npy"))
    X_test_w2v  = np.load(os.path.join(out_dir, "X_test_w2v.npy"))

    print(f"[Word2Vec] Loaded from: {out_dir}")
    return model, X_train_w2v, X_test_w2v


def get_similar_words(model, word, topn=5):
    """
    Return the most similar words to a given word.
    مفيدة لاختبار إن الـ model اتعلم صح.

    Args:
        model : trained Word2Vec model.
        word  : query word string.
        topn  : number of similar words to return.

    Returns:
        List of (word, similarity_score) tuples,
        or empty list if word not in vocabulary.
    """
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    else:
        print(f"[Word2Vec] '{word}' not in vocabulary.")
        return []
