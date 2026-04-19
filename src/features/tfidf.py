"""
src/features/tfidf.py
----------------------
TF-IDF feature extraction for the Smart Review Analyzer.

الدوال دي بتتكلم مع main.py بالأسماء دي بالظبط:
    - build_tfidf(X_train)
    - transform_tfidf(vectorizer, X)
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


CUSTOM_STOP = ['text', 'found', 'review', 'amazon', 'review text', 'text found']

def build_tfidf(X_train,
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=3,
                max_df=0.90):
    
    # جمعي الـ stopwords
    all_stops = list(stopwords.words("english")) + CUSTOM_STOP

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        stop_words=all_stops 
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    print(f"[TF-IDF] Vocabulary size : {len(vectorizer.vocabulary_):,} terms")
    print(f"[TF-IDF] Train matrix    : {X_train_tfidf.shape}")

    return vectorizer, X_train_tfidf


def transform_tfidf(vectorizer, X):
    """
    Transform a text split using an already-fitted TF-IDF vectorizer.

    Args:
        vectorizer : fitted TfidfVectorizer (from build_tfidf).
        X          : iterable of cleaned text strings (val or test set).

    Returns:
        Sparse matrix of TF-IDF features.
    """
    X_tfidf = vectorizer.transform(X)
    print(f"[TF-IDF] Transformed matrix: {X_tfidf.shape}")
    return X_tfidf


def save_vectorizer(vectorizer, path="outputs/features/tfidf_vectorizer.pkl"):
    """
    Save the fitted TF-IDF vectorizer to disk.
    بنحفظه عشان نقدر نستخدمه تاني من غير ما نعيد التدريب.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[TF-IDF] Vectorizer saved to: {path}")


def load_vectorizer(path="outputs/features/tfidf_vectorizer.pkl"):
    """Load a previously saved TF-IDF vectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[TF-IDF] Vectorizer loaded from: {path}")
    return vectorizer


def get_top_terms(vectorizer, X_tfidf, y, sentiment_value, top_n=15):
    """
    Return the top TF-IDF terms for a given sentiment class.
    مفيدة في الـ insights عشان توضح أهم الكلمات لكل sentiment.

    Args:
        vectorizer      : fitted TfidfVectorizer.
        X_tfidf         : sparse matrix (training split).
        y               : array-like of labels (0 or 1).
        sentiment_value : 0 = Negative | 1 = Positive.
        top_n           : number of top terms to return.

    Returns:
        terms  : list of term strings.
        scores : list of mean TF-IDF scores.
    """
    feature_names = vectorizer.get_feature_names_out()
    y = np.array(y)
    mask = (y == sentiment_value)
    mean_scores = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
    top_idx = mean_scores.argsort()[::-1][:top_n]
    return feature_names[top_idx].tolist(), mean_scores[top_idx].tolist()
