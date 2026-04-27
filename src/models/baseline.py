import numpy as np
import pandas as pd
import pickle

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing.preprocess import preprocess_text

def predict_tfidf(cleaned_review, model, vectorizer):
    # vectorize
    vec = vectorizer.transform([cleaned_review])
    
    # predict
    pred = model.predict(vec)[0]
    
    return pred


def get_w2v_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def predict_w2v(cleaned_review, model, w2v_model, vector_size=100):
    tokens = cleaned_review.split()
    
    # vectorize
    vec = get_w2v_vector(tokens, w2v_model, vector_size).reshape(1, -1)
    
    # predict
    pred = model.predict(vec)[0]
    
    return pred

def predict(cleaned_review, method="tfidf"):
    if method == "tfidf":
        return predict_tfidf(cleaned_review, lr_tfidf, tfidf_vectorizer)
    elif method == "w2v":
        return predict_w2v(cleaned_review, lr_w2v, w2v_model)