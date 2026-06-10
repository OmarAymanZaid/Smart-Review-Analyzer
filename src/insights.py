# src/insights.py
# ─────────────────────────────────────────────
# Insight extraction helpers used by app.py.
# Handles: reason/keyword extraction, session state,
#          confusion matrix figures, and metrics computation.
# ─────────────────────────────────────────────

import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import INSIGHT_KEYWORDS


# ── Reason & keyword extraction ───────────────────────────────────────────────

def extract_insights(raw_text: str, pred: int) -> tuple[str, str]:
    """
    Return (reason, keywords_string) for a prediction.

    Parameters
    ----------
    raw_text : original review text (not cleaned)
    pred     : 0 = Negative, 1 = Positive
    """
    text      = raw_text.lower()
    sentiment = "positive" if pred == 1 else "negative"
    found     = [w for w in INSIGHT_KEYWORDS[sentiment] if w in text]

    if found:
        reason   = f"{sentiment.capitalize()} keyword '{found[0]}' detected"
        keywords = ", ".join(found)
    else:
        reason   = f"General {sentiment} tone detected"
        keywords = "N/A"

    return reason, keywords


# ── Session state ─────────────────────────────────────────────────────────────

def make_session() -> dict:
    """Return a fresh session-state dict for the Single Review tab."""
    return {
        "total":      0,
        "positive":   0,
        "negative":   0,
        "all_tokens": [],
        "correct":    0,
        "incorrect":  0,
        "all_preds":  [],
        "all_labels": [],
    }


def update_session(
    session: dict,
    cleaned_text: str,
    pred: int,
    true_label,          # int (0/1) or None
) -> dict:
    """
    Update session counters in-place and return a dict of formatted strings.

    Returns
    -------
    {
        "stats":    "Positive: X% | Negative: Y% (Total: N)",
        "patterns": "word1, word2, ...",
        "correct":  "Correct: X | Incorrect: Y" or note string,
    }
    """
    session["total"]    += 1
    session["all_preds"].append(pred)

    if pred == 1:
        session["positive"] += 1
    else:
        session["negative"] += 1

    session["all_tokens"].extend(re.findall(r"\w+", cleaned_text.lower()))

    if true_label is not None:
        session["all_labels"].append(true_label)
        if pred == true_label:
            session["correct"] += 1
        else:
            session["incorrect"] += 1
        correct_str = (
            f"Correct: {session['correct']} | "
            f"Incorrect: {session['incorrect']}"
        )
    else:
        session["all_labels"].append(pred)  # treat pred as ground truth
        correct_str = "True label not provided — accuracy not tracked"

    total   = session["total"]
    pos_pct = session["positive"] / total * 100
    neg_pct = session["negative"] / total * 100
    stats_str = (
        f"Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% "
        f"(Total: {total})"
    )

    top_words   = [w for w, _ in Counter(session["all_tokens"]).most_common(5)]
    patterns_str = ", ".join(top_words) if top_words else "N/A"

    return {
        "stats":    stats_str,
        "patterns": patterns_str,
        "correct":  correct_str,
    }


# ── Confusion matrix figure ───────────────────────────────────────────────────

def build_confusion_matrix_fig(
    preds: list[int],
    labels: list[int],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Return a matplotlib Figure with the confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 3))

    if len(preds) < 2:
        ax.text(
            0.5, 0.5, "Not enough data\nfor confusion matrix",
            ha="center", va="center", fontsize=11
        )
        ax.axis("off")
        return fig

    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Model-comparison metrics figure ──────────────────────────────────────────

def build_metrics_comparison_fig(
    true_labels: list[int],
    preds_tfidf: list[int],
    preds_w2v:   list[int],
    preds_bert:  list[int],
) -> plt.Figure:
    """
    Return a 2×2 bar-chart figure comparing Accuracy, Precision, Recall, F1
    across the three models.
    """
    model_names = ["TF-IDF", "Word2Vec", "BERT"]
    all_preds   = [preds_tfidf, preds_w2v, preds_bert]

    accs, precs, recs, f1s = [], [], [], []
    for preds in all_preds:
        accs.append(accuracy_score(true_labels, preds))
        precs.append(precision_score(true_labels, preds, zero_division=0))
        recs.append(recall_score(true_labels, preds, zero_division=0))
        f1s.append(f1_score(true_labels, preds, zero_division=0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = np.arange(len(model_names))

    metrics_config = [
        (accs,  "Accuracy",  "#4C72B0", axes[0, 0]),
        (precs, "Precision", "#55A868", axes[0, 1]),
        (recs,  "Recall",    "#DD8452", axes[1, 0]),
        (f1s,   "F1-Score",  "#C44E52", axes[1, 1]),
    ]

    for values, title, color, ax in metrics_config:
        bars = ax.bar(x, values, color=color, width=0.5)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontweight="bold", fontsize=11)
        ax.set_ylim([0, 1.15])
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    return fig
