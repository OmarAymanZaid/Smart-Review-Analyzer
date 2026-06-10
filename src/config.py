# src/config.py
# ─────────────────────────────────────────────
# Centralized configuration for the entire project.
# Import this module anywhere you need a shared constant.
# ─────────────────────────────────────────────

from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # project root
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"

CLEAN_CSV   = DATA_PROC / "clean_reviews.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Train / Test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.2

# ── DistilBERT hyper-parameters ───────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LEN    = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS     = 3
BERT_LR         = 2e-5

# ── TF-IDF hyper-parameters ───────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000

# ── Word2Vec hyper-parameters ─────────────────────────────────────────────────
W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 1
W2V_WORKERS     = 4

# ── Logistic Regression ───────────────────────────────────────────────────────
LR_MAX_ITER = 1000

# ── Insights: sentiment keyword lists ────────────────────────────────────────
INSIGHT_KEYWORDS = {
    "negative": ["slow", "bad", "terrible", "refund", "broken", "worst",
                 "expensive", "late", "poor", "awful", "disappointed",
                 "damaged", "missing", "lost", "cancel", "delay"],
    "positive": ["great", "fast", "love", "excellent", "amazing", "best",
                 "good", "cheap", "helpful", "easy", "perfect", "happy",
                 "satisfied", "recommend"],
}
