import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Preprocessing
from src.preprocessing.preprocess import preprocess_dataset

# Features
from src.features.tfidf import build_tfidf, transform_tfidf

# Models
from src.models.baseline import train_logistic_regression, predict

# Evaluation
from src.evaluation.metrics import evaluate

# Insights
from src.insights.extractor import extract_keywords, dataset_patterns


# -----------------------------
# 1. Full Training Pipeline
# -----------------------------
def run_full_pipeline(data_path="data/raw/reviews.csv"):
    print("\n[1] Loading dataset...")
    data = pd.read_csv(data_path)

    # -----------------------------
    # STEP 1: Preprocessing
    # -----------------------------
    print("[2] Preprocessing data...")
    data = preprocess_dataset(data)

    # Save processed version
    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/clean_reviews.csv"
    data.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}")

    # -----------------------------
    # STEP 2: Train/Test Split
    # -----------------------------
    print("[3] Splitting data...")
    X = data["Cleaned Review"]
    y = data["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # STEP 3: Feature Extraction (TF-IDF)
    # -----------------------------
    print("[4] Building TF-IDF features...")
    vectorizer, X_train_tfidf = build_tfidf(X_train)
    X_test_tfidf = transform_tfidf(vectorizer, X_test)

    # -----------------------------
    # STEP 4: Train Baseline Model
    # -----------------------------
    #print("[5] Training baseline model (Logistic Regression)...")
    #baseline_model = train_logistic_regression(X_train_tfidf, y_train)

    # -----------------------------
    # STEP 5: Prediction
    # -----------------------------
    print("[6] Evaluating baseline model...")
    print(predict("product amazing quality", method="tfidf"))
    print(predict("bad service terrible", method="w2v"))
    #y_pred = predict(baseline_model, X_test_tfidf)

    # -----------------------------
    # STEP 6: Evaluation
    # -----------------------------
    results = evaluate(y_test, y_pred)

    # -----------------------------
    # STEP 7: Insights Extraction
    # -----------------------------
    print("[7] Extracting insights...")

    # Example: keywords from first few reviews
    sample_texts = data["Full Review"].head(5).tolist()
    for i, text in enumerate(sample_texts):
        keywords = extract_keywords(text)
        print(f"\nReview {i+1} Keywords: {keywords}")

    # Dataset-level patterns
    patterns = dataset_patterns(data["Cleaned Review"], data["Sentiment"])
    print("\nDataset Patterns:")
    print(patterns)

    print("\nPipeline completed successfully.")


# -----------------------------
# 2. Single Review Inference
# -----------------------------
def run_single_inference(review_text, model, vectorizer):
    print("\nRunning inference on single review...")

    # Preprocess
    from src.preprocessing.preprocess import preprocess_text
    cleaned = preprocess_text(review_text)

    # Feature extraction
    X = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(X)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    print(f"\nReview: {review_text}")
    print(f"Predicted Sentiment: {sentiment}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    # -------- Option 1: Run full pipeline --------
    run_full_pipeline("data/raw/reviews.csv")

    # -------- Option 2: Single inference (after training) --------
    # NOTE: You would need to reuse trained model & vectorizer
    # Example placeholder:
    #
    # sample_review = "The product quality is amazing but delivery was slow"
    # run_single_inference(sample_review, model, vectorizer)