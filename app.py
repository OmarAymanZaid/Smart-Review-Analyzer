# app.py
# ─────────────────────────────────────────────
# Gradio front-end.  Run from the project root:
#   python app.py
#
# Models must already be trained and saved in models/.
# ─────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import gradio as gr

from src.config      import DATA_PROC, SEED
from src.inference   import predict, predict_batch, MODEL_KEYS
from src.insights    import (
    extract_insights,
    make_session,
    update_session,
    build_confusion_matrix_fig,
    build_metrics_comparison_fig,
)
from src.preprocessing import preprocess_text

# ── Load test split (written by notebook 2_model_training.ipynb) ─────────────
_test_csv = DATA_PROC / "test_split.csv"
test_df   = pd.read_csv(_test_csv) if _test_csv.exists() else None

# ── Session state (Single Review tab) ────────────────────────────────────────
session = make_session()


# ── Tab 1: Single Review ──────────────────────────────────────────────────────

def analyze_review(raw_text: str, model_choice: str, true_label_str: str):
    """Classify one review, update session state, return all UI outputs."""
    if not raw_text or not raw_text.strip():
        empty_fig = build_confusion_matrix_fig([], [])
        return "N/A", "N/A", "N/A", "Please enter a review.", "N/A", "N/A", empty_fig

    try:
        pred    = predict(raw_text, model_choice)
        reason, keywords = extract_insights(raw_text, pred)

        true_label_map = {"Positive": 1, "Negative": 0, "Unknown": None}
        true_label     = true_label_map.get(true_label_str)

        cleaned = preprocess_text(raw_text)
        stats   = update_session(session, cleaned, pred, true_label)

        sentiment_label = "Positive ✓" if pred == 1 else "Negative ✗"
        cm_fig = build_confusion_matrix_fig(
            session["all_preds"],
            session["all_labels"],
            title="Confusion Matrix (Session)",
        )

        return (
            sentiment_label,
            reason,
            keywords,
            stats["patterns"],
            stats["stats"],
            stats["correct"],
            cm_fig,
        )

    except Exception as exc:
        empty_fig = build_confusion_matrix_fig([], [])
        return "Error", str(exc), "N/A", "N/A", "N/A", "N/A", empty_fig


# ── Tab 2: Batch Test ─────────────────────────────────────────────────────────

def run_batch_test(model_choice: str):
    """
    Run all three models on a sample of test_df.
    Returns Tab-2 outputs AND the Tab-3 metrics figure.
    """
    if test_df is None or test_df.empty:
        empty_fig = build_confusion_matrix_fig([], [])
        empty_metrics = build_metrics_comparison_fig([], [], [], [])
        return "test_split.csv not found in data/processed/.", pd.DataFrame(), empty_fig, empty_metrics

    sample = test_df.sample(min(200, len(test_df)), random_state=SEED).reset_index(drop=True)

    rows             = []
    batch_preds      = []
    batch_labels     = []
    all_preds_tfidf  = []
    all_preds_w2v    = []
    all_preds_bert   = []
    true_labels_list = []

    for _, row in sample.iterrows():
        raw_text   = str(row.get("text", ""))
        true_label = int(row.get("label", -1))

        if not raw_text.strip():
            continue

        try:
            pred_tfidf = predict(raw_text, "Linear (TF-IDF)")
            pred_w2v   = predict(raw_text, "Linear (W2V)")
            pred_bert  = predict(raw_text, "BERT")

            all_preds_tfidf.append(pred_tfidf)
            all_preds_w2v.append(pred_w2v)
            all_preds_bert.append(pred_bert)
            true_labels_list.append(true_label)

            # Tab-2 uses whichever model the user selected
            pred_chosen = {
                "Linear (TF-IDF)": pred_tfidf,
                "Linear (W2V)":    pred_w2v,
                "BERT":            pred_bert,
            }[model_choice]

            reason, keywords = extract_insights(raw_text, pred_chosen)
            batch_preds.append(pred_chosen)
            batch_labels.append(true_label)

            rows.append({
                "Review":     raw_text[:100] + ("..." if len(raw_text) > 100 else ""),
                "True Label": "Positive" if true_label == 1 else "Negative",
                "Prediction": "Positive" if pred_chosen == 1 else "Negative",
                "Correct":    "✓" if pred_chosen == true_label else "✗",
                "Reason":     reason,
                "Keywords":   keywords,
            })
        except Exception:
            continue

    if not batch_preds:
        empty_fig = build_confusion_matrix_fig([], [])
        empty_metrics = build_metrics_comparison_fig([], [], [], [])
        return "No samples processed.", pd.DataFrame(), empty_fig, empty_metrics

    n          = len(batch_preds)
    n_correct  = sum(p == l for p, l in zip(batch_preds, batch_labels))
    accuracy   = n_correct / n
    summary    = (
        f"Batch test on {n} samples\n"
        f"Correct   : {n_correct}\n"
        f"Incorrect : {n - n_correct}\n"
        f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    cm_fig      = build_confusion_matrix_fig(batch_preds, batch_labels, "Confusion Matrix (Batch)")
    metrics_fig = build_metrics_comparison_fig(
        true_labels_list, all_preds_tfidf, all_preds_w2v, all_preds_bert
    )

    return summary, pd.DataFrame(rows), cm_fig, metrics_fig


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Sentiment Insight System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📊 Sentiment Analysis & Insight Extraction")
    gr.Markdown(
        "Classifies review sentiment as **Positive / Negative** using one of three models "
        "and extracts explainable keywords and session statistics."
    )

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        with gr.TabItem("🔍 Single Review"):
            with gr.Row():
                with gr.Column():
                    review_input = gr.Textbox(
                        label="Review Text", lines=4,
                        placeholder="e.g. The service was slow and delivery took forever.",
                    )
                    model_radio = gr.Radio(
                        list(MODEL_KEYS.keys()),
                        value="Linear (TF-IDF)",
                        label="Model  (Linear = Logistic Regression)",
                    )
                    true_label_radio = gr.Radio(
                        ["Positive", "Negative", "Unknown"],
                        value="Unknown",
                        label="True Label (optional — for accuracy tracking)",
                    )
                    analyze_btn = gr.Button("Extract Insights", variant="primary")

                with gr.Column():
                    out_sentiment = gr.Textbox(label="Sentiment Classification")
                    out_reason    = gr.Textbox(label="Reason")
                    out_keywords  = gr.Textbox(label="Important Keywords")
                    out_correct   = gr.Textbox(label="Correct / Incorrect (Session)")

            gr.Markdown("---")
            with gr.Row():
                out_patterns = gr.Textbox(label="Common Patterns — Top Session Words")
                out_stats    = gr.Textbox(label="Session Statistics (% Positive vs Negative)")

            gr.Markdown("### Session Confusion Matrix")
            out_cm = gr.Plot()

            analyze_btn.click(
                fn=analyze_review,
                inputs=[review_input, model_radio, true_label_radio],
                outputs=[out_sentiment, out_reason, out_keywords,
                         out_patterns, out_stats, out_correct, out_cm],
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        with gr.TabItem("🧪 Batch Test"):
            gr.Markdown(
                "Runs predictions automatically on a sample of `test_split.csv`. "
                "Also populates the Model Comparison tab."
            )
            batch_model_radio = gr.Radio(
                list(MODEL_KEYS.keys()),
                value="Linear (TF-IDF)",
                label="Model for per-review table  (all 3 models are evaluated for comparison)",
            )
            batch_btn    = gr.Button("Run Batch Test", variant="primary")
            batch_output = gr.Textbox(label="Summary", lines=5)
            batch_table  = gr.Dataframe(label="Per-Review Results", interactive=False)
            batch_cm     = gr.Plot(label="Confusion Matrix")

        # ── Tab 3 ─────────────────────────────────────────────────────────────
        with gr.TabItem("📈 Model Comparison"):
            gr.Markdown(
                "Populated automatically when you click **Run Batch Test** in Tab 2."
            )
            metrics_plot = gr.Plot(label="Accuracy · Precision · Recall · F1")

            batch_btn.click(
                fn=run_batch_test,
                inputs=[batch_model_radio],
                outputs=[batch_output, batch_table, batch_cm, metrics_plot],
            )


if __name__ == "__main__":
    demo.launch(share=True)
