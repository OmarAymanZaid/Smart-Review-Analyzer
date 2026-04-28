import gradio as gr
import numpy as np
import torch
import warnings
from collections import Counter
import re

warnings.filterwarnings("ignore")

# Ensure BERT model is in evaluation mode
if 'model' in globals():
    model.eval()

# Global session tracker for statistics and common patterns
session = {
    'total': 0, 
    'positive': 0, 
    'negative': 0, 
    'all_tokens': []
}

# Define a list of sentiment-heavy keywords for "Reason" extraction
INSIGHT_KEYWORDS = {
    'negative': ['slow', 'bad', 'terrible', 'refund', 'broken', 'worst', 'expensive', 'late', 'poor'],
    'positive': ['great', 'fast', 'love', 'excellent', 'amazing', 'best', 'good', 'cheap', 'helpful']
}

def extract_insights(text, label_idx):
    """Extracts the reason and important keywords from the text."""
    text = text.lower()
    sentiment = 'positive' if label_idx == 1 else 'negative'
    
    # Identify important keywords mentioned in the text
    found_keywords = [word for word in INSIGHT_KEYWORDS[sentiment] if word in text]
    
    # Generate the "Reason" based on found keywords or general tone
    if found_keywords:
        reason = f"{sentiment} {found_keywords[0]} mentioned"
    else:
        reason = f"General {sentiment} tone detected"
        
    return reason, ", ".join(found_keywords) if found_keywords else "N/A"

def analyze_review(review_text, model_choice):
    if not review_text or not str(review_text).strip():
        return "N/A", "N/A", "N/A", "N/A", "Please enter a review."

    try:
        # Preprocessing (assumes preprocess_text is defined in your notebook)
        cleaned_review = preprocess_text(str(review_text))
        
        # 1. Classify Sentiment
        if model_choice == "BERT" and 'model' in globals():
            encoding = tokenizer(cleaned_review, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                pred = np.argmax(probs)
        elif model_choice == "TF-IDF":
            features = tfidf_vectorizer.transform([cleaned_review])
            pred = lr_tfidf.predict(features)[0]
            probs = lr_tfidf.predict_proba(features)[0]
        else: # W2V fallback
            features = np.array([get_sentence_vector(cleaned_review.split(), w2v_model)])
            pred = lr_w2v.predict(features)[0]
            probs = lr_w2v.predict_proba(features)[0]

        # 2. Extract Reason and Keywords
        reason, keywords = extract_insights(review_text, pred)
        
        # 3. Update Session Statistics
        session['total'] += 1
        if pred == 1:
            session['positive'] += 1
        else:
            session['negative'] += 1
        
        # Track words for "Common Patterns" (simple tokenization)
        words = re.findall(r'\w+', cleaned_review.lower())
        session['all_tokens'].extend(words)
        
        # Calculate Statistics and Patterns
        pos_pct = (session['positive'] / session['total']) * 100
        neg_pct = (session['negative'] / session['total']) * 100
        stats_str = f"Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% (Total: {session['total']})"
        
        common_words = [w for w, c in Counter(session['all_tokens']).most_common(5)]
        patterns = ", ".join(common_words)

        sentiment_label = "Positive (✓)" if pred == 1 else "Negative (✘)"
        
        return sentiment_label, reason, keywords, patterns, stats_str

    except Exception as e:
        return "Error", str(e), "N/A", "N/A", "N/A"

# Build Gradio UI
with gr.Blocks(title="Sentiment Insight System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Sentiment Analysis & Insight Extraction")
    gr.Markdown("Builds a system that classifies sentiment and extracts reasons behind predictions.")

    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(label="Review Text", lines=4, placeholder="e.g., The service was slow")
            model_radio = gr.Radio(["TF-IDF", "W2V", "BERT"], value="TF-IDF", label="Model Selection")
            analyze_btn = gr.Button("Extract Insights", variant="primary")

        with gr.Column():
            out_sentiment = gr.Textbox(label="Sentiment Classification")
            out_reason = gr.Textbox(label="Reason")
            out_keywords = gr.Textbox(label="Important Keywords")
    
    gr.Markdown("---")
    with gr.Row():
        out_patterns = gr.Textbox(label="Common Patterns (Top Words Across Session)")
        out_stats = gr.Textbox(label="Simple Statistics (% Positive vs Negative)")

    analyze_btn.click(
        fn=analyze_review,
        inputs=[review_input, model_radio],
        outputs=[out_sentiment, out_reason, out_keywords, out_patterns, out_stats]
    )

demo.launch(share=False)