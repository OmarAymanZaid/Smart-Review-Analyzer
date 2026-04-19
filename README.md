# Smart Review Analyzer

A Natural Language Processing (NLP) system that analyzes user reviews, classifies sentiment (positive/negative), and extracts meaningful insights such as keywords, reasons, and common patterns.

---

## 📌 Project Overview

This project builds an end-to-end pipeline for analyzing textual reviews. It goes beyond simple sentiment classification by providing **explainable outputs**, including:

* Sentiment prediction (Positive / Negative)
* Key keywords influencing the decision
* Extracted reasons behind sentiment
* Common patterns across multiple reviews
* Dataset-level statistics

---

## 🚀 Features

* Text preprocessing (cleaning, tokenization, stopword removal)
* Feature extraction using:

  * TF-IDF (required)
  * Word embeddings (Word2Vec / GloVe / BERT)
* Two modeling approaches:

  * Baseline model (Logistic Regression / Naive Bayes)
  * Advanced model (LSTM / Transformer such as BERT)
* Model evaluation and comparison
* Insight extraction (keywords, reasons, patterns, statistics)

---

## 🧠 System Pipeline

```
Raw Reviews
   ↓
Preprocessing
   ↓
Feature Extraction (TF-IDF + Embeddings)
   ↓
Model Training (Baseline + Advanced)
   ↓
Evaluation & Comparison
   ↓
Insights Extraction
   ↓
Final Output (Sentiment + Explanation)
```

---

## 📂 Project Structure

```
smart-review-analyzer/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── insights/
│   └── utils/
│
├── configs/
│   └── config.yaml
│
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/smart-review-analyzer.git
cd smart-review-analyzer
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the full pipeline:

```bash
python main.py
```

---

## 🧹 Preprocessing Steps

* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization

---

## 🔍 Feature Extraction

* **TF-IDF**: Captures word importance
* **Embeddings**:

  * Word2Vec / GloVe (static embeddings)
  * BERT (contextual embeddings)

---

## 🤖 Models

### 🔹 Baseline Model

* Logistic Regression or Naive Bayes
* Fast and interpretable

### 🔹 Advanced Model

* LSTM or Transformer (BERT)
* Captures contextual meaning and sequence information

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📈 Insights Extraction

The system extracts:

* **Keywords** (important words influencing sentiment)
* **Reasons** (phrases explaining predictions)
* **Common patterns** across reviews
* **Statistics**:

  * % Positive vs Negative
  * Most frequent complaints
  * Most praised aspects

---

## 🧪 Example Output

**Input Review:**

```
"The service was slow but the food was great"
```

**Output:**

```
Sentiment: Negative
Reason: slow service
Keywords: slow, service
```

---

## 📊 Results & Comparison

| Model          | Accuracy | Notes                        |
| -------------- | -------- | ---------------------------- |
| Baseline Model | TBD      | Fast, simple                 |
| Advanced Model | TBD      | More accurate, context-aware |

---

## 📝 Report

The final report includes:

* System description
* Dataset details
* Preprocessing techniques
* Models used
* Results and comparison
* Key findings and conclusions

---

## 📌 Future Improvements

* Improve reason extraction using attention mechanisms
* Add aspect-based sentiment analysis
* Deploy as a web application
* Use larger transformer models for better accuracy

---

## 👥 Team

* Your Name(s)

---

## 📜 License

This project is for academic purposes.
