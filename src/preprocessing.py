import re
import string
import pandas as pd

# Optional NLP tools
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


# -----------------------------
# Text Cleaning Functions
# -----------------------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords
    """
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered_tokens)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


# -----------------------------
# Dataset-Level Processing
# -----------------------------

def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to dataset.

    Expected columns:
    - 'Review Title'
    - 'Review Text'
    - 'Rating'

    Output:
    - Adds 'Full Review'
    - Adds 'Sentiment'
    - Adds 'Cleaned Review'
    """

    # Drop unnecessary columns safely
    columns_to_drop = [
        'Reviewer Name',
        'Profile Link',
        'Review Count',
        'Review Date',
        'Date of Experience'
    ]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

    # Handle missing values
    data = data.dropna(subset=['Review Text', 'Rating'])
    data['Review Title'] = data['Review Title'].fillna('')
    data['Country'] = data['Country'].fillna('Unknown')

    # Extract numeric rating
    data['Rating'] = data['Rating'].astype(str).str.extract(r'(\d)').astype(float)
    data = data.dropna(subset=['Rating'])
    data['Rating'] = data['Rating'].astype(int)

    # Convert rating to sentiment
    def rating_to_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating >= 4:
            return 1  # Positive
        else:
            return None  # Neutral (drop later)

    data['Sentiment'] = data['Rating'].apply(rating_to_sentiment)
    data = data.dropna(subset=['Sentiment'])
    data['Sentiment'] = data['Sentiment'].astype(int)

    # Combine title + text
    data['Full Review'] = data['Review Title'] + " " + data['Review Text']

    # Apply text preprocessing
    data['Cleaned Review'] = data['Full Review'].apply(preprocess_text)

    # Optional: Review length (useful for analysis)
    data['Review Length'] = data['Cleaned Review'].apply(lambda x: len(x.split()))

    return data


# -----------------------------
# Utility Function
# -----------------------------

def save_processed_data(data: pd.DataFrame, output_path: str):
    """
    Save processed dataset to CSV
    """
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


# -----------------------------
# Quick Test (Optional)
# -----------------------------
if __name__ == "__main__":
    # Example usage
    input_path = "data/raw/reviews.csv"
    output_path = "data/processed/clean_reviews.csv"

    df = pd.read_csv(input_path)
    df_processed = preprocess_dataset(df)
    save_processed_data(df_processed, output_path)