# Copilot Prompt:
# Load kaggle-data.csv which has columns:
# - Sentence (text)
# - Sentiment (positive, negative, neutral)
# Map labels to integers {negative:0, neutral:1, positive:2}.
# Shuffle and split into train (80%) and test (20%).
# Save as train.csv and test.csv.

# ↓ Copilot will generate code here

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(csv_path, output_dir="./data"):
    """
    Load kaggle-data.csv, map sentiment labels, shuffle, and split.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Map sentiment labels to integers
    sentiment_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set: {len(train_df)} samples -> {train_path}")
    print(f"Test set: {len(test_df)} samples -> {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "../../kaggle-data.csv")
    load_and_split_data(csv_path)
