# Copilot Prompt:
# Load the cleaned HuggingFace dataset of financial news text.
# Load pretrained FinBERT model "yiyanghkust/finbert-tone".
# For each text sample, predict sentiment probabilities.
# Map predictions to labels: {negative:0, neutral:1, positive:2}.
# Save dataset with fields: text, sentiment_label, sentiment_score.
# Split into train/val/test (80/10/10) with stratified sampling.
# Save splits to disk.

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_financial_news_dataset(data_path="./data/cleaned_test.csv"):
    """
    Load cleaned financial news dataset.
    Assumes CSV with 'Sentence' and 'Sentiment' columns.
    """
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    return df

def auto_label_with_finbert(texts, tokenizer, model, batch_size=8):
    """
    Predict sentiment for each text using FinBERT.
    Returns labels and confidence scores.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    label_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    predictions = []
    confidences = []
    
    print("Auto-labeling texts with FinBERT...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_confs = torch.max(probs, dim=-1).values.cpu().numpy()
        
        predictions.extend(batch_preds)
        confidences.extend(batch_confs)
    
    return np.array(predictions), np.array(confidences)

def prepare_auto_labeled_dataset(data_path="./data/cleaned_test.csv",
                                 output_dir="./auto_labeled_data",
                                 test_size=0.2,
                                 val_size=0.1):
    """
    Load financial news, auto-label with FinBERT, and save splits.
    Splits: train (80%), val (10%), test (10%)
    """
    print("\n" + "="*80)
    print("AUTO-LABELING FINANCIAL NEWS WITH FINBERT")
    print("="*80 + "\n")
    
    # Load data
    df = load_financial_news_dataset(data_path)
    if df is None:
        return
    
    # Rename columns if needed
    if 'Sentence' in df.columns:
        df.rename(columns={'Sentence': 'text'}, inplace=True)
    
    texts = df['text'].values
    
    # Load FinBERT model and tokenizer
    print("Loading FinBERT (yiyanghkust/finbert-tone)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            num_labels=3  # 0: negative, 1: neutral, 2: positive
        )
    except Exception as e:
        print(f"Note: Could not load yiyanghkust/finbert-tone. Using ProsusAI/finbert instead.")
        print(f"Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            num_labels=3,
            trust_remote_code=True
        )
    
    # Auto-label texts
    pred_labels, confidences = auto_label_with_finbert(texts, tokenizer, model, batch_size=8)
    
    # Create labeled dataset
    labeled_df = pd.DataFrame({
        'text': texts,
        'sentiment_label': pred_labels,
        'sentiment_score': confidences
    })
    
    print(f"\nAuto-labeled {len(labeled_df)} samples")
    print(f"Label distribution:")
    print(f"  Negative (0): {(pred_labels == 0).sum()}")
    print(f"  Neutral  (1): {(pred_labels == 1).sum()}")
    print(f"  Positive (2): {(pred_labels == 2).sum()}")
    
    # Split into train/val/test with stratified sampling
    print(f"\nSplitting into train (80%) / val (10%) / test (10%) with stratified sampling...")
    
    # First split: train (80%) vs val+test (20%)
    train_df, temp_df = train_test_split(
        labeled_df,
        test_size=(val_size + test_size),
        random_state=42,
        stratify=pred_labels
    )
    
    # Second split: val (10%) and test (10%) from remaining 20%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),  # 0.5 of remaining
        random_state=42,
        stratify=temp_df['sentiment_label'].values
    )
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "auto_labeled_train.csv")
    val_path = os.path.join(output_dir, "auto_labeled_val.csv")
    test_path = os.path.join(output_dir, "auto_labeled_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n" + "="*80)
    print("AUTO-LABELING COMPLETE")
    print("="*80)
    print(f"Train set saved to: {train_path}")
    print(f"Val set saved to: {val_path}")
    print(f"Test set saved to: {test_path}")
    print(f"Fields: text, sentiment_label, sentiment_score")
    print(f"Label mapping: {{negative: 0, neutral: 1, positive: 2}}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_auto_labeled_dataset()
