# Copilot Prompt:
# Given multiple news articles per company per day,
# compute average daily sentiment score St:
# St = (1/Nt) * sum(sentiment_strength_i)
# Save daily sentiment per company as CSV.

import pandas as pd
import numpy as np
import os
from datetime import datetime

def aggregate_daily_sentiment(inference_results_df):
    """
    Aggregate sentiment across articles per ticker per day.
    
    Computes daily sentiment score: St = (1/N) * sum(confidence_i)
    
    Args:
        inference_results_df: DataFrame with columns:
            - ticker: Stock ticker symbol
            - date: Publication date
            - confidence: Sentiment confidence score (0-1)
    
    Returns:
        DataFrame with columns:
            - ticker: Stock ticker
            - date: Publication date
            - daily_sentiment: Average sentiment score for the day
            - article_count: Number of articles for that ticker/date
    """
    
    # Group by ticker and date, compute mean confidence (daily sentiment) and count articles
    daily_sentiment = inference_results_df.groupby(['ticker', 'date']).agg({
        'confidence': ['mean', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['ticker', 'date', 'daily_sentiment', 'article_count']
    
    # Sort by ticker and date
    daily_sentiment = daily_sentiment.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    return daily_sentiment

def process_inference_batch(batch_results, company, date):
    """
    Convert inference results from sentiment_infer.py to DataFrame format.
    
    Args:
        batch_results: List of inference results from infer_batch()
        company: Company ticker
        date: Publication date
    
    Returns:
        DataFrame ready for aggregation
    """
    data = []
    for result in batch_results:
        data.append({
            'company': company,
            'date': date,
            'sentiment_strength': result['sentiment_strength']
        })
    
    return pd.DataFrame(data)

def save_daily_sentiment(daily_sentiment_df, output_path="./daily_sentiment.csv"):
    """
    Save daily sentiment scores to CSV.
    """
    daily_sentiment_df.to_csv(output_path, index=False)
    print(f"Daily sentiment saved to {output_path}")
    print(f"Shape: {daily_sentiment_df.shape}")
    print(daily_sentiment_df.head())
    
    return output_path

def compute_daily_sentiment_from_file(
    inference_csv_path="./results/sentiment_predictions.csv",
    output_csv_path="./results/daily_sentiment.csv"
):
    """
    Load sentiment_predictions.csv and compute aggregated daily sentiment.
    
    Steps:
    1. Load sentiment_predictions.csv (from sentiment_infer.py)
    2. Group by ticker and date
    3. Compute daily sentiment score: St = (1/N) * sum(confidence_i)
    4. Return dataframe with: ticker, date, daily_sentiment, article_count
    5. Save to CSV
    
    Args:
        inference_csv_path: Path to sentiment_predictions.csv
        output_csv_path: Path to save daily_sentiment.csv
    
    Returns:
        DataFrame with aggregated daily sentiment scores
    """
    print("\n" + "="*80)
    print("SENTIMENT AGGREGATION")
    print("="*80 + "\n")
    
    # Step 1: Load sentiment predictions
    print(f"[Step 1] Loading sentiment predictions from {inference_csv_path}...")
    if not os.path.exists(inference_csv_path):
        print(f"[ERROR] File not found: {inference_csv_path}")
        return None
    
    df = pd.read_csv(inference_csv_path)
    print(f"  Loaded {df.shape[0]:,} predictions")
    print(f"  Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['ticker', 'date', 'confidence']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] Missing required columns. Expected: {required_cols}")
        print(f"        Found: {list(df.columns)}")
        return None
    
    # Step 2: Aggregate daily sentiment
    print(f"\n[Step 2] Aggregating sentiment by ticker and date...")
    daily_sentiment = aggregate_daily_sentiment(df)
    print(f"  Aggregated to {daily_sentiment.shape[0]:,} ticker-date combinations")
    print(f"  Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
    print(f"  Unique tickers: {daily_sentiment['ticker'].nunique()}")
    
    # Step 3: Print statistics
    print(f"\n[Statistics]")
    print(f"  Average daily sentiment: {daily_sentiment['daily_sentiment'].mean():.4f}")
    print(f"  Std dev: {daily_sentiment['daily_sentiment'].std():.4f}")
    print(f"  Min: {daily_sentiment['daily_sentiment'].min():.4f}")
    print(f"  Max: {daily_sentiment['daily_sentiment'].max():.4f}")
    print(f"  Average articles per day per ticker: {daily_sentiment['article_count'].mean():.2f}")
    print(f"  Min articles: {daily_sentiment['article_count'].min()}")
    print(f"  Max articles: {daily_sentiment['article_count'].max()}")
    
    # Step 4: Save to CSV
    print(f"\n[Step 3] Saving aggregated sentiment to {output_csv_path}...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    daily_sentiment.to_csv(output_csv_path, index=False)
    print(f"  Saved {daily_sentiment.shape[0]:,} rows to {output_csv_path}")
    
    print("\n" + "="*80)
    print("Sample aggregated sentiment:")
    print("="*80)
    print(daily_sentiment.head(10))
    print("="*80 + "\n")
    
    return daily_sentiment

if __name__ == "__main__":
    # Run sentiment aggregation on sentiment_predictions.csv
    daily_sentiment = compute_daily_sentiment_from_file(
        inference_csv_path="./results/sentiment_predictions.csv",
        output_csv_path="./results/daily_sentiment.csv"
    )
