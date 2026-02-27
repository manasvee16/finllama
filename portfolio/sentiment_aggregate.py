# Copilot Prompt:
# Given multiple news articles per company per day,
# compute average daily sentiment score St:
# St = (1/Nt) * sum(sentiment_strength_i)
# Save daily sentiment per company as CSV.

import pandas as pd
import numpy as np
from datetime import datetime

def aggregate_daily_sentiment(inference_results_df):
    """
    Aggregate sentiment across articles per company per day.
    
    Args:
        inference_results_df: DataFrame with columns:
            - company: Company ticker
            - date: Publication date
            - sentiment_strength: Sentiment score (0-1)
    
    Returns:
        DataFrame with columns: company, date, daily_sentiment
    """
    
    # Group by company and date
    daily_sentiment = inference_results_df.groupby(['company', 'date'])['sentiment_strength'].agg([
        ('daily_sentiment', 'mean'),
        ('article_count', 'count')
    ]).reset_index()
    
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

def compute_daily_sentiment_from_file(inference_csv_path, output_csv_path="./daily_sentiment.csv"):
    """
    Load inference results CSV and compute daily sentiment.
    """
    # Load inference results
    df = pd.read_csv(inference_csv_path)
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(df)
    
    # Save to CSV
    save_daily_sentiment(daily_sentiment, output_csv_path)
    
    return daily_sentiment

if __name__ == "__main__":
    # Example: Process sample inference results
    # Assuming you have an inference results CSV with columns:
    # company, date, sentiment_strength
    
    sample_df = pd.DataFrame({
        'company': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
        'sentiment_strength': [0.8, 0.75, 0.6, 0.85, 0.7]
    })
    
    daily_sentiment = aggregate_daily_sentiment(sample_df)
    save_daily_sentiment(daily_sentiment)
