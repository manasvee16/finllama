# Copilot Prompt:
# Write preprocessing code for financial news text.
# Steps:
# - Lowercase text
# - Remove HTML tags
# - Remove URLs
# - Keep financial symbols like $ and %
# - Truncate text to max 512 tokens using HuggingFace tokenizer.
# Save cleaned dataset as cleaned_train.csv and cleaned_test.csv.

import pandas as pd
import re
import os
from transformers import AutoTokenizer

def preprocess_text(text):
    """
    Preprocess financial news text.
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Keep financial symbols like $ and %
    # Just clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def truncate_to_tokens(text, tokenizer, max_tokens=512):
    """
    Truncate text to max_tokens using HuggingFace tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=False)

def preprocess_dataset(input_path, output_path, tokenizer_name="distilbert-base-uncased"):
    """
    Load CSV, preprocess text, truncate to 512 tokens, and save.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Preprocess text column
    df['Sentence'] = df['Sentence'].apply(preprocess_text)
    df['Sentence'] = df['Sentence'].apply(lambda x: truncate_to_tokens(x, tokenizer, max_tokens=512))
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Preprocess train and test sets
    preprocess_dataset("./train.csv", "./cleaned_train.csv")
    preprocess_dataset("./test.csv", "./cleaned_test.csv")
