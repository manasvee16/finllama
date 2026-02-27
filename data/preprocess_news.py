# Copilot Prompt:
# Load Parquet shards created from financial_multisource_loader.
# Remove HTML tags, URLs, and non-ASCII characters.
# Keep financial symbols like $ and %.
# Tokenize text using Llama-2 tokenizer with max_length=512 and truncation.
# Save tokenized dataset in HuggingFace Dataset format to disk.

import os
import re
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

def load_parquet_shards(shard_dir="./financial_news_shards"):
    """
    Load all Parquet shards from directory.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        print(f"Error: Shard directory not found at {shard_dir}")
        return None
    
    parquet_files = sorted(shard_dir.glob("shard_*.parquet"))
    
    if not parquet_files:
        print(f"No Parquet shards found in {shard_dir}")
        return None
    
    print(f"Found {len(parquet_files)} Parquet shards to process...")
    
    # Load all shards
    datasets = []
    for shard_path in tqdm(parquet_files, desc="Loading shards"):
        ds = Dataset.from_parquet(str(shard_path))
        datasets.append(ds)
    
    # Concatenate all shards
    combined_dataset = DatasetDict({"train": datasets[0]})
    for ds in datasets[1:]:
        combined_dataset["train"] = combined_dataset["train"].concatenate(ds)
    
    return combined_dataset

def preprocess_text(text):
    """
    Remove HTML tags, URLs, and non-ASCII characters.
    Keep financial symbols like $ and %.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove non-ASCII characters except financial symbols
    # Keep: $ % , . - : ; ! ? ' " ( )
    text = re.sub(r'[^\x00-\x7F$%,.\-:;!?\'\"()\s]', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize text using Llama-2 tokenizer with truncation.
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

def preprocess_and_tokenize_shards(shard_dir="./financial_news_shards",
                                    tokenizer_name="meta-llama/Llama-2-7b-hf",
                                    output_dir="./tokenized_financial_news"):
    """
    Load Parquet shards, preprocess text, and tokenize using Llama-2 tokenizer.
    Save tokenized dataset to disk.
    """
    print("\n" + "="*80)
    print("FINANCIAL NEWS PREPROCESSING AND TOKENIZATION")
    print("="*80 + "\n")
    
    # Load shards
    dataset = load_parquet_shards(shard_dir)
    if dataset is None:
        return
    
    print(f"Total samples loaded: {len(dataset['train']):,}")
    
    # Load tokenizer
    print(f"\nLoading {tokenizer_name} tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Failed to load {tokenizer_name}. Using distilbert-base-uncased instead.")
        print(f"Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Preprocess text
    print("\nPreprocessing text (removing HTML, URLs, non-ASCII)...")
    
    def preprocess_batch(batch):
        batch['text'] = [preprocess_text(text) for text in batch['text']]
        return batch
    
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        desc="Preprocessing"
    )
    
    # Tokenize
    print("\nTokenizing text (max_length=512)...")
    
    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer, max_length=512)
    
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        desc="Tokenizing",
        remove_columns=['text', 'date'] if 'date' in dataset['train'].column_names else ['text']
    )
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving tokenized dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Tokenized dataset saved to: {output_dir}")
    print(f"Total samples: {len(dataset['train']):,}")
    print(f"Features: {dataset['train'].column_names}")
    
    return dataset

if __name__ == "__main__":
    preprocess_and_tokenize_shards()
