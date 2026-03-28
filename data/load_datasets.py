# Load processed financial news dataset from Parquet shards
# Dataset: HuggingFace "financial-news-multisource" (preprocessed)
# Columns: date, text
# Task: Load, combine shards, add sentiment labels, and split into train/val/test

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_parquet_shards(shard_dir="./financial_news_shards_labeled/"):
    """
    Load all Parquet shards from directory and combine them.
    
    Args:
        shard_dir: Path to directory containing shard_*.parquet files
    
    Returns:
        pandas.DataFrame with columns: date, text
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        print(f"Error: Shard directory not found at {shard_dir}")
        return None
    
    parquet_files = sorted(shard_dir.glob("shard_*.parquet"))
    
    if not parquet_files:
        print(f"No Parquet shards found in {shard_dir}")
        return None
    
    print(f"Found {len(parquet_files)} Parquet shards to load...")
    
    # Load all shards
    datasets_list = []
    total_records = 0
    
    for shard_path in tqdm(parquet_files, desc="Loading shards"):
        df = pd.read_parquet(str(shard_path))
        datasets_list.append(df)
        total_records += len(df)
    
    # Combine all shards
    combined_df = pd.concat(datasets_list, ignore_index=True)
    
    print(f"\n[OK] Loaded {total_records:,} records from {len(parquet_files)} shards")
    print(f"  Columns: {list(combined_df.columns)}")
    
    return combined_df

def add_sentiment_labels(df, sentiment_col="sentiment"):
    """
    Add sentiment labels to dataset.
    
    Sentiment labels must already exist in the labeled Parquet shards.
    
    Args:
        df: Input DataFrame with text and date columns
        sentiment_col: Name of sentiment column to create
    
    Returns:
        DataFrame with added sentiment column
    """
    if sentiment_col not in df.columns:
        print("Sentiment labels missing. Run label_finbert_on_parquet_shards.py first.")
        raise KeyError(
            f"Missing required column '{sentiment_col}' in Parquet shards."
        )

    # Ensure integer class labels (0,1,2).
    df[sentiment_col] = pd.to_numeric(df[sentiment_col], errors="raise").astype(int)
    
    return df

def extract_ticker(text):
    """
    Attempt to extract stock ticker from text.
    Looks for patterns like $AAPL, $TSLA, etc.
    
    Args:
        text: Text to search for ticker
    
    Returns:
        Ticker string or None
    """
    import re
    match = re.search(r'\$([A-Z]{1,5})\b', str(text))
    return match.group(1) if match else None

def add_ticker_column(df, text_col="text", ticker_col="ticker"):
    """
    Extract stock tickers from text field.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        ticker_col: Name of ticker column to create
    
    Returns:
        DataFrame with ticker column
    """
    print(f"\nExtracting stock tickers from text...")
    df[ticker_col] = df[text_col].apply(extract_ticker)
    
    ticker_coverage = df[ticker_col].notna().sum()
    print(f"[OK] Extracted tickers for {ticker_coverage:,} records ({ticker_coverage/len(df)*100:.1f}%)")
    
    return df

def split_dataset_stratified(df, test_size=0.2, val_size=0.1, 
                            stratify_col="sentiment", random_state=42):
    """
    Split dataset into train, validation, and test sets.
    Uses stratified sampling based on sentiment labels.
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set (default 0.2 = 10%)
        val_size: Fraction of train data for validation (default 0.1 = 10% of train)
        stratify_col: Column to stratify on (must have labels)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"\nSplitting dataset with stratification on '{stratify_col}'...")
    
    # Check if stratify column has actual labels (not all same value)
    if df[stratify_col].nunique() == 1:
        print(f"[WARNING] Column '{stratify_col}' has only one unique value.")
        print(f"   Stratified sampling not possible - using random split instead.")
        stratify = None
    else:
        stratify = df[stratify_col]
    
    # Split into train+val and test (80/20 split)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    # Further split train+val into train and val (90/10 split)
    train_frac = 1 - (val_size / (1 - test_size))
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        stratify=train_val_df[stratify_col] if stratify is not None else None,
        random_state=random_state
    )
    
    print(f"[OK] Split complete:")
    print(f"  Train: {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def create_huggingface_dataset(train_df, val_df, test_df):
    """
    Convert DataFrames to HuggingFace Datasets.
    
    Args:
        train_df, val_df, test_df: Input DataFrames
    
    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    print(f"\nConverting to HuggingFace Dataset format...")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    print(f"[OK] HuggingFace DatasetDict created")
    
    return dataset_dict

def tokenize_dataset(dataset, tokenizer_name="meta-llama/Llama-2-7b-hf", 
                     text_col="text", label_col="sentiment",
                     max_length=512, batch_size=32):
    """
    Tokenize HuggingFace dataset using Llama-2-7b tokenizer.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict to tokenize
        tokenizer_name: Tokenizer model name (default: meta-llama/Llama-2-7b-hf)
        text_col: Name of text column to tokenize (default: "text")
        label_col: Name of sentiment/label column (default: "sentiment")
        max_length: Maximum sequence length (default: 512)
        batch_size: Batch size for tokenization (default: 32)
    
    Returns:
        HuggingFace Dataset or DatasetDict with tokenized fields:
        - input_ids: Token IDs
        - attention_mask: Attention mask for tokens
        - labels: Sentiment labels (integer encoded)
    """
    from transformers import AutoTokenizer
    
    print(f"\n[INFO] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Handle DatasetDict (train/val/test splits)
    if isinstance(dataset, DatasetDict):
        print(f"\n[INFO] Tokenizing DatasetDict with {len(dataset)} splits...")
        tokenized_datasets = DatasetDict()
        
        for split_name, split_data in dataset.items():
            print(f"  Tokenizing '{split_name}' split ({len(split_data):,} samples)...")
            tokenized = _tokenize_split(
                split_data, 
                tokenizer, 
                text_col, 
                label_col, 
                max_length, 
                batch_size
            )
            tokenized_datasets[split_name] = tokenized
        
        return tokenized_datasets
    
    # Handle single Dataset
    else:
        print(f"\n[INFO] Tokenizing Dataset ({len(dataset):,} samples)...")
        return _tokenize_split(dataset, tokenizer, text_col, label_col, max_length, batch_size)


def _tokenize_split(dataset, tokenizer, text_col, label_col, max_length, batch_size):
    """
    Helper function to tokenize a single dataset split.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: AutoTokenizer instance
        text_col: Text column name
        label_col: Label column name
        max_length: Maximum sequence length
        batch_size: Batch size
    
    Returns:
        Tokenized Dataset with input_ids, attention_mask, and labels
    """
    def tokenize_function(examples):
        """Tokenize text examples."""
        tokenized = tokenizer(
            examples[text_col],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        # Add labels from sentiment column if it exists
        if label_col in examples:
            tokenized["labels"] = examples[label_col]
        
        return tokenized
    
    # Apply tokenization with batching
    print(f"    Tokenizing with max_length={max_length}, truncation=True, padding='max_length'")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels", "sentiment", text_col, label_col]],
        desc=f"Tokenizing"
    )
    
    # Remove original text and sentiment columns, keep only tokenized fields + labels
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    cols_to_remove = [col for col in tokenized_dataset.column_names if col not in cols_to_keep]
    if cols_to_remove:
        tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
    
    print(f"    [OK] Tokenization complete. Columns: {tokenized_dataset.column_names}")
    
    return tokenized_dataset


def load_financial_news_dataset(shard_dir="./financial_news_shards_labeled/",
                               add_stats=True):
    """
    Load processed financial news dataset from Parquet shards.
    Complete pipeline: Load → Add labels → Split → Convert to HuggingFace format.
    
    Args:
        shard_dir: Path to financial_news_shards directory
        add_stats: Whether to print dataset statistics
    
    Returns:
        DatasetDict with keys: 'train', 'validation', 'test'
    """
    print("\n" + "="*80)
    print("LOADING FINANCIAL NEWS DATASET FROM PARQUET SHARDS")
    print("="*80)
    
    # Step 1: Load parquet shards
    df = load_parquet_shards(shard_dir)
    if df is None:
        return None
    
    # Step 2: Validate sentiment labels
    df = add_sentiment_labels(df)

    # Validation check: print distribution before splitting
    print("\nSentiment label distribution (0=negative, 1=neutral, 2=positive):")
    vc = df["sentiment"].value_counts(dropna=False).sort_index()
    for label_id in [0, 1, 2]:
        print(f"  {label_id}: {int(vc.get(label_id, 0)):,}")
    
    # Step 3: Extract ticker information
    df = add_ticker_column(df)
    
    # Step 4: Split into train/val/test
    train_df, val_df, test_df = split_dataset_stratified(df)
    
    # Step 5: Convert to HuggingFace format
    dataset_dict = create_huggingface_dataset(train_df, val_df, test_df)
    
    # Step 6: Print statistics
    if add_stats:
        print(f"\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"Train set shape: {dataset_dict['train'].num_rows} rows")
        print(f"Val set shape:   {dataset_dict['validation'].num_rows} rows")
        print(f"Test set shape:  {dataset_dict['test'].num_rows} rows")
        print(f"\nColumn names: {dataset_dict['train'].column_names}")
        print(f"\nSample from train set:")
        sample = dataset_dict['train'][0]
        for key in sample:
            value = str(sample[key])[:100]
            print(f"  {key}: {value}...")
    
    return dataset_dict

if __name__ == "__main__":
    # Load dataset
    dataset_dict = load_financial_news_dataset()
    
    if dataset_dict is not None:
        print(f"\n[OK] Successfully loaded dataset!")
        print(f"  Access splits via: dataset_dict['train'], dataset_dict['validation'], dataset_dict['test']")
