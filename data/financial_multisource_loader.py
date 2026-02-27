# Copilot Prompt:
# Load the HuggingFace dataset "Brianferrell787/financial-news-multisource" in streaming mode.
# Extract fields: date and text.
# Filter out rows with empty or null text.
# Normalize text to lowercase.
# Deduplicate identical texts.
# Save output into Parquet shards of 50k samples each.
# Show a progress bar while processing.

import os
import hashlib
from datasets import load_dataset
from pathlib import Path

def load_and_process_financial_news():
    """
    Load the "Brianferrell787/financial-news-multisource" dataset from HuggingFace in streaming mode.
    Filter out entries missing text or with empty text.
    Extract fields: date and text.
    Save the cleaned output into a set of Parquet shards for efficient future use.
    Include deduplication and lowercase normalisation.
    """
    
    print("=" * 80)
    print("FINANCIAL NEWS MULTISOURCE DATASET LOADER")
    print("=" * 80)
    print()
    print("Loading financial news dataset from HuggingFace...")
    print()
    
    # Load dataset in streaming mode to handle large datasets
    try:
        ds = load_dataset("Brianferrell787/financial-news-multisource", streaming=True)
    except Exception as e:
        print(f"WARNING: AUTHENTICATION REQUIRED")
        print(f"Error: {e}")
        print()
        print("This dataset is gated on HuggingFace and requires authentication.")
        print()
        print("SETUP INSTRUCTIONS:")
        print("-" * 80)
        print("1. Accept the dataset license:")
        print("   Visit: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource")
        print("   Click 'Agree and access dataset'")
        print()
        print("2. Authenticate with HuggingFace CLI:")
        print("   Run: huggingface-cli login")
        print("   Paste your HuggingFace API token when prompted")
        print("   (Get token from: https://huggingface.co/settings/tokens)")
        print()
        print("3. Run this script again:")
        print("   python data/financial_multisource_loader.py")
        print("-" * 80)
        print()
        print("EXPECTED OUTPUT ONCE AUTHENTICATED:")
        print("  - Streams dataset in memory-efficient mode")
        print("  - Filters empty/missing text entries")
        print("  - Extracts date and text fields")
        print("  - Deduplicates records (MD5 hash)")
        print("  - Normalizes text to lowercase")
        print("  - Saves to Parquet shards (10k records per file)")
        print("  - Outputs: financial_news_shards/shard_XXXX.parquet")
        print()
        return
    
    # Process the dataset
    seen_hashes = set()
    processed_records = []
    shard_size = 50000
    shard_count = 0
    total_processed = 0
    
    print("Processing dataset: filtering, deduplicating, and normalizing...")
    
    # Iterate through the dataset
    for split_name, split_data in ds.items():
        print(f"Processing split: {split_name}")
        
        for record in split_data:
            # Extract text field - handle different possible field names
            text = record.get('text') or record.get('content') or record.get('body')
            
            # Filter: Skip entries with missing or empty text
            if not text or (isinstance(text, str) and text.strip() == ""):
                continue
            
            # Extract date field - handle different possible field names
            date = record.get('date') or record.get('timestamp') or record.get('published_at') or "unknown"
            
            # Normalize text to lowercase
            text_normalized = text.lower().strip()
            
            # Deduplication: Create hash of text to identify duplicates
            text_hash = hashlib.md5(text_normalized.encode()).hexdigest()
            
            if text_hash in seen_hashes:
                continue  # Skip duplicate
            
            seen_hashes.add(text_hash)
            
            # Store processed record
            processed_records.append({
                'date': str(date),
                'text': text_normalized
            })
            
            total_processed += 1
            
            # Progress bar: Show progress every 10k records
            if total_processed % 10000 == 0:
                print(f"  Progress: {total_processed:,} records processed...")
            
            # Save to Parquet shards when reaching shard size (50k)
            if len(processed_records) >= shard_size:
                save_shard(processed_records, shard_count)
                shard_count += 1
                processed_records = []
    
    # Save remaining records to final shard
    if processed_records:
        save_shard(processed_records, shard_count)
        shard_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total shards created: {shard_count}")
    print(f"Unique records after deduplication: {len(seen_hashes)}")
    print(f"Shard size: {shard_size:,} samples per shard")

def save_shard(records, shard_number):
    """Save a batch of records to Parquet format."""
    from datasets import Dataset
    
    shard_dir = Path("financial_news_shards")
    shard_dir.mkdir(exist_ok=True)
    
    dataset = Dataset.from_dict({
        'date': [r['date'] for r in records],
        'text': [r['text'] for r in records]
    })
    
    shard_path = shard_dir / f"shard_{shard_number:04d}.parquet"
    dataset.to_parquet(str(shard_path))
    print(f"Saved shard {shard_number}: {shard_path} ({len(records)} records)")

if __name__ == "__main__":
    load_and_process_financial_news()