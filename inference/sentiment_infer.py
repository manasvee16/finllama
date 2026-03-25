# Copilot Prompt:
# Write inference script.
# Load base Llama-2-7B and LoRA adapter weights.
# Accept a news article text input.
# Output:
# - sentiment label (positive/neutral/negative)
# - sentiment strength (SoftMax probability).

import torch
import pandas as pd
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F

# Import data loading functions
sys.path.insert(0, '../data')
from load_datasets import load_financial_news_dataset

class SentimentInferencer:
    """
    Inference class for FinLlama sentiment classification.
    """
    
    def __init__(self, base_model="meta-llama/Llama-2-7b-hf", adapter_path=None):
        """
        Initialize inferencer with base Llama-2 model and LoRA adapter.
        """
        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base Llama-2-7B model for sequence classification
        print(f"Loading base model {base_model}...")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Sentiment label mapping
        self.id_to_label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        print(f"Inferencer loaded on device: {self.device}")
    
    def infer(self, text, return_probabilities=False):
        """
        Run inference on a news article text.
        
        Args:
            text: News article text
            return_probabilities: If True, return softmax probabilities
        
        Returns:
            Dictionary with sentiment label and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy()
        confidence = probabilities[predicted_class]
        
        # Get label
        sentiment_label = self.id_to_label[predicted_class]
        
        result = {
            'sentiment_label': sentiment_label,
            'confidence': float(confidence),
            'predicted_class': predicted_class
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.id_to_label[i]: float(p) for i, p in enumerate(probabilities)
            }
        
        return result

def infer_batch(texts, base_model="meta-llama/Llama-2-7b-hf", adapter_path=None):
    """
    Run batch inference on multiple texts.
    
    Args:
        texts: List of news article texts
        base_model: Base model name
        adapter_path: Path to LoRA adapter
    
    Returns:
        List of result dictionaries with sentiment_label and confidence
    """
    inferencer = SentimentInferencer(base_model, adapter_path)
    results = []
    
    for i, text in enumerate(texts):
        result = inferencer.infer(text)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(texts)} texts")
    
    return results

def run_batch_sentiment_inference(
    output_csv="./results/sentiment_predictions.csv",
    base_model="meta-llama/Llama-2-7b-hf",
    adapter_path="./model_output/finllama",
    dataset_split="test"
):
    """
    Run batch sentiment inference on processed dataset shards.
    
    Steps:
    1. Load processed dataset (train/val/test splits)
    2. Run FinLLaMA inference on each article
    3. Store results in dataframe with fields: text, sentiment_label, confidence, ticker, date
    4. Save predictions to CSV
    
    Args:
        output_csv: Path to save predictions CSV
        base_model: Base model name
        adapter_path: Path to trained LoRA adapter
        dataset_split: Which split to run inference on ('test', 'validation', 'train')
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*80)
    print("BATCH SENTIMENT INFERENCE")
    print("="*80 + "\n")
    
    # Step 1: Load dataset
    print(f"[Step 1] Loading dataset split: {dataset_split}...")
    dataset_dict = load_financial_news_dataset(shard_dir="./financial_news_shards")
    
    if dataset_dict is None:
        print("[ERROR] Failed to load dataset")
        return None
    
    dataset = dataset_dict[dataset_split]
    print(f"  Loaded {dataset.num_rows:,} articles")
    
    # Step 2: Initialize inferencer
    print(f"\n[Step 2] Initializing FinLLaMA inferencer...")
    if adapter_path and os.path.exists(adapter_path):
        inferencer = SentimentInferencer(base_model, adapter_path)
    else:
        print(f"  Warning: Adapter path {adapter_path} not found, using base model only")
        inferencer = SentimentInferencer(base_model)
    
    # Step 3: Run inference on each article
    print(f"\n[Step 3] Running inference on {dataset.num_rows:,} articles...")
    
    predictions = []
    for i, article in enumerate(dataset):
        # Extract text and other fields
        text = article.get('text', '')
        ticker = article.get('ticker', 'N/A')
        date = article.get('date', 'N/A')
        
        # Run inference
        result = inferencer.infer(text)
        
        # Store result with metadata
        prediction = {
            'text': text,
            'sentiment_label': result['sentiment_label'],
            'confidence': result['confidence'],
            'ticker': ticker,
            'date': date
        }
        predictions.append(prediction)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1:,}/{dataset.num_rows:,} articles")
    
    # Step 4: Create dataframe
    print(f"\n[Step 4] Creating results dataframe...")
    predictions_df = pd.DataFrame(predictions)
    
    # Ensure correct column order
    predictions_df = predictions_df[['text', 'sentiment_label', 'confidence', 'ticker', 'date']]
    
    print(f"  DataFrame shape: {predictions_df.shape}")
    print(f"  Sentiment distribution:")
    print(predictions_df['sentiment_label'].value_counts())
    
    # Step 5: Save to CSV
    print(f"\n[Step 5] Saving predictions to {output_csv}...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    predictions_df.to_csv(output_csv, index=False)
    print(f"  Saved {predictions_df.shape[0]:,} predictions to {output_csv}")
    
    # Print statistics
    print(f"\n" + "="*80)
    print("INFERENCE STATISTICS")
    print("="*80)
    print(f"Total articles: {predictions_df.shape[0]:,}")
    print(f"Average confidence: {predictions_df['confidence'].mean():.4f}")
    print(f"Confidence std dev: {predictions_df['confidence'].std():.4f}")
    print(f"\nSentiment distribution:")
    for label in ['negative', 'neutral', 'positive']:
        count = (predictions_df['sentiment_label'] == label).sum()
        pct = 100 * count / len(predictions_df)
        print(f"  {label:12s}: {count:6,} ({pct:5.2f}%)")
    print(f"\nUnique tickers: {predictions_df['ticker'].nunique():,}")
    print("="*80 + "\n")
    
    return predictions_df

if __name__ == "__main__":
    # Run batch sentiment inference on test set
    predictions_df = run_batch_sentiment_inference(
        output_csv="./results/sentiment_predictions.csv",
        base_model="meta-llama/Llama-2-7b-hf",
        adapter_path="./model_output/finllama",
        dataset_split="test"
    )
    
    if predictions_df is not None:
        print("Sample predictions:")
        print(predictions_df.head())
