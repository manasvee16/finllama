# Copilot Prompt:
# Load "bert-base-uncased" with a 3-class classification head.
# Fine-tune on the same labeled train split.
# Evaluate on validation and test splits.
# Save predictions and metrics.

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import warnings
warnings.filterwarnings('ignore')

def compute_metrics(eval_preds):
    """Compute metrics for BERT baseline."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def tokenize_function(examples, tokenizer):
    """Tokenize input text."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

def train_bert_baseline(train_path="./data/cleaned_train.csv",
                       val_path="./data/cleaned_test.csv",
                       output_dir="./model_output/bert_baseline"):
    """
    Fine-tune BERT-base on financial news sentiment.
    """
    print("\n" + "="*60)
    print("BERT BASELINE SENTIMENT CLASSIFICATION")
    print("="*60 + "\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Rename columns if needed
    if 'Sentence' in train_df.columns:
        train_df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
        val_df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    
    # Rename label column
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Load BERT model
    print("Loading BERT-base-uncased model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch"
    )
    
    # Create Trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.train()
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"\nModel saved to {output_dir}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
    print("="*60)
    
    # Save results
    results_df = pd.DataFrame([val_results])
    results_df.to_csv(os.path.join(output_dir, "validation_results.csv"), index=False)
    
    return model, trainer, val_results

if __name__ == "__main__":
    train_bert_baseline()
