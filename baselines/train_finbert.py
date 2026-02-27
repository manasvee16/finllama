# Copilot Prompt:
# Load pretrained FinBERT "yiyanghkust/finbert-tone" with classification head.
# Fine-tune on the labeled train split.
# Evaluate on validation and test splits.
# Save predictions and metrics (accuracy, precision, recall, F1).

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def compute_metrics(eval_preds):
    """Compute metrics for FinBERT evaluation."""
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

def train_finbert(train_path="./data/cleaned_train.csv",
                  val_path="./data/cleaned_test.csv",
                  output_dir="./model_output/finbert"):
    """
    Fine-tune FinBERT model on financial news sentiment classification.
    """
    print("\n" + "="*80)
    print("FINBERT SENTIMENT CLASSIFICATION")
    print("="*80 + "\n")
    
    # Load tokenizer
    print("Loading FinBERT tokenizer (yiyanghkust/finbert-tone)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    except Exception as e:
        print(f"Note: Could not load yiyanghkust/finbert-tone. Using ProsusAI/finbert instead.")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    # Load data
    print("Loading datasets...")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Error: Dataset files not found at {train_path} or {val_path}")
        return None
    
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
    
    # Load FinBERT model
    print("Loading FinBERT model (yiyanghkust/finbert-tone)...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            num_labels=3
        )
    except Exception as e:
        print(f"Note: Could not load yiyanghkust/finbert-tone. Using ProsusAI/finbert instead.")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            num_labels=3,
            trust_remote_code=True
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
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
    print("="*80)
    
    # Generate predictions on validation set
    print("\nGenerating predictions on validation set...")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = val_df['label'].values
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': pred_labels
    })
    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    # Save detailed metrics
    results_df = pd.DataFrame([val_results])
    results_path = os.path.join(output_dir, "metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Metrics saved to {results_path}")
    
    # Print detailed classification report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(true_labels, pred_labels, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    return model, trainer, val_results

if __name__ == "__main__":
    train_finbert()
