import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import FinLlama model and dataset loading/tokenization functions
from finllama_lora_model import create_finllama_lora_model
sys.path.insert(0, '../data')
from load_datasets import load_financial_news_dataset, tokenize_dataset

def compute_metrics(eval_preds):
    """Compute evaluation metrics for FinLlama."""
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

def train_finllama(output_dir="./model_output/finllama"):
    """
    Fine-tune FinLlama model using HuggingFace Trainer with AdamW optimizer.
    
    Pipeline:
    1. Load datasets from data/load_datasets.py
    2. Tokenize dataset using Llama tokenizer via tokenize_dataset()
    3. Load FinLLaMA model with LoRA adapters
    4. Use HuggingFace Trainer API for training
    
    Training will only update LoRA parameters and classification head.
    """
    print("\n" + "="*80)
    print("FINLLAMA SENTIMENT FINE-TUNING WITH LORA")
    print("="*80 + "\n")
    
    # Step 1: Load datasets from parquet shards
    print("[Step 1] Loading financial news dataset from parquet shards...")
    dataset_dict = load_financial_news_dataset(shard_dir="./financial_news_shards_labeled/")
    
    if dataset_dict is None:
        print("[ERROR] Failed to load dataset")
        return None
    
    print(f"  Train: {dataset_dict['train'].num_rows:,} samples")
    print(f"  Val:   {dataset_dict['validation'].num_rows:,} samples")
    print(f"  Test:  {dataset_dict['test'].num_rows:,} samples")
    
    # Step 2: Tokenize dataset using Llama-2-7b tokenizer
    print("\n[Step 2] Tokenizing dataset with Llama-2-7b tokenizer...")
    tokenized_datasets = tokenize_dataset(
        dataset_dict,
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        text_col="text",
        label_col="sentiment",
        max_length=512,
        batch_size=32
    )
    
    if tokenized_datasets is None:
        print("[ERROR] Failed to tokenize dataset")
        return None
    
    print(f"  Tokenized columns: {tokenized_datasets['train'].column_names}")
    if "labels" not in tokenized_datasets["train"].column_names:
        raise KeyError("Tokenized dataset is missing required 'labels' column. Check sentiment->labels mapping.")

    # Quick sanity-check that labels are integer-coded (0/1/2).
    train_ds = tokenized_datasets["train"]
    n_sample = min(2000, len(train_ds))
    # Use select() so we don't accidentally materialize the entire label column.
    train_labels_sample = train_ds.select(range(n_sample))["labels"]
    unique_labels = sorted(set(int(x) for x in train_labels_sample))
    print(f"  Sample label values (train, up to 2000 samples): {unique_labels}")
    
    # Step 3: Load FinLlama model with LoRA adapters
    print("\n[Step 3] Loading FinLLaMA model with LoRA adapters...")
    model = create_finllama_lora_model(
        base_model="meta-llama/Llama-2-7b-hf",
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        num_labels=3
    )
    
    # Verify parameter configuration
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Step 4: Use HuggingFace Trainer API with cross entropy loss
    print("\n[Step 4] Configuring HuggingFace Trainer...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # As per requirements
        per_device_train_batch_size=8,  # As per requirements
        per_device_eval_batch_size=8,  # As per requirements
        learning_rate=2e-5,  # As per requirements
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",  # As per requirements
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch",  # AdamW optimizer (cross entropy loss default)
        fp16=False  # Disable FP16 for stability
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    )
    
    # Train model (only LoRA adapters and classification head will be updated)
    print("\n[TRAINING] Starting training...\n")
    trainer.train()
    
    # Save trained adapters
    print("\n[SAVING] Saving trained adapters...")
    os.makedirs(output_dir, exist_ok=True)
    adapter_save_path = os.path.join(output_dir, "adapter_model.bin")
    config_save_path = os.path.join(output_dir, "adapter_config.json")
    model.save_pretrained(output_dir)
    print(f"  Adapter weights saved to {output_dir}")
    
    # Evaluate on validation set
    print("\n[EVALUATION] Evaluating on validation set...")
    val_results = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
    print("="*80)

    # Save validation predictions
    print("\n[INFERENCE] Generating predictions on validation set...")
    val_predictions = trainer.predict(tokenized_datasets['validation'])
    val_pred_labels = np.argmax(val_predictions.predictions, axis=-1)
    val_true_labels = tokenized_datasets['validation']['labels']

    val_pred_df = pd.DataFrame({
        "true_label": val_true_labels,
        "predicted_label": val_pred_labels,
    })
    val_pred_path = os.path.join(output_dir, "val_predictions.csv")
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"  Validation predictions saved to {val_pred_path}")
    
    # Evaluate on test set
    print("\n[EVALUATION] Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
    print("="*80)
    
    # Generate predictions on test set
    print("\n[INFERENCE] Generating predictions on test set...")
    predictions = trainer.predict(tokenized_datasets['test'])
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = tokenized_datasets['test']['labels']
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': pred_labels
    })
    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")

    # Also save a clearer test-only filename.
    test_pred_path = os.path.join(output_dir, "test_predictions.csv")
    pred_df.to_csv(test_pred_path, index=False)
    print(f"  Test predictions saved to {test_pred_path}")
    
    # Save validation metrics
    val_metrics_df = pd.DataFrame([val_results])
    val_metrics_path = os.path.join(output_dir, "val_metrics.csv")
    val_metrics_df.to_csv(val_metrics_path, index=False)
    print(f"  Validation metrics saved to {val_metrics_path}")
    
    # Save test metrics
    test_metrics_df = pd.DataFrame([test_results])
    test_metrics_path = os.path.join(output_dir, "test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_path, index=False)
    print(f"  Test metrics saved to {test_metrics_path}")
    
    # Print classification report
    label_names = ['Negative', 'Neutral', 'Positive']
    print("\n" + "="*80)
    print("TEST SET CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(true_labels, pred_labels, target_names=label_names))
    
    return model, trainer, val_results, test_results

if __name__ == "__main__":
    train_finllama()
