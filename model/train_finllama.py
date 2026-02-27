# Copilot Prompt:
# Load tokenized labeled train/val/test splits.
# Load FinLlama model from finllama_lora_model.py.
# Fine-tune using HuggingFace Trainer with AdamW optimizer.
# Train only LoRA adapters and classification head.
# Save trained LoRA adapter weights.
# Evaluate on validation and test splits.
# Save predictions and metrics.

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import FinLlama model from finllama_lora_model.py
from finllama_lora_model import create_finllama_lora_model

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

def tokenize_function(examples, tokenizer):
    """Tokenize input text."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

def train_finllama(train_path="./data/cleaned_train.csv",
                   val_path="./data/cleaned_test.csv",
                   output_dir="./model_output/finllama_lora"):
    """
    Fine-tune FinLlama model using HuggingFace Trainer with AdamW optimizer.
    Trains only LoRA adapters and classification head.
    """
    print("\n" + "="*80)
    print("FINLLAMA SENTIMENT FINE-TUNING WITH LORA")
    print("="*80 + "\n")
    
    # Load tokenizer
    print("Loading Llama-2-7B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training and validation data
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
    
    # Rename label column and set format
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Load FinLlama model with LoRA from finllama_lora_model.py
    print("Loading FinLlama model with LoRA adapters...")
    model = create_finllama_lora_model(
        base_model="meta-llama/Llama-2-7b-hf",
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        num_labels=3
    )
    
    # Training arguments with AdamW optimizer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch",  # AdamW optimizer
        fp16=False  # Disable FP16 for stability
    )
    
    # Create Trainer
    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train model (only LoRA adapters and classification head will be updated)
    print("\nStarting training...\n")
    trainer.train()
    
    # Save trained LoRA adapter weights
    os.makedirs(output_dir, exist_ok=True)
    adapter_save_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"\nLoRA adapter weights saved to {adapter_save_path}")
    
    # Evaluate on validation and test splits
    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
    print("="*80)
    
    # Generate predictions on validation set
    print("\nGenerating predictions on validation set...")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = val_df['label'].values
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': pred_labels
    })
    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([val_results])
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Print classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(true_labels, pred_labels,
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    return model, trainer, val_results

if __name__ == "__main__":
    train_finllama()
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, 
                            output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = self.classification_head(last_hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def tokenize_func(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, 
                    max_length=512, return_tensors='pt')

def train_llama_with_lora(train_path="./data/cleaned_train.csv", 
                          val_path="./data/cleaned_test.csv",
                          output_dir="./model_output/llama"):
    print("\n" + "="*60)
    print("LLAMA-2 SENTIMENT FINE-TUNING WITH LORA")
    print("="*60 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    if 'Sentence' in train_df.columns:
        train_df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
        val_df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_dataset = train_dataset.map(lambda x: tokenize_func(x, tokenizer), batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(lambda x: tokenize_func(x, tokenizer), batched=True, remove_columns=['text'])
    
    train_dataset = train_dataset.rename_column("label", "labels").set_format('torch')
    val_dataset = val_dataset.rename_column("label", "labels").set_format('torch')
    
    # Load model with LoRA
    print("Loading Llama-2-7B and applying LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16, device_map="auto", load_in_8bit=True, trust_remote_code=True)
    
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)\n")
    
    # Training
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=4,
        per_device_eval_batch_size=4, learning_rate=2e-5, warmup_steps=100,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", fp16=True, optim="adamw_torch"
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics, tokenizer=tokenizer)
    
    print("Starting training...\n")
    trainer.train()
    
    # Save adapter
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"\nAdapter saved to {output_dir}")
    
    # Evaluate
    results = trainer.evaluate(eval_dataset=val_dataset)
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Validation)")
    print("="*60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
    print("="*60)
    
    return model, trainer

if __name__ == "__main__":
    train_llama_with_lora()
