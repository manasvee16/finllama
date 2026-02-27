# Copilot Prompt:
# Write training script for FinLlama.
# Load cleaned_train.csv and cleaned_test.csv.
# Tokenize text using Llama-2 tokenizer.
# Use Trainer API from HuggingFace.
# Training settings:
# - epochs=5
# - batch_size=4
# - learning_rate=2e-5
# - optimizer=AdamW
# - evaluation strategy every epoch
# Save fine-tuned LoRA adapter weights.

import pandas as pd
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
import os

def load_and_tokenize_data(train_csv, test_csv, tokenizer_name="meta-llama/Llama-2-7b-hf"):
    """
    Load CSV files and tokenize text using Llama-2 tokenizer.
    """
    # Load Llama-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Convert to HF Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['Sentence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Rename label column
    train_dataset = train_dataset.rename_column('Sentiment', 'labels')
    test_dataset = test_dataset.rename_column('Sentiment', 'labels')
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return train_dataset, test_dataset, tokenizer

def create_model_with_lora(num_labels=3):
    """
    Create model with LoRA configuration.
    """
    # Load base model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model

def train_finllama(train_csv, test_csv, output_dir="./model_output"):
    """
    Train FinLlama model using HuggingFace Trainer.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and tokenize data
    train_dataset, test_dataset, tokenizer = load_and_tokenize_data(train_csv, test_csv)
    
    # Create model with LoRA
    model = create_model_with_lora()
    
    # Training arguments
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
        metric_for_best_model="accuracy",
        optim="adamw_torch"
    )
    
    # Define compute metrics function
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save LoRA adapter weights
    model.save_pretrained(os.path.join(output_dir, "finllama_lora_adapter"))
    print(f"LoRA adapter saved to {os.path.join(output_dir, 'finllama_lora_adapter')}")
    
    return model, trainer

if __name__ == "__main__":
    train_finllama("./cleaned_train.csv", "./cleaned_test.csv")
