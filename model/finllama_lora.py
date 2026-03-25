# Copilot Prompt:
# Implement FinLlama model using HuggingFace Transformers.
# Base model: "meta-llama/Llama-2-7b-hf".
# Add a classification head with SoftMax over 3 classes.
# Apply LoRA using peft library:
# - r=8
# - lora_alpha=32
# - target modules: q_proj, v_proj
# Freeze base model weights.
# Return a PEFT-wrapped model ready for training.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class FinLlamaModel(nn.Module):
    """
    FinLlama model with LoRA adapter and classification head.
    """
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", num_classes=3):
        super().__init__()
        
        # Load base Llama-2-7B model for sequence classification
        from transformers import AutoModelForSequenceClassification
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            trust_remote_code=True
        )
        
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        """
        # Get outputs from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {'loss': outputs.loss, 'logits': outputs.logits}

def create_finllama_with_lora(model_name="meta-llama/Llama-2-7b-hf"):
    """
    Create FinLlama model with LoRA adapter using Llama-2-7B as base.
    """
    # Load base model for sequence classification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        trust_remote_code=True
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
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    # Freeze base model weights
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")
    
    return model

def get_lora_model(base_model="meta-llama/Llama-2-7b-hf",
                   r=8,
                   lora_alpha=32,
                   lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj"],
                   num_labels=3):
    """
    Get LoRA-wrapped model for sequence classification.
    
    Uses PEFT library to apply LoRA adapters to Llama-2-7b-hf base model.
    All base model parameters are frozen, only LoRA adapters are trainable.
    
    Args:
        base_model: Base model name (default: meta-llama/Llama-2-7b-hf)
        r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling factor (default: 32)
        lora_dropout: Dropout rate for LoRA layers (default: 0.05)
        target_modules: List of module names to apply LoRA (default: ["q_proj", "v_proj"])
        num_labels: Number of classification labels (default: 3 for sentiment)
    
    Returns:
        PEFT-wrapped model with LoRA adapters, ready for fine-tuning.
        Classification head is included for sequence classification.
    """
    print(f"\n[INFO] Loading base model: {base_model}")
    
    # Load base model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        trust_remote_code=True
    )
    
    print(f"[INFO] Applying LoRA configuration:")
    print(f"  r={r}")
    print(f"  lora_alpha={lora_alpha}")
    print(f"  lora_dropout={lora_dropout}")
    print(f"  target_modules={target_modules}")
    print(f"  task_type=SEQ_CLS")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Freeze all base model parameters - only LoRA layers trainable
    for name, param in model.named_parameters():
        if 'lora' not in name and 'classifier' not in name:
            param.requires_grad = False
    
    # Print parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Parameter statistics:")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model

if __name__ == "__main__":
    model = get_lora_model()
    print("\n[OK] Model created successfully!")
    print(model)
