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

if __name__ == "__main__":
    model = create_finllama_with_lora()
    print(model)
