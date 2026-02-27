# Copilot Prompt:
# Load base Llama-2-7B model.
# Load trained LoRA adapter weights.
# Accept a raw news text input.
# Tokenize input and run model inference.
# Output sentiment label and confidence score.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np

class FinLlamaInferencer:
    """
    Inference class for FinLlama sentiment classification.
    Loads base Llama-2-7B model and trained LoRA adapter weights.
    """
    
    def __init__(self, base_model="meta-llama/Llama-2-7b-hf", adapter_path=None):
        """
        Initialize with base Llama-2-7B model and LoRA adapter.
        
        Args:
            base_model: Base model name (default: meta-llama/Llama-2-7b-hf)
            adapter_path: Path to trained LoRA adapter weights
        """
        print(f"Loading Llama-2-7B base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model for sequence classification
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            trust_remote_code=True
        )
        
        # Load trained LoRA adapter if provided
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Label mapping
        self.id_to_label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
    
    def infer(self, text):
        """
        Run inference on a news text input.
        
        Args:
            text: News article text
        
        Returns:
            Dictionary with sentiment_label and confidence_score
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Get confidence score (max probability after softmax)
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        confidence_score = float(probabilities[predicted_class])
        
        # Get label
        sentiment_label = self.id_to_label[predicted_class]
        
        result = {
            'sentiment_label': sentiment_label,
            'confidence_score': confidence_score,
            'predicted_class': predicted_class,
            'probabilities': {
                self.id_to_label[i]: float(p) for i, p in enumerate(probabilities)
            }
        }
        
        return result
    
    def infer_batch(self, texts):
        """
        Run batch inference on multiple texts.
        
        Args:
            texts: List of news article texts
        
        Returns:
            List of result dictionaries
        """
        results = []
        for text in texts:
            result = self.infer(text)
            results.append(result)
        return results

if __name__ == "__main__":
    # Example usage
    sample_text = "Apple stock surged 5% today on strong quarterly earnings."
    
    # Initialize inferencer (without adapter for demo)
    inferencer = FinLlamaInferencer()
    
    # Run inference
    result = inferencer.infer(sample_text)
    
    print(f"\nText: {sample_text}")
    print(f"Sentiment Label: {result['sentiment_label']}")
    print(f"Confidence Score: {result['confidence_score']:.4f}")
    print(f"All Probabilities: {result['probabilities']}")
