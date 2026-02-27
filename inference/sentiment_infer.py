# Copilot Prompt:
# Write inference script.
# Load base Llama-2-7B and LoRA adapter weights.
# Accept a news article text input.
# Output:
# - sentiment label (positive/neutral/negative)
# - sentiment strength (SoftMax probability).

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F

class SentimentInferencer:
    """
    Inference class for FinLlama sentiment classification.
    """
    
    def __init__(self, base_model="meta-llama/Llama-2-7b-hf", adapter_path=None):
        """
        Initialize inferencer with base Llama-2 model and LoRA adapter.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base Llama-2-7B model for sequence classification
        from transformers import AutoModelForSequenceClassification
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        
        # Sentiment label mapping
        self.id_to_label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
    
    def infer(self, text, return_probabilities=True):
        """
        Run inference on a news article text.
        
        Args:
            text: News article text
            return_probabilities: If True, return softmax probabilities
        
        Returns:
            Dictionary with sentiment label and strength
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
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy()
        sentiment_strength = probabilities[predicted_class]
        
        # Get label
        sentiment_label = self.id_to_label[predicted_class]
        
        result = {
            'sentiment_label': sentiment_label,
            'sentiment_strength': float(sentiment_strength),
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
    """
    inferencer = SentimentInferencer(base_model, adapter_path)
    results = []
    
    for text in texts:
        result = inferencer.infer(text)
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_text = "Apple stock surged 5% today on strong quarterly earnings."
    
    inferencer = SentimentInferencer()
    result = inferencer.infer(sample_text)
    
    print(f"Text: {sample_text}")
    print(f"Sentiment: {result['sentiment_label']}")
    print(f"Strength: {result['sentiment_strength']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
