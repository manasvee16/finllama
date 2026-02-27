# Financial News Sentiment Analysis with FinLLaMA

A comprehensive pipeline for fine-tuning and deploying the FinLLaMA model (Llama-2 with LoRA adapters) for financial sentiment analysis and portfolio management. This project processes financial news data to predict market sentiment and inform investment decisions.

## Project Overview

This project implements an end-to-end workflow for:

1. **Data Loading & Preprocessing**: Extract and clean financial news from multiple sources
2. **Model Fine-tuning**: Fine-tune Llama-2-7B with LoRA adapters for sentiment classification
3. **Sentiment Inference**: Run sentiment analysis on financial news articles
4. **Sentiment Aggregation**: Compute daily sentiment scores per company
5. **Portfolio Analysis**: Use sentiment signals for long-short portfolio strategies
6. **Model Evaluation**: Comprehensive analysis and benchmarking against baseline models

### Key Components

- **Baselines**: BERT and FinBERT baseline models for comparison
- **FinLLaMA**: Llama-2-7B fine-tuned with LoRA for financial sentiment analysis
- **Data Pipeline**: Multi-source financial news data loader with deduplication
- **Analysis Tools**: Error analysis, metrics computation, and model comparison utilities

## Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (for GPU support)
- **Git LFS** (for large model files)
- **HuggingFace Account** (for model access and authentication)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Manasvee16/FINLLAMA.git
cd base-paper-code
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n finllama python=3.10
conda activate finllama
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up HuggingFace Authentication

The financial news dataset is gated and requires HuggingFace authentication:

```bash
# Option 1: Interactive setup
python setup_huggingface_auth.py

# Option 2: Manual setup using CLI
huggingface-cli login
# Paste your HuggingFace API token when prompted
# Get your token at: https://huggingface.co/settings/tokens

# Option 3: Set environment variable
export HF_TOKEN="your_token_here"
```

**Important**: You also need to accept the dataset license:
- Visit: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
- Click "Agree and access dataset"

## Quick Start Guide

### Step 1: Load and Preprocess Data

```bash
# Load financial news data from HuggingFace
cd data
python financial_multisource_loader.py

# This will:
# - Download the financial news dataset
# - Remove duplicates and normalize text
# - Save data into Parquet shards
# Output: financial_news_shards/ directory with shard_0000.parquet, etc.
```

### Step 2: Prepare Labeled Data

```bash
# Auto-label data using FinBERT for sentiment
python auto_label_finbert.py

# Or preprocess existing labeled dataset
python preprocess.py

# Output: cleaned_train.csv, cleaned_test.csv in data/
```

### Step 3: Train FinLLaMA Model

```bash
cd ../model

# Train FinLLaMA with LoRA adapters
python train_finllama.py

# This will:
# - Load Llama-2-7B base model
# - Fine-tune LoRA adapters
# - Train sentiment classification head
# - Save trained weights to model_output/finllama_lora/
# - Output: training metrics and evaluation results
```

### Step 4: Run Sentiment Inference

```bash
cd ../inference

# Use trained model for inference
python sentiment_infer.py

# Or use as a class in your code:
# from sentiment_infer import SentimentInferencer
# inferencer = SentimentInferencer(adapter_path="path/to/lora/weights")
# results = inferencer.infer_batch(texts_list)
```

### Step 5: Aggregate Sentiment Scores

```bash
cd ../portfolio

# Compute daily sentiment per company
python sentiment_aggregate.py

# Output: daily_sentiment.csv with aggregated sentiment signals
```

### Step 6: Portfolio Analysis

```bash
# Generate long-short portfolio strategies based on sentiment
python long_short_portfolio.py

# Output: portfolio performance metrics and trading signals
```

## Project Structure

```
base-paper-code/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── setup_huggingface_auth.py          # HuggingFace authentication setup
├── kaggle-data.csv                    # Sample Kaggle dataset (optional)
│
├── data/                              # Data loading and preprocessing
│   ├── financial_multisource_loader.py    # Load data from HuggingFace
│   ├── auto_label_finbert.py              # Auto-label using FinBERT
│   ├── load_datasets.py                   # Dataset loading utilities
│   ├── preprocess.py                      # Text preprocessing
│   └── preprocess_news.py                 # News-specific preprocessing
│
├── baselines/                         # Baseline models for comparison
│   ├── train_bert_baseline.py             # BERT baseline
│   └── train_finbert.py                   # FinBERT baseline
│
├── model/                             # FinLLaMA model training
│   ├── train_finllama.py                  # Main training script
│   ├── finllama_lora_model.py             # LoRA model architecture
│   ├── finllama_lora.py                   # LoRA utilities
│   ├── inference_finllama.py              # Model inference utilities
│   └── train.py                           # Alternative training script
│
├── inference/                         # Inference and prediction
│   └── sentiment_infer.py                 # Sentiment inference class
│
├── portfolio/                         # Portfolio analysis tools
│   ├── sentiment_aggregate.py             # Aggregate daily sentiment
│   └── long_short_portfolio.py            # Portfolio strategies
│
├── analysis/                          # Analysis and evaluation
│   └── model_behaviour_analysis.py        # Model analysis tools
│
└── utils/                             # Utilities
    ├── metrics.py                         # Evaluation metrics
    ├── evaluate_all.py                    # Comprehensive evaluation
    ├── error_analysis.py                  # Error analysis tools
    └── hf_login.py                        # HuggingFace login utilities
```

## Running Individual Components

### Train Baseline Models

```bash
# Train BERT baseline
python baselines/train_bert_baseline.py

# Train FinBERT baseline
python baselines/train_finbert.py
```

### Evaluate All Models

```bash
# Comprehensive evaluation of all trained models
python utils/evaluate_all.py
```

### Model Behavior Analysis

```bash
# Analyze model predictions and behavior
python analysis/model_behaviour_analysis.py
```

## Configuration

Key hyperparameters and configurations can be modified in each script:

- **Model**: Change base model in `model/train_finllama.py` (default: meta-llama/Llama-2-7b-hf)
- **LoRA Rank**: Adjust `lora_r` parameter for LoRA configuration
- **Batch Size**: Modify `per_device_train_batch_size` in trainer arguments
- **Learning Rate**: Adjust `learning_rate` in training configuration
- **Max Sequence Length**: Change `max_length` in tokenization (default: 512)

## Output Structure

After running the pipeline, you'll have:

```
model_output/
├── finllama_lora/              # FinLLaMA trained weights
│   ├── adapter_config.json
│   └── adapter_model.bin
├── bert_baseline/              # BERT baseline results
└── finbert_baseline/           # FinBERT baseline results

results/
├── finllama_predictions.csv    # Model predictions
├── training_metrics.json       # Training statistics
├── evaluation_results.json     # Test set evaluation
└── sentiment_scores.csv        # Sentiment analysis results
```

## Performance Metrics

Models are evaluated using:

- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class performance metrics
- **Weighted Average**: Handling class imbalance
- **Confusion Matrix**: Detailed error analysis

## Troubleshooting

### HuggingFace Authentication Issues

```bash
# Clear cached credentials and re-authenticate
rm -rf ~/.cache/huggingface/
rm ~/.huggingface/token
python setup_huggingface_auth.py
```

### Out of Memory (CUDA)

Reduce `per_device_train_batch_size` in training scripts:

```python
per_device_train_batch_size=4,  # Reduce from 8 or 16
gradient_accumulation_steps=2,
```

### Model Download Issues

Ensure you have sufficient disk space (~15GB for Llama-2-7B) and stable internet connection. Use:

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')"
```

## Environment Variables

Key environment variables for configuration:

```bash
# HuggingFace token for authentication
export HF_TOKEN="your_token_here"

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Disable CUDA if needed
export CUDA_VISIBLE_DEVICES=""
```

## Model Weights

Pre-trained model weights are available at:

- **FinLLaMA**: `model_output/finllama_lora/`
- **BERT Baseline**: `model_output/bert_baseline/`
- **FinBERT Baseline**: `model_output/finbert_baseline/`

Download or train these models before running inference.

## Citation

If you use this project in your research, please cite:

```bibtex
@project{finllama2026,
  title={Financial News Sentiment Analysis with FinLLaMA},
  author={Your Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check existing issues for similar problems
- Review the troubleshooting section above

## References

- **Llama-2**: https://huggingface.co/meta-llama/Llama-2-7b-hf
- **LoRA**: https://huggingface.co/docs/peft/methods/lora
- **FinBERT**: https://huggingface.co/ProsusAI/finbert
- **Financial News Dataset**: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource

## Authors

Created as part of financial NLP research project.

Last Updated: February 27, 2026
