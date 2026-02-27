# Copilot Prompt:
# Load predictions from FinBERT, BERT, and FinLlama.
# Load true test labels.
# Compute accuracy, precision, recall, and F1-score for each model.
# Display results in a comparison table.
# Plot confusion matrices.

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_predictions(model_name, pred_dir="./"):
    """
    Load predictions from CSV file.
    """
    pred_path = os.path.join(pred_dir, f"{model_name}_predictions.csv")
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        return df['prediction'].values if 'prediction' in df.columns else df.iloc[:, 1].values
    else:
        print(f"Warning: {pred_path} not found")
        return None

def load_true_labels(test_csv_path="./data/cleaned_test.csv"):
    """
    Load true labels from test dataset.
    """
    if os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
        return df['Sentiment'].values
    else:
        print(f"Warning: {test_csv_path} not found")
        return None

def compute_metrics(y_true, y_pred, model_name):
    """
    Compute evaluation metrics.
    """
    if y_pred is None:
        print(f"Skipping {model_name} - predictions not available")
        return None
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir="./results"):
    """
    Plot confusion matrix for a model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

def evaluate_all_models(test_csv_path="./data/cleaned_test.csv", 
                       pred_dir="./", 
                       output_dir="./results"):
    """
    Evaluate all models and compare results.
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*70)
    
    # Load true labels
    y_true = load_true_labels(test_csv_path)
    if y_true is None:
        print("Cannot load true labels. Exiting.")
        return None
    
    print(f"Loaded {len(y_true)} true labels from {test_csv_path}")
    
    # Models to evaluate
    models = ['FinBERT', 'BERT', 'FinLlama']
    
    results = []
    
    # Compute metrics for each model
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        
        y_pred = load_predictions(model_name, pred_dir)
        
        if y_pred is not None:
            metrics = compute_metrics(y_true, y_pred, model_name)
            if metrics:
                results.append(metrics)
                
                # Plot confusion matrix
                plot_confusion_matrix(y_true, y_pred, model_name, output_dir)
                
                # Print detailed classification report
                print(f"\n{model_name} Classification Report:")
                print(classification_report(y_true, y_pred, 
                                          target_names=['Negative', 'Neutral', 'Positive'],
                                          zero_division=0))
    
    # Create comparison table
    if results:
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON TABLE")
        print("="*70)
        print(results_df.to_string(index=False))
        print("="*70)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "model_comparison.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved comparison results to {results_path}")
        
        # Determine best model
        best_model_idx = results_df['F1-Score'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        print(f"\nBest Model: {best_model['Model']} (F1-Score: {best_model['F1-Score']:.4f})")
        
        return results_df
    
    return None

if __name__ == "__main__":
    results = evaluate_all_models()
