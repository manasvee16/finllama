# Copilot Prompt:
# Load test texts and predictions from FinBERT and FinLlama.
# Print examples where FinLlama is correct and FinBERT is wrong.
# Print examples where FinBERT is correct and FinLlama is wrong.
# Save examples for qualitative analysis.

import os
import pandas as pd
import numpy as np
from pathlib import Path

class ErrorAnalyzer:
    """
    Analyze prediction errors between FinBERT and FinLlama models.
    Identifies cases where one model is correct and the other is wrong.
    """
    
    def __init__(self, test_texts_path="./data/cleaned_test.csv"):
        """
        Initialize analyzer with test data.
        
        Args:
            test_texts_path: Path to cleaned test CSV with 'Sentence' and 'Sentiment' columns
        """
        self.test_texts_path = test_texts_path
        self.test_data = None
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        # Load test data
        if os.path.exists(test_texts_path):
            self.test_data = pd.read_csv(test_texts_path)
            print(f"Loaded {len(self.test_data)} test samples from {test_texts_path}")
        else:
            print(f"Warning: Test data not found at {test_texts_path}")
    
    def load_predictions(self, model_name, pred_dir="./"):
        """
        Load model predictions from CSV.
        
        Args:
            model_name: Name of model (e.g., 'FinBERT', 'FinLlama')
            pred_dir: Directory containing prediction files
        
        Returns:
            Array of predictions or None if not found
        """
        pred_path = os.path.join(pred_dir, f"{model_name}_predictions.csv")
        
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            # Try different column names
            if 'prediction' in df.columns:
                return df['prediction'].values
            elif 'predicted_label' in df.columns:
                return df['predicted_label'].values
            else:
                # Try second column
                return df.iloc[:, 1].values
        else:
            print(f"Warning: Predictions for {model_name} not found at {pred_path}")
            return None
    
    def analyze_disagreements(self, finbert_pred_dir="./", finllama_pred_dir="./", 
                             output_dir="./results", num_examples=10):
        """
        Identify and print examples where models disagree.
        
        Args:
            finbert_pred_dir: Directory with FinBERT predictions
            finllama_pred_dir: Directory with FinLlama predictions
            output_dir: Directory to save results
            num_examples: Number of examples to print for each category
        """
        if self.test_data is None:
            print("Test data not loaded. Cannot analyze.")
            return
        
        print("\n" + "="*80)
        print("ERROR ANALYSIS: FinBERT vs FinLlama")
        print("="*80)
        
        # Load predictions
        finbert_preds = self.load_predictions('FinBERT', finbert_pred_dir)
        finllama_preds = self.load_predictions('FinLlama', finllama_pred_dir)
        
        if finbert_preds is None or finllama_preds is None:
            print("Cannot load predictions from both models.")
            return
        
        # Get true labels and test texts
        true_labels = self.test_data['Sentiment'].values if 'Sentiment' in self.test_data.columns else None
        test_texts = self.test_data['Sentence'].values if 'Sentence' in self.test_data.columns else None
        
        if true_labels is None or test_texts is None:
            print("Cannot find required columns in test data.")
            return
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'text': test_texts,
            'true_label': true_labels,
            'finbert_pred': finbert_preds,
            'finllama_pred': finllama_preds
        })
        
        # Add correctness flags
        analysis_df['finbert_correct'] = analysis_df['finbert_pred'] == analysis_df['true_label']
        analysis_df['finllama_correct'] = analysis_df['finllama_pred'] == analysis_df['true_label']
        
        # Cases where FinLlama is correct but FinBERT is wrong
        finllama_correct_finbert_wrong = analysis_df[
            (analysis_df['finllama_correct']) & (~analysis_df['finbert_correct'])
        ]
        
        # Cases where FinBERT is correct but FinLlama is wrong
        finbert_correct_finllama_wrong = analysis_df[
            (analysis_df['finbert_correct']) & (~analysis_df['finllama_correct'])
        ]
        
        # Print statistics
        print(f"\nTotal test samples: {len(analysis_df)}")
        print(f"FinBERT correct: {analysis_df['finbert_correct'].sum()}")
        print(f"FinLlama correct: {analysis_df['finllama_correct'].sum()}")
        print(f"\nFinLlama correct but FinBERT wrong: {len(finllama_correct_finbert_wrong)}")
        print(f"FinBERT correct but FinLlama wrong: {len(finbert_correct_finllama_wrong)}")
        
        # Print examples
        print("\n" + "="*80)
        print("EXAMPLES: FinLlama Correct, FinBERT Wrong")
        print("="*80)
        self._print_examples(finllama_correct_finbert_wrong.head(num_examples))
        
        print("\n" + "="*80)
        print("EXAMPLES: FinBERT Correct, FinLlama Wrong")
        print("="*80)
        self._print_examples(finbert_correct_finllama_wrong.head(num_examples))
        
        # Save to CSV files
        os.makedirs(output_dir, exist_ok=True)
        
        finllama_better_path = os.path.join(output_dir, "finllama_correct_finbert_wrong.csv")
        finllama_correct_finbert_wrong.to_csv(finllama_better_path, index=False)
        print(f"\nSaved FinLlama correct examples to {finllama_better_path}")
        
        finbert_better_path = os.path.join(output_dir, "finbert_correct_finllama_wrong.csv")
        finbert_correct_finllama_wrong.to_csv(finbert_better_path, index=False)
        print(f"Saved FinBERT correct examples to {finbert_better_path}")
        
        # Save full analysis
        analysis_path = os.path.join(output_dir, "error_analysis_full.csv")
        analysis_df.to_csv(analysis_path, index=False)
        print(f"Saved full analysis to {analysis_path}")
        
        return analysis_df
    
    def _print_examples(self, df, max_examples=10):
        """
        Print example cases for manual inspection.
        
        Args:
            df: DataFrame with examples
            max_examples: Maximum number of examples to print
        """
        for idx, (i, row) in enumerate(df.iterrows()):
            if idx >= max_examples:
                break
            
            print(f"\n{idx+1}. Text: {row['text'][:120]}...")
            print(f"   True Label:      {self.label_map[row['true_label']]}")
            print(f"   FinBERT Pred:    {self.label_map[row['finbert_pred']]}")
            print(f"   FinLlama Pred:   {self.label_map[row['finllama_pred']]}")
            print(f"   {'-'*75}")

if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.analyze_disagreements()
