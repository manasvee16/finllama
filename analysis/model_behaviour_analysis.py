# Copilot Prompt:
# Load test texts and model predictions.
# Identify examples where FinLlama is correct but FinBERT is wrong.
# Identify examples where FinBERT is correct but FinLlama is wrong.
# Print 20 such examples for manual inspection.

import pandas as pd
import numpy as np
import os

class ModelBehaviorAnalyzer:
    """
    Analyze where different models disagree and make correct/incorrect predictions.
    """
    
    def __init__(self, test_csv_path="./data/cleaned_test.csv"):
        """
        Initialize with test data.
        """
        self.test_data = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else None
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def load_predictions(self, model_name, pred_dir="./"):
        """
        Load model predictions from CSV.
        """
        pred_path = os.path.join(pred_dir, f"{model_name}_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            return df['prediction'].values if 'prediction' in df.columns else df.iloc[:, 1].values
        else:
            print(f"Warning: Predictions for {model_name} not found at {pred_path}")
            return None
    
    def analyze_disagreements(self, pred_dir="./", output_dir="./results"):
        """
        Analyze where FinLlama and FinBERT disagree.
        """
        if self.test_data is None:
            print("Test data not loaded. Cannot analyze.")
            return
        
        print("\n" + "="*80)
        print("MODEL BEHAVIOR ANALYSIS: FinLlama vs FinBERT")
        print("="*80)
        
        # Load predictions
        finllama_preds = self.load_predictions('FinLlama', pred_dir)
        finbert_preds = self.load_predictions('FinBERT', pred_dir)
        true_labels = self.test_data['Sentiment'].values
        
        if finllama_preds is None or finbert_preds is None:
            print("Cannot load predictions. Exiting.")
            return
        
        # Get test texts
        test_texts = self.test_data['Sentence'].values
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'text': test_texts,
            'true_label': true_labels,
            'finllama_pred': finllama_preds,
            'finbert_pred': finbert_preds,
            'finllama_correct': finllama_preds == true_labels,
            'finbert_correct': finbert_preds == true_labels
        })
        
        # Cases where FinLlama is correct but FinBERT is wrong
        finllama_correct_finbert_wrong = analysis_df[
            (analysis_df['finllama_correct']) & (~analysis_df['finbert_correct'])
        ]
        
        # Cases where FinBERT is correct but FinLlama is wrong
        finbert_correct_finllama_wrong = analysis_df[
            (analysis_df['finbert_correct']) & (~analysis_df['finllama_correct'])
        ]
        
        # Print analysis results
        print(f"\nTotal test samples: {len(analysis_df)}")
        print(f"FinLlama correct: {analysis_df['finllama_correct'].sum()}")
        print(f"FinBERT correct: {analysis_df['finbert_correct'].sum()}")
        print(f"Both correct: {((analysis_df['finllama_correct']) & (analysis_df['finbert_correct'])).sum()}")
        print(f"Both wrong: {((~analysis_df['finllama_correct']) & (~analysis_df['finbert_correct'])).sum()}")
        
        # Cases where only FinLlama is correct
        print(f"\nFinLlama correct but FinBERT wrong: {len(finllama_correct_finbert_wrong)}")
        print(f"FinBERT correct but FinLlama wrong: {len(finbert_correct_finllama_wrong)}")
        
        # Print examples
        print("\n" + "="*80)
        print("EXAMPLES: FinLlama Correct, FinBERT Wrong (up to 20)")
        print("="*80)
        self._print_examples(finllama_correct_finbert_wrong.head(20))
        
        print("\n" + "="*80)
        print("EXAMPLES: FinBERT Correct, FinLlama Wrong (up to 20)")
        print("="*80)
        self._print_examples(finbert_correct_finllama_wrong.head(20))
        
        # Save analysis
        os.makedirs(output_dir, exist_ok=True)
        
        analysis_path = os.path.join(output_dir, "model_behavior_analysis.csv")
        analysis_df.to_csv(analysis_path, index=False)
        print(f"\n\nSaved full analysis to {analysis_path}")
        
        # Save specific cases
        finllama_better_path = os.path.join(output_dir, "finllama_better.csv")
        finllama_correct_finbert_wrong.to_csv(finllama_better_path, index=False)
        print(f"Saved FinLlama better cases to {finllama_better_path}")
        
        finbert_better_path = os.path.join(output_dir, "finbert_better.csv")
        finbert_correct_finllama_wrong.to_csv(finbert_better_path, index=False)
        print(f"Saved FinBERT better cases to {finbert_better_path}")
        
        return analysis_df
    
    def _print_examples(self, df, max_examples=20):
        """
        Print example cases for manual inspection.
        """
        for idx, (i, row) in enumerate(df.iterrows()):
            if idx >= max_examples:
                break
            
            print(f"\n{idx+1}. Text: {row['text'][:100]}...")
            print(f"   True Label:      {self.label_map[row['true_label']]}")
            print(f"   FinLlama Pred:   {self.label_map[row['finllama_pred']]}")
            print(f"   FinBERT Pred:    {self.label_map[row['finbert_pred']]}")
            print(f"   {'-'*75}")
    
    def error_analysis_by_label(self, pred_dir="./", output_dir="./results"):
        """
        Analyze prediction errors broken down by true label.
        """
        if self.test_data is None:
            print("Test data not loaded.")
            return
        
        finllama_preds = self.load_predictions('FinLlama', pred_dir)
        finbert_preds = self.load_predictions('FinBERT', pred_dir)
        true_labels = self.test_data['Sentiment'].values
        
        if finllama_preds is None or finbert_preds is None:
            return
        
        print("\n" + "="*80)
        print("ERROR ANALYSIS BY SENTIMENT CLASS")
        print("="*80)
        
        for label_id, label_name in self.label_map.items():
            mask = true_labels == label_id
            count = mask.sum()
            
            finllama_correct = (finllama_preds[mask] == true_labels[mask]).sum()
            finbert_correct = (finbert_preds[mask] == true_labels[mask]).sum()
            
            finllama_acc = 100 * finllama_correct / count if count > 0 else 0
            finbert_acc = 100 * finbert_correct / count if count > 0 else 0
            
            print(f"\n{label_name} ({count} samples):")
            print(f"  FinLlama Accuracy:  {finllama_acc:.2f}%")
            print(f"  FinBERT Accuracy:   {finbert_acc:.2f}%")
            print(f"  Difference:         {finllama_acc - finbert_acc:+.2f}%")

if __name__ == "__main__":
    analyzer = ModelBehaviorAnalyzer()
    analyzer.analyze_disagreements()
    analyzer.error_analysis_by_label()
