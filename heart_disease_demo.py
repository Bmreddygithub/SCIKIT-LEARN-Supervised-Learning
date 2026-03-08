"""
Heart Disease Prediction Demo - Supervised Learning System
Author: Assignment 3 Demo
Date: February 2026

This demo implements a binary classification system to predict the presence 
of heart disease based on medical attributes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    balanced_accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    roc_curve,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_heart_disease_data, get_feature_descriptions


class HeartDiseasePredictor:
    """
    Supervised Learning System for Heart Disease Prediction
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        
    def load_data(self):
        """Load and prepare the heart disease dataset"""
        print("="*70)
        print("LOADING HEART DISEASE DATASET")
        print("="*70)
        
        data = load_heart_disease_data()
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.feature_names, self.scaler = data
        
        print("\n" + "="*70)
        print("DATA LOADED SUCCESSFULLY")
        print("="*70 + "\n")
        
    def initialize_models(self):
        """Initialize multiple supervised learning models"""
        print("="*70)
        print("INITIALIZING AI MODELS")
        print("="*70)
        
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5, 
                min_samples_split=5,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        print()
        
    def train_models(self):
        """Train all models"""
        print("="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  ✓ {name} trained successfully")
        
        print("\n" + "="*70)
        print("ALL MODELS TRAINED")
        print("="*70 + "\n")
        
    def evaluate_models(self):
        """Evaluate all models with comprehensive metrics"""
        print("="*70)
        print("EVALUATING MODELS")
        print("="*70 + "\n")
        
        for name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"MODEL: {name}")
            print('='*70)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] \
                if hasattr(model, 'predict_proba') else model.decision_function(self.X_test)
            
            # Calculate metrics
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = accuracy_score(self.y_test, y_pred)
            balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'confusion_matrix': cm,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print confusion matrix
            print("\nConfusion Matrix:")
            print(f"                  Predicted")
            print(f"                  No Disease  Disease")
            print(f"Actual No Disease    {tn:4d}      {fp:4d}")
            print(f"Actual Disease       {fn:4d}      {tp:4d}")
            
            print(f"\nConfusion Matrix Components:")
            print(f"  True Negatives (TN):  {tn:4d}")
            print(f"  False Positives (FP): {fp:4d}")
            print(f"  False Negatives (FN): {fn:4d}")
            print(f"  True Positives (TP):  {tp:4d}")
            
            # Print metrics
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision:         {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall:            {recall:.4f} ({recall*100:.2f}%)")
            print(f"  F1-Score:          {f1:.4f}")
            print(f"  ROC-AUC:           {roc_auc:.4f}")
            
            # Interpretation
            print(f"\nInterpretation:")
            print(f"  - Model correctly classified {accuracy*100:.1f}% of all cases")
            print(f"  - Of patients predicted to have disease, {precision*100:.1f}% actually have it")
            print(f"  - Of patients who have disease, model detected {recall*100:.1f}%")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70 + "\n")
        
    def compare_models(self):
        """Compare all models side by side"""
        print("="*70)
        print("MODEL COMPARISON")
        print("="*70 + "\n")
        
        # Create comparison dataframe
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Balanced Acc': metrics['balanced_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model_name = df_comparison.iloc[0]['Model']
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"  F1-Score: {df_comparison.iloc[0]['F1-Score']:.4f}")
        print(f"  ROC-AUC:  {df_comparison.iloc[0]['ROC-AUC']:.4f}")
        print('='*70 + "\n")
        
        return df_comparison
        
    def visualize_results(self):
        """Create visualizations of model performance"""
        print("="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrices
        for idx, (name, metrics) in enumerate(self.results.items(), 1):
            ax = plt.subplot(3, 5, idx)
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       cbar=False, ax=ax)
            ax.set_title(f'{name}\nConfusion Matrix', fontsize=10, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=9)
            ax.set_xlabel('Predicted', fontsize=9)
        
        # 6. Metrics Comparison - Accuracy
        ax = plt.subplot(3, 5, 6)
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        bars = ax.bar(range(len(models)), accuracies, color='steelblue')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title('Accuracy Comparison', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        # 7. Metrics Comparison - Precision
        ax = plt.subplot(3, 5, 7)
        precisions = [self.results[m]['precision'] for m in models]
        bars = ax.bar(range(len(models)), precisions, color='coral')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Precision', fontsize=9)
        ax.set_title('Precision Comparison', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(precisions):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        # 8. Metrics Comparison - Recall
        ax = plt.subplot(3, 5, 8)
        recalls = [self.results[m]['recall'] for m in models]
        bars = ax.bar(range(len(models)), recalls, color='lightgreen')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Recall', fontsize=9)
        ax.set_title('Recall Comparison', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(recalls):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        # 9. Metrics Comparison - F1-Score
        ax = plt.subplot(3, 5, 9)
        f1_scores = [self.results[m]['f1_score'] for m in models]
        bars = ax.bar(range(len(models)), f1_scores, color='mediumpurple')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1-Score', fontsize=9)
        ax.set_title('F1-Score Comparison', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(f1_scores):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        # 10. ROC-AUC Comparison
        ax = plt.subplot(3, 5, 10)
        roc_aucs = [self.results[m]['roc_auc'] for m in models]
        bars = ax.bar(range(len(models)), roc_aucs, color='gold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ROC-AUC', fontsize=9)
        ax.set_title('ROC-AUC Comparison', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(roc_aucs):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        # 11-15. ROC Curves
        for idx, (name, metrics) in enumerate(self.results.items(), 11):
            ax = plt.subplot(3, 5, idx)
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            roc_auc = metrics['roc_auc']
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=9)
            ax.set_ylabel('True Positive Rate', fontsize=9)
            ax.set_title(f'{name}\nROC Curve', fontsize=10, fontweight='bold')
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('heart_disease_evaluation.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'heart_disease_evaluation.png'\n")
        
        # Show plot
        plt.show()
        
    def demonstrate_prediction(self, n_samples=5):
        """Demonstrate predictions on test samples"""
        print("="*70)
        print("DEMONSTRATION: SAMPLE PREDICTIONS")
        print("="*70 + "\n")
        
        # Select best model (by F1-score)
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['f1_score'])[0]
        best_model = self.models[best_model_name]
        
        print(f"Using best model: {best_model_name}\n")
        
        # Select random samples from test set
        np.random.seed(42)
        indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        for i, idx in enumerate(indices, 1):
            X_sample = self.X_test[idx:idx+1]
            y_actual = self.y_test[idx]
            
            y_pred = best_model.predict(X_sample)[0]
            y_proba = best_model.predict_proba(X_sample)[0, 1] \
                if hasattr(best_model, 'predict_proba') else None
            
            print(f"Sample {i}:")
            print(f"  Actual:     {'Disease' if y_actual == 1 else 'No Disease'} (class {y_actual})")
            print(f"  Predicted:  {'Disease' if y_pred == 1 else 'No Disease'} (class {y_pred})")
            if y_proba is not None:
                print(f"  Confidence: {y_proba*100:.1f}% probability of disease")
            print(f"  Result:     {'✓ Correct' if y_pred == y_actual else '✗ Incorrect'}")
            print()
        
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("\n" + "="*70)
        print(" "*15 + "HEART DISEASE PREDICTION SYSTEM")
        print(" "*20 + "Supervised Learning Demo")
        print("="*70 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Evaluate models
        self.evaluate_models()
        
        # Step 5: Compare models
        comparison_df = self.compare_models()
        
        # Step 6: Demonstrate predictions
        self.demonstrate_prediction()
        
        # Step 7: Visualize results
        self.visualize_results()
        
        print("="*70)
        print(" "*25 + "DEMO COMPLETE")
        print("="*70 + "\n")
        
        return comparison_df


def main():
    """Main function to run the demo"""
    predictor = HeartDiseasePredictor()
    results = predictor.run_full_demo()
    
    print("\nTo view detailed results, check:")
    print("  - Console output above")
    print("  - heart_disease_evaluation.png (visualization)")
    

if __name__ == "__main__":
    main()
