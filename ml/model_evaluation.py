"""
Advanced Model Evaluation and Performance Analysis
================================================

This script provides comprehensive evaluation of the stroke prediction models
including performance metrics, visualizations, and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           precision_score, recall_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import pickle
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    """
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def load_models(self, model_prefix='advanced_stroke_model'):
        """Load all trained models and components."""
        print("📁 Loading models and components...")
        
        try:
            # Load ensemble model
            self.models['ensemble'] = joblib.load(f'{model_prefix}_ensemble.pkl')
            
            # Load individual models
            model_names = ['xgboost', 'lightgbm', 'catboost', 'randomforest', 'neuralnetwork']
            for name in model_names:
                try:
                    self.models[name] = joblib.load(f'{model_prefix}_{name}.pkl')
                except:
                    print(f"⚠️ Could not load {name} model")
            
            # Load scaler and feature components
            self.scalers['main'] = joblib.load(f'{model_prefix}_scaler.pkl')
            self.feature_columns = joblib.load(f'{model_prefix}_features.pkl')
            self.feature_selector = joblib.load(f'{model_prefix}_selector.pkl')
            
            # Load results
            with open(f'{model_prefix}_results.pkl', 'rb') as f:
                self.results = pickle.load(f)
            
            print("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def evaluate_model_performance(self, X, y, model, model_name):
        """Comprehensive model evaluation."""
        print(f"📊 Evaluating {model_name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'cv_scores': cv_scores
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, ax=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        return ax
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, ax=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name, ax=None):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(recall, precision, label=model_name)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def create_performance_dashboard(self, X, y):
        """Create comprehensive performance dashboard."""
        print("📈 Creating performance dashboard...")
        
        # Evaluate all models
        model_results = {}
        
        for name, model in self.models.items():
            try:
                result = self.evaluate_model_performance(X, y, model, name)
                model_results[name] = result
            except Exception as e:
                print(f"⚠️ Error evaluating {name}: {str(e)}")
                continue
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model comparison metrics
        ax1 = plt.subplot(3, 4, 1)
        model_names = list(model_results.keys())
        auc_scores = [model_results[name]['metrics']['auc'] for name in model_names]
        
        bars = ax1.bar(model_names, auc_scores, color='skyblue')
        ax1.set_title('Model AUC Comparison')
        ax1.set_ylabel('AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Accuracy comparison
        ax2 = plt.subplot(3, 4, 2)
        accuracy_scores = [model_results[name]['metrics']['accuracy'] for name in model_names]
        
        bars = ax2.bar(model_names, accuracy_scores, color='lightgreen')
        ax2.set_title('Model Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, accuracy_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. F1-Score comparison
        ax3 = plt.subplot(3, 4, 3)
        f1_scores = [model_results[name]['metrics']['f1'] for name in model_names]
        
        bars = ax3.bar(model_names, f1_scores, color='orange')
        ax3.set_title('Model F1-Score Comparison')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 4. Cross-validation scores
        ax4 = plt.subplot(3, 4, 4)
        cv_means = [model_results[name]['metrics']['cv_mean'] for name in model_names]
        cv_stds = [model_results[name]['metrics']['cv_std'] for name in model_names]
        
        bars = ax4.bar(model_names, cv_means, yerr=cv_stds, color='lightcoral', capsize=5)
        ax4.set_title('Cross-Validation Scores')
        ax4.set_ylabel('CV AUC Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. ROC Curves
        ax5 = plt.subplot(3, 4, 5)
        for name in model_names:
            if name in model_results:
                self.plot_roc_curve(y, model_results[name]['probabilities'], name, ax5)
        ax5.set_title('ROC Curves Comparison')
        
        # 6. Precision-Recall Curves
        ax6 = plt.subplot(3, 4, 6)
        for name in model_names:
            if name in model_results:
                self.plot_precision_recall_curve(y, model_results[name]['probabilities'], name, ax6)
        ax6.set_title('Precision-Recall Curves')
        
        # 7. Confusion Matrix for best model
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['auc'])
        ax7 = plt.subplot(3, 4, 7)
        self.plot_confusion_matrix(y, model_results[best_model]['predictions'], best_model, ax7)
        
        # 8. Feature importance (if available)
        ax8 = plt.subplot(3, 4, 8)
        try:
            if hasattr(self.models[best_model], 'feature_importances_'):
                importances = self.models[best_model].feature_importances_
                feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(len(importances))]
                
                # Get top 10 features
                indices = np.argsort(importances)[::-1][:10]
                ax8.barh(range(10), importances[indices])
                ax8.set_yticks(range(10))
                ax8.set_yticklabels([feature_names[i] for i in indices])
                ax8.set_title(f'Top 10 Features - {best_model.title()}')
                ax8.set_xlabel('Feature Importance')
        except:
            ax8.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
            ax8.set_title('Feature Importance')
        
        # 9. Model performance heatmap
        ax9 = plt.subplot(3, 4, 9)
        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': model_results[name]['metrics']['accuracy'],
                'Precision': model_results[name]['metrics']['precision'],
                'Recall': model_results[name]['metrics']['recall'],
                'F1-Score': model_results[name]['metrics']['f1'],
                'AUC': model_results[name]['metrics']['auc']
            } for name in model_names
        }).T
        
        sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', ax=ax9, fmt='.3f')
        ax9.set_title('Performance Metrics Heatmap')
        
        # 10. Model ranking
        ax10 = plt.subplot(3, 4, 10)
        ranking_data = []
        for name in model_names:
            if name in model_results:
                ranking_data.append({
                    'Model': name,
                    'Score': model_results[name]['metrics']['auc']
                })
        
        ranking_df = pd.DataFrame(ranking_data).sort_values('Score', ascending=True)
        bars = ax10.barh(ranking_df['Model'], ranking_df['Score'], color='lightblue')
        ax10.set_title('Model Ranking (by AUC)')
        ax10.set_xlabel('AUC Score')
        
        # 11. Precision vs Recall scatter
        ax11 = plt.subplot(3, 4, 11)
        precisions = [model_results[name]['metrics']['precision'] for name in model_names]
        recalls = [model_results[name]['metrics']['recall'] for name in model_names]
        
        scatter = ax11.scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax11.annotate(name, (recalls[i], precisions[i]), xytext=(5, 5), textcoords='offset points')
        
        ax11.set_xlabel('Recall')
        ax11.set_ylabel('Precision')
        ax11.set_title('Precision vs Recall')
        ax11.grid(True, alpha=0.3)
        
        # 12. Model stability (CV std)
        ax12 = plt.subplot(3, 4, 12)
        cv_stds = [model_results[name]['metrics']['cv_std'] for name in model_names]
        bars = ax12.bar(model_names, cv_stds, color='lightpink')
        ax12.set_title('Model Stability (CV Std)')
        ax12.set_ylabel('CV Standard Deviation')
        ax12.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('advanced_model_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return model_results
    
    def generate_report(self, model_results):
        """Generate comprehensive evaluation report."""
        print("📝 Generating evaluation report...")
        
        report = []
        report.append("# Advanced Stroke Prediction Model Evaluation Report")
        report.append("=" * 60)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model performance summary
        report.append("## Model Performance Summary")
        report.append("")
        
        # Create performance table
        performance_data = []
        for name, result in model_results.items():
            metrics = result['metrics']
            performance_data.append({
                'Model': name.title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'CV Mean': f"{metrics['cv_mean']:.4f}",
                'CV Std': f"{metrics['cv_std']:.4f}"
            })
        
        df_performance = pd.DataFrame(performance_data)
        report.append(df_performance.to_string(index=False))
        report.append("")
        
        # Best model analysis
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['auc'])
        best_metrics = model_results[best_model]['metrics']
        
        report.append("## Best Performing Model")
        report.append(f"**Model**: {best_model.title()}")
        report.append(f"**AUC Score**: {best_metrics['auc']:.4f}")
        report.append(f"**Accuracy**: {best_metrics['accuracy']:.4f}")
        report.append(f"**F1-Score**: {best_metrics['f1']:.4f}")
        report.append(f"**Cross-Validation Mean**: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
        report.append("")
        
        # Model comparison
        report.append("## Model Comparison")
        report.append("")
        
        # Rank models by AUC
        ranked_models = sorted(model_results.items(), key=lambda x: x[1]['metrics']['auc'], reverse=True)
        
        report.append("### Ranking by AUC Score:")
        for i, (name, result) in enumerate(ranked_models, 1):
            auc = result['metrics']['auc']
            report.append(f"{i}. {name.title()}: {auc:.4f}")
        report.append("")
        
        # Performance insights
        report.append("## Performance Insights")
        report.append("")
        
        # Calculate average performance
        avg_auc = np.mean([result['metrics']['auc'] for result in model_results.values()])
        avg_accuracy = np.mean([result['metrics']['accuracy'] for result in model_results.values()])
        
        report.append(f"- **Average AUC Score**: {avg_auc:.4f}")
        report.append(f"- **Average Accuracy**: {avg_accuracy:.4f}")
        report.append(f"- **Number of Models Evaluated**: {len(model_results)}")
        report.append("")
        
        # Model stability analysis
        cv_stds = [result['metrics']['cv_std'] for result in model_results.values()]
        most_stable = min(model_results.keys(), key=lambda x: model_results[x]['metrics']['cv_std'])
        
        report.append("### Model Stability Analysis")
        report.append(f"- **Most Stable Model**: {most_stable.title()} (CV Std: {model_results[most_stable]['metrics']['cv_std']:.4f})")
        report.append(f"- **Average CV Standard Deviation**: {np.mean(cv_stds):.4f}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("### For Production Use:")
        report.append(f"- **Primary Model**: {best_model.title()} (highest AUC)")
        report.append(f"- **Backup Model**: {most_stable.title()} (most stable)")
        report.append("")
        
        report.append("### For Further Improvement:")
        report.append("- Consider ensemble methods combining top 3 models")
        report.append("- Implement advanced feature engineering")
        report.append("- Use more sophisticated hyperparameter optimization")
        report.append("- Collect more diverse training data")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('model_evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Evaluation report saved as 'model_evaluation_report.txt'")
        return report_text

def main():
    """Main evaluation function."""
    print("🔍 Advanced Model Evaluation System")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    if not evaluator.load_models():
        print("❌ Could not load models. Please run advanced_model_training.py first.")
        return
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    # Apply same preprocessing as training
    from advanced_model_training import AdvancedStrokePredictor
    predictor = AdvancedStrokePredictor()
    processed_data = predictor.advanced_preprocessing(data)
    
    # Prepare features and target
    target_column = 'stroke'
    feature_columns = [col for col in processed_data.columns if col != target_column]
    X = processed_data[feature_columns]
    y = processed_data[target_column]
    
    # Apply feature selection
    X_selected, selected_features = predictor.feature_selection(X, y, method='rfe', k=25)
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    # Scale features
    X_scaled = evaluator.scalers['main'].transform(X_selected)
    
    # Create performance dashboard
    model_results = evaluator.create_performance_dashboard(X_scaled, y)
    
    # Generate report
    report = evaluator.generate_report(model_results)
    
    print("\n📈 Evaluation Complete!")
    print("=" * 30)
    print("Files generated:")
    print("- advanced_model_performance_dashboard.png")
    print("- model_evaluation_report.txt")
    
    return evaluator, model_results

if __name__ == "__main__":
    evaluator, results = main()
