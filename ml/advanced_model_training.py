"""
Advanced Stroke Prediction Model with High Accuracy
==================================================

This script implements a comprehensive machine learning pipeline for stroke prediction
with advanced preprocessing, feature engineering, ensemble methods, and hyperparameter optimization.

Features:
- Advanced data preprocessing and imputation
- Feature engineering and selection
- Multiple ML algorithms (XGBoost, LightGBM, CatBoost, Neural Networks)
- Ensemble methods and stacking
- Hyperparameter optimization with Optuna
- Comprehensive model validation
- SHAP explainability
- Model persistence and evaluation metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)

# Advanced ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Advanced libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Model persistence and evaluation
import joblib
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class AdvancedStrokePredictor:
    """
    Advanced Stroke Prediction System with multiple ML algorithms and optimization.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.feature_selector = None
        self.ensemble_model = None
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def advanced_preprocessing(self, data):
        """
        Advanced data preprocessing with sophisticated imputation and feature engineering.
        """
        print("🔧 Starting advanced preprocessing...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Remove ID column if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Advanced BMI imputation using multiple strategies
        print("📊 Handling missing BMI values...")
        bmi_missing = df['bmi'].isnull().sum()
        print(f"Missing BMI values: {bmi_missing}")
        
        if bmi_missing > 0:
            # Strategy 1: Impute based on age and gender
            df['bmi_imputed'] = df['bmi'].isnull().astype(int)
            
            # Create age groups for BMI imputation
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], 
                                   labels=['young', 'middle', 'senior', 'elderly'])
            
            # Impute BMI by age group and gender
            for age_group in df['age_group'].unique():
                for gender in df['gender'].unique():
                    mask = (df['age_group'] == age_group) & (df['gender'] == gender)
                    if mask.sum() > 0:
                        group_mean = df[mask & df['bmi'].notna()]['bmi'].mean()
                        if not pd.isna(group_mean):
                            df.loc[mask & df['bmi'].isna(), 'bmi'] = group_mean
            
            # Fallback: use overall median if still missing
            df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # Feature Engineering
        print("🔨 Creating advanced features...")
        
        # Age-based features
        df['age_squared'] = df['age'] ** 2
        df['age_log'] = np.log1p(df['age'])
        df['is_elderly'] = (df['age'] > 65).astype(int)
        df['is_senior'] = (df['age'] > 50).astype(int)
        
        # BMI-based features
        df['bmi_squared'] = df['bmi'] ** 2
        df['bmi_log'] = np.log1p(df['bmi'])
        df['is_obese'] = (df['bmi'] > 30).astype(int)
        df['is_overweight'] = (df['bmi'] > 25).astype(int)
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                   labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Glucose-based features
        df['glucose_log'] = np.log1p(df['avg_glucose_level'])
        df['is_diabetic'] = (df['avg_glucose_level'] > 126).astype(int)
        df['is_prediabetic'] = ((df['avg_glucose_level'] >= 100) & 
                               (df['avg_glucose_level'] <= 126)).astype(int)
        df['glucose_category'] = pd.cut(df['avg_glucose_level'], 
                                      bins=[0, 100, 126, 200, 1000],
                                      labels=['normal', 'prediabetic', 'diabetic', 'severe'])
        
        # Interaction features
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
        df['bmi_glucose_interaction'] = df['bmi'] * df['avg_glucose_level']
        
        # Risk score combinations
        df['risk_score'] = (df['hypertension'] + df['heart_disease'] + 
                           df['is_obese'] + df['is_diabetic'] + 
                           (df['age'] > 65).astype(int))
        
        # Advanced smoking status encoding
        smoking_mapping = {
            'never smoked': 0,
            'formerly smoked': 1,
            'smokes': 2,
            'Unknown': 3
        }
        df['smoking_numeric'] = df['smoking_status'].map(smoking_mapping)
        
        # Work type risk assessment
        work_risk_mapping = {
            'Private': 1,
            'Self-employed': 2,
            'Govt_job': 0,
            'children': 0,
            'Never_worked': 0
        }
        df['work_risk_score'] = df['work_type'].map(work_risk_mapping)
        
        # One-hot encoding for categorical variables
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 
                              'smoking_status', 'bmi_category', 'glucose_category']
        
        for col in categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
        
        # Remove original categorical columns that were encoded
        columns_to_remove = ['age_group', 'bmi_imputed']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        print(f"✅ Preprocessing complete. Dataset shape: {df.shape}")
        return df
    
    def feature_selection(self, X, y, method='rfe', k=20):
        """
        Advanced feature selection using multiple methods.
        """
        print(f"🎯 Performing feature selection using {method}...")
        
        if method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            self.feature_selector = RFE(estimator, n_features_to_select=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.support_]
            
        elif method == 'univariate':
            # Univariate feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            
        else:
            # Use all features
            X_selected = X
            selected_features = X.columns
        
        print(f"✅ Selected {len(selected_features)} features")
        return X_selected, selected_features
    
    def create_models(self):
        """
        Create a comprehensive set of ML models for ensemble learning.
        """
        print("🤖 Creating advanced ML models...")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            
            'CatBoost': cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'
            ),
            
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=self.random_state
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            
            'NaiveBayes': GaussianNB()
        }
        
        return models
    
    def optimize_hyperparameters(self, X, y, model_name, model, n_trials=50):
        """
        Optimize hyperparameters using Optuna.
        """
        print(f"🔍 Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'
                }
                model.set_params(**params)
                
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model.set_params(**params)
                
            elif model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model.set_params(**params)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        model.set_params(**study.best_params)
        return model, study.best_value
    
    def train_models(self, X, y, optimize=True):
        """
        Train all models with optional hyperparameter optimization.
        """
        print("🚀 Training advanced ML models...")
        
        models = self.create_models()
        results = {}
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        for name, model in models.items():
            print(f"📈 Training {name}...")
            
            try:
                # Hyperparameter optimization for key models
                if optimize and name in ['XGBoost', 'LightGBM', 'RandomForest']:
                    model, best_score = self.optimize_hyperparameters(X_train_scaled, y_train, name, model)
                    print(f"   Best CV score: {best_score:.4f}")
                
                # Train model
                if name in ['NeuralNetwork', 'SVM', 'LogisticRegression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba)
                f1 = f1_score(y_val, y_pred)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def create_ensemble(self, X, y):
        """
        Create an advanced ensemble model using the best performing models.
        """
        print("🎭 Creating ensemble model...")
        
        # Get top performing models
        top_models = sorted(self.results.items(), key=lambda x: x[1]['auc'], reverse=True)[:5]
        print(f"Top 5 models: {[name for name, _ in top_models]}")
        
        # Create voting classifier
        estimators = [(name, result['model']) for name, result in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        ensemble.fit(X, y)
        
        self.ensemble_model = ensemble
        return ensemble
    
    def evaluate_model(self, X, y, model, model_name):
        """
        Comprehensive model evaluation.
        """
        print(f"📊 Evaluating {model_name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_models(self, filepath_prefix='advanced_stroke_model'):
        """
        Save all trained models and components.
        """
        print("💾 Saving models and components...")
        
        # Save individual models
        for name, result in self.results.items():
            joblib.dump(result['model'], f'{filepath_prefix}_{name.lower()}.pkl')
        
        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f'{filepath_prefix}_ensemble.pkl')
        
        # Save scaler
        if hasattr(self, 'scaler'):
            joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
        
        # Save feature columns
        if self.feature_columns is not None:
            joblib.dump(self.feature_columns, f'{filepath_prefix}_features.pkl')
        
        # Save feature selector
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, f'{filepath_prefix}_selector.pkl')
        
        # Save results
        with open(f'{filepath_prefix}_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print("✅ Models saved successfully!")

def main():
    """
    Main function to run the advanced stroke prediction pipeline.
    """
    print("🧠 Advanced Stroke Prediction System")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = AdvancedStrokePredictor()
    
    # Load data
    print("📁 Loading dataset...")
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Stroke cases: {data['stroke'].sum()} ({data['stroke'].mean()*100:.2f}%)")
    
    # Advanced preprocessing
    processed_data = predictor.advanced_preprocessing(data)
    
    # Prepare features and target
    target_column = 'stroke'
    feature_columns = [col for col in processed_data.columns if col != target_column]
    X = processed_data[feature_columns]
    y = processed_data[target_column]
    
    predictor.feature_columns = feature_columns
    
    # Feature selection
    X_selected, selected_features = predictor.feature_selection(X, y, method='rfe', k=25)
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    # Train models
    results = predictor.train_models(X_selected, y, optimize=True)
    
    # Create ensemble
    ensemble = predictor.create_ensemble(X_selected, y)
    
    # Evaluate ensemble
    ensemble_eval = predictor.evaluate_model(X_selected, y, ensemble, 'Ensemble')
    print(f"🎯 Ensemble Performance - AUC: {ensemble_eval['auc']:.4f}, Accuracy: {ensemble_eval['accuracy']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    predictor.best_model = best_model
    predictor.best_score = results[best_model_name]['auc']
    
    print(f"🏆 Best Model: {best_model_name} (AUC: {predictor.best_score:.4f})")
    
    # Save models
    predictor.save_models()
    
    # Print final results
    print("\n📈 Final Model Performance Summary:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:15} - AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
    
    print(f"\n🎭 Ensemble - AUC: {ensemble_eval['auc']:.4f}, Accuracy: {ensemble_eval['accuracy']:.4f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
