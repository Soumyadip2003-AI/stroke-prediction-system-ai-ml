#!/usr/bin/env python3
"""
Simplified Stroke Prediction Model Training
This script trains a basic but effective stroke prediction model without XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the stroke dataset"""
    print("Loading dataset...")
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_ever_married = LabelEncoder()
    le_work_type = LabelEncoder()
    le_residence_type = LabelEncoder()
    le_smoking_status = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['ever_married'] = le_ever_married.fit_transform(df['ever_married'])
    df['work_type'] = le_work_type.fit_transform(df['work_type'])
    df['Residence_type'] = le_residence_type.fit_transform(df['Residence_type'])
    df['smoking_status'] = le_smoking_status.fit_transform(df['smoking_status'])
    
    # Feature engineering
    df['age_squared'] = df['age'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['hypertension_heart_disease'] = df['hypertension'] + df['heart_disease']
    
    # Select features
    feature_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
        'smoking_status', 'age_squared', 'bmi_squared', 'age_bmi_interaction',
        'hypertension_heart_disease'
    ]
    
    X = df[feature_columns]
    y = df['stroke']
    
    print(f"Features: {feature_columns}")
    print(f"Target distribution: {y.value_counts()}")
    
    return X, y, feature_columns

def create_models():
    """Create and return a dictionary of models"""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    return models

def train_and_evaluate_models(X, y, models):
    """Train and evaluate all models"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'LogisticRegression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train if name not in ['SVM', 'LogisticRegression'] else X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results, X_test, y_test, scaler

def create_ensemble_model(results, X_train, y_train, scaler):
    """Create an ensemble model from the best performing models"""
    print("\nCreating ensemble model...")
    
    # Get the top 3 models by CV score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
    top_models = sorted_models[:3]
    
    print(f"Top 3 models for ensemble: {[name for name, _ in top_models]}")
    
    # Create voting classifier
    estimators = []
    for name, result in top_models:
        if name in ['SVM', 'LogisticRegression']:
            # For scaled models, we need to create a pipeline
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', result['model'])
            ])
            estimators.append((name, pipeline))
        else:
            estimators.append((name, result['model']))
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    
    return ensemble

def main():
    """Main training function"""
    print("=== Stroke Prediction Model Training ===")
    
    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data()
    
    # Create models
    models = create_models()
    
    # Train and evaluate models
    results, X_test, y_test, scaler = train_and_evaluate_models(X, y, models)
    
    # Create ensemble model
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ensemble = create_ensemble_model(results, X_train, y_train, scaler)
    
    # Evaluate ensemble
    ensemble_pred = ensemble.predict(X_test_split)
    ensemble_proba = ensemble.predict_proba(X_test_split)[:, 1]
    ensemble_accuracy = accuracy_score(y_test_split, ensemble_pred)
    
    print(f"\nEnsemble Model - Accuracy: {ensemble_accuracy:.4f}")
    
    # Save models and scaler
    print("\nSaving models...")
    joblib.dump(ensemble, 'stroke_prediction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    # Save individual models
    for name, result in results.items():
        joblib.dump(result['model'], f'{name.lower()}_model.pkl')
    
    print("Models saved successfully!")
    
    # Print final results
    print("\n=== Final Results ===")
    print("Individual Models:")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f} (CV: {result['cv_mean']:.4f})")
    
    print(f"Ensemble Model: {ensemble_accuracy:.4f}")
    
    return ensemble, scaler, feature_columns

if __name__ == "__main__":
    model, scaler, features = main()
