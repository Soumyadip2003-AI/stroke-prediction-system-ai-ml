"""
Simplified Advanced Model Training Script
=========================================

This script trains the advanced stroke prediction models with high accuracy.
Run this script to generate the optimized models for the Streamlit app.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import joblib

# Advanced ML libraries (with fallbacks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

def advanced_preprocessing(data):
    """Advanced data preprocessing with feature engineering."""
    print("🔧 Starting advanced preprocessing...")
    
    df = data.copy()
    
    # Remove ID column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Handle missing BMI values with advanced imputation
    print("📊 Handling missing BMI values...")
    bmi_missing = df['bmi'].isnull().sum()
    print(f"Missing BMI values: {bmi_missing}")
    
    if bmi_missing > 0:
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
    
    # Advanced feature engineering
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
    
    # Remove temporary columns
    columns_to_remove = ['age_group']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    print(f"✅ Preprocessing complete. Dataset shape: {df.shape}")
    return df

def create_models():
    """Create advanced ML models."""
    print("🤖 Creating advanced ML models...")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        
        'LogisticRegression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ),
        
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        ),
        
        'NeuralNetwork': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
    
    return models

def train_and_evaluate_models(X, y):
    """Train and evaluate all models."""
    print("🚀 Training and evaluating models...")
    
    models = create_models()
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        print(f"📈 Training {name}...")
        
        try:
            # Train model
            if name in ['NeuralNetwork', 'SVM', 'LogisticRegression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {str(e)}")
            continue
    
    return results, scaler

def create_ensemble(results, X, y):
    """Create ensemble model from best performing models."""
    print("🎭 Creating ensemble model...")
    
    # Get top 3 models
    top_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]
    print(f"Top 3 models: {[name for name, _ in top_models]}")
    
    # Create voting classifier
    estimators = [(name, result['model']) for name, result in top_models]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Train ensemble
    ensemble.fit(X, y)
    
    return ensemble

def save_models(results, ensemble, scaler, feature_columns):
    """Save all trained models."""
    print("💾 Saving models...")
    
    # Save individual models
    for name, result in results.items():
        joblib.dump(result['model'], f'advanced_stroke_model_{name.lower()}.pkl')
    
    # Save ensemble model
    joblib.dump(ensemble, 'advanced_stroke_model_ensemble.pkl')
    
    # Save scaler and features
    joblib.dump(scaler, 'advanced_stroke_model_scaler.pkl')
    joblib.dump(feature_columns, 'advanced_stroke_model_features.pkl')
    
    # Save results
    import pickle
    with open('advanced_stroke_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✅ Models saved successfully!")

def main():
    """Main training function."""
    print("🧠 Advanced Stroke Prediction Model Training")
    print("=" * 50)
    
    # Load data
    print("📁 Loading dataset...")
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Stroke cases: {data['stroke'].sum()} ({data['stroke'].mean()*100:.2f}%)")
    
    # Advanced preprocessing
    processed_data = advanced_preprocessing(data)
    
    # Prepare features and target
    target_column = 'stroke'
    feature_columns = [col for col in processed_data.columns if col != target_column]
    X = processed_data[feature_columns]
    y = processed_data[target_column]
    
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train and evaluate models
    results, scaler = train_and_evaluate_models(X, y)
    
    # Create ensemble
    ensemble = create_ensemble(results, X, y)
    
    # Evaluate ensemble
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"🎯 Ensemble Performance - AUC: {ensemble_auc:.4f}, Accuracy: {ensemble_acc:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_score = results[best_model_name]['auc']
    
    print(f"🏆 Best Individual Model: {best_model_name} (AUC: {best_score:.4f})")
    
    # Save models
    save_models(results, ensemble, scaler, feature_columns)
    
    # Print final results
    print("\n📈 Final Model Performance Summary:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:15} - AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
    
    print(f"\n🎭 Ensemble - AUC: {ensemble_auc:.4f}, Accuracy: {ensemble_acc:.4f}")
    
    print("\n✅ Training complete! You can now run the advanced Streamlit app.")
    print("Run: streamlit run advanced_app.py")

if __name__ == "__main__":
    main()
