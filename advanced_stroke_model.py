"""
Advanced Stroke Prediction Model with XGBoost
=============================================

This script creates a high-performance stroke prediction model using:
- XGBoost with optimized hyperparameters
- SMOTE for imbalanced data handling
- Advanced feature engineering
- Cross-validation and evaluation
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import logging
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Installing XGBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("Installing imbalanced-learn...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'imbalanced-learn'])
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    IMBALANCED_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedStrokePredictor:
    """
    Advanced stroke prediction system using XGBoost and ensemble methods.
    """

    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', test_size=0.2, random_state=42):
        """Initialize the predictor."""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.performance_metrics = {}

        # Create output directory
        self.output_dir = Path('advanced_models')
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Advanced Stroke Predictor initialized")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")

        # Handle missing values
        df = self.handle_missing_values(df)

        # Advanced feature engineering
        df = self.advanced_feature_engineering(df)

        # Handle outliers
        df = self.handle_outliers(df)

        # Encode categorical variables
        df = self.encode_categorical_variables(df)

        # Split data
        X = df.drop('stroke', axis=1)
        y = df['stroke']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Stroke distribution in training set: {y_train.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, df

    def handle_missing_values(self, df):
        """Handle missing values intelligently."""
        logger.info("Handling missing values...")

        # Smart BMI imputation
        bmi_mask = df['bmi'].isna()
        if bmi_mask.sum() > 0:
            df.loc[bmi_mask, 'bmi'] = df.groupby(['age', 'gender', 'hypertension'])['bmi'].transform(
                lambda x: x.median() if not pd.isna(x.median()) else df['bmi'].median()
            )
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())

        # Smoking status imputation
        smoking_mask = df['smoking_status'].isna() | (df['smoking_status'] == 'Unknown')
        if smoking_mask.sum() > 0:
            df.loc[smoking_mask & (df['age'] < 30), 'smoking_status'] = 'never smoked'
            df.loc[smoking_mask & (df['age'] >= 30), 'smoking_status'] = 'formerly smoked'

        return df

    def advanced_feature_engineering(self, df):
        """Create advanced features."""
        logger.info("Performing advanced feature engineering...")

        # Basic derived features
        df['age_squared'] = df['age'] ** 2
        df['age_log'] = np.log1p(df['age'])
        df['age_sqrt'] = np.sqrt(df['age'])

        df['bmi_squared'] = df['bmi'] ** 2
        df['bmi_log'] = np.log1p(df['bmi'])

        df['glucose_log'] = np.log1p(df['avg_glucose_level'])
        df['glucose_squared'] = df['avg_glucose_level'] ** 2

        # Age categories
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        df['is_senior'] = (df['age'] >= 50).astype(int)

        # BMI categories
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        df['is_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)

        # Glucose categories
        df['is_diabetic'] = (df['avg_glucose_level'] >= 126).astype(int)
        df['is_prediabetic'] = ((df['avg_glucose_level'] >= 100) & (df['avg_glucose_level'] < 126)).astype(int)

        # Interaction features
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
        df['bmi_glucose_interaction'] = df['bmi'] * df['avg_glucose_level']

        # Risk scores
        df['cardiovascular_risk'] = (df['hypertension'] + df['heart_disease'] +
                                   df['is_diabetic'] + df['is_obese'])
        df['overall_risk_score'] = df['cardiovascular_risk'] + (df['age'] / 20)

        return df

    def handle_outliers(self, df):
        """Handle outliers using IQR method."""
        logger.info("Handling outliers...")

        numeric_cols = ['age', 'bmi', 'avg_glucose_level']
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = np.clip(df[col], lower_bound, upper_bound)

        return df

    def encode_categorical_variables(self, df):
        """Encode categorical variables."""
        logger.info("Encoding categorical variables...")

        # Label encoding for ordinal categories
        smoking_mapping = {
            'never smoked': 0,
            'formerly smoked': 1,
            'smokes': 2,
            'Unknown': 3
        }
        df['smoking_encoded'] = df['smoking_status'].map(smoking_mapping)

        # One-hot encoding for nominal categories
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']

        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        return df

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train advanced models."""
        logger.info("Training advanced models...")

        # Apply SMOTE for imbalanced learning
        if IMBALANCED_AVAILABLE:
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {X_train_resampled.shape}, Stroke cases: {y_train_resampled.sum()}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        models = {}

        # XGBoost - Primary model
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost...")
            xgb_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'eval_metric': 'logloss'
            }

            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            models['xgboost'].fit(X_train_resampled, y_train_resampled)

        # Random Forest
        logger.info("Training Random Forest...")
        models['randomforest'] = RandomForestClassifier(
            n_estimators=1000, max_depth=12, min_samples_split=3, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, oob_score=True, random_state=self.random_state
        )
        models['randomforest'].fit(X_train_resampled, y_train_resampled)

        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        models['gradientboosting'] = GradientBoostingClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.05, subsample=0.8,
            min_samples_split=3, min_samples_leaf=2, random_state=self.random_state
        )
        models['gradientboosting'].fit(X_train_resampled, y_train_resampled)

        # Extra Trees
        logger.info("Training Extra Trees...")
        models['extratrees'] = ExtraTreesClassifier(
            n_estimators=1000, max_depth=12, min_samples_split=3, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, random_state=self.random_state
        )
        models['extratrees'].fit(X_train_resampled, y_train_resampled)

        # Balanced Random Forest
        if IMBALANCED_AVAILABLE:
            logger.info("Training Balanced Random Forest...")
            models['balanced_rf'] = BalancedRandomForestClassifier(
                n_estimators=1000, max_depth=12, min_samples_split=3, min_samples_leaf=2,
                max_features='sqrt', random_state=self.random_state
            )
            models['balanced_rf'].fit(X_train_resampled, y_train_resampled)

        # Create ensemble
        logger.info("Creating ensemble...")
        estimators = [(name, model) for name, model in models.items() if hasattr(model, 'predict_proba')]

        # Soft voting ensemble with optimized weights
        weights = []
        for name, _ in estimators:
            if 'xgb' in name:
                weights.append(0.4)  # XGBoost gets highest weight
            elif 'balanced' in name:
                weights.append(0.2)  # Balanced RF gets good weight
            else:
                weights.append(0.1)  # Others get lower weight

        models['ensemble'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        models['ensemble'].fit(X_train_resampled, y_train_resampled)

        self.models = models
        logger.info(f"Trained {len(models)} models")

        return models

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models and select the best one."""
        logger.info("Evaluating models...")

        results = {}

        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    results[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba)
                    }

                    logger.info(f"{name}: F1={results[name]['f1']:.4f}, AUC={results[name]['roc_auc']:.4f}")
                else:
                    logger.warning(f"{name} does not support probability prediction")

            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")

        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
            self.best_model = models[best_model_name]
            self.best_model_name = best_model_name
            self.performance_metrics = results

            logger.info(f"Best model: {best_model_name} with F1={results[best_model_name]['f1']".4f"}")

        return results

    def save_models(self):
        """Save all trained models."""
        logger.info("Saving models...")

        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, self.output_dir / f'{name}_model.pkl')

        # Save performance metrics
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            import json
            json.dump(self.performance_metrics, f, indent=2)

        # Save model info
        model_info = {
            'best_model': self.best_model_name,
            'total_models': len(self.models),
            'training_date': datetime.now().isoformat(),
            'description': 'Advanced XGBoost-based stroke prediction model'
        }

        with open(self.output_dir / 'model_info.json', 'w') as f:
            import json
            json.dump(model_info, f, indent=2)

        logger.info(f"Models saved to {self.output_dir}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete advanced training pipeline...")

        # Load and preprocess data
        X_train, X_test, y_train, y_test, df = self.load_and_preprocess_data()

        # Train models
        models = self.train_models(X_train, X_test, y_train, y_test)

        # Evaluate models
        results = self.evaluate_models(models, X_test, y_test)

        # Save everything
        self.save_models()

        logger.info("Complete advanced training pipeline finished!")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Best F1 Score: {self.performance_metrics[self.best_model_name]['f1']".4f"}")

        return self.best_model, self.performance_metrics


def main():
    """Main function."""
    logger.info("Starting Advanced Stroke Prediction System...")

    predictor = AdvancedStrokePredictor()

    try:
        best_model, metrics = predictor.run_complete_pipeline()

        logger.info("SUCCESS: Advanced model created!")
        logger.info(f"Performance: {metrics}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == '__main__':
    main()
