"""
Simple but Advanced XGBoost Stroke Predictor
===========================================

This creates a high-performance XGBoost model with:
- Advanced preprocessing
- SMOTE for imbalanced data
- Optimized XGBoost parameters
- Proper evaluation
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
from pathlib import Path

# Core libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# XGBoost
import xgboost as xgb

# SMOTE for imbalanced data
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAdvancedXGBoost:
    """
    Simple but advanced XGBoost stroke predictor.
    """

    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', test_size=0.2, random_state=42):
        """Initialize the predictor."""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.performance_metrics = {}

        # Create output directory
        self.output_dir = Path('advanced_models')
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Simple Advanced XGBoost Predictor initialized")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")

        # Handle missing values
        df = self.handle_missing_values(df)

        # Feature engineering
        df = self.feature_engineering(df)

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

        return X_train, X_test, y_train, y_test

    def handle_missing_values(self, df):
        """Handle missing values."""
        logger.info("Handling missing values...")

        # Smart BMI imputation
        bmi_mask = df['bmi'].isna()
        if bmi_mask.sum() > 0:
            df.loc[bmi_mask, 'bmi'] = df.groupby(['age', 'gender'])['bmi'].transform(
                lambda x: x.median() if not pd.isna(x.median()) else df['bmi'].median()
            )
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())

        # Smoking status imputation
        smoking_mask = df['smoking_status'].isna() | (df['smoking_status'] == 'Unknown')
        if smoking_mask.sum() > 0:
            df.loc[smoking_mask & (df['age'] < 30), 'smoking_status'] = 'never smoked'
            df.loc[smoking_mask & (df['age'] >= 30), 'smoking_status'] = 'formerly smoked'

        return df

    def feature_engineering(self, df):
        """Create advanced features."""
        logger.info("Creating advanced features...")

        # Derived features
        df['age_squared'] = df['age'] ** 2
        df['bmi_squared'] = df['bmi'] ** 2
        df['glucose_log'] = np.log1p(df['avg_glucose_level'])

        # Risk indicators
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        df['is_diabetic'] = (df['avg_glucose_level'] >= 126).astype(int)

        # Interaction features
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']

        # Risk scores
        df['cardiovascular_risk'] = (df['hypertension'] + df['heart_disease'] +
                                   df['is_diabetic'] + df['is_obese'])

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

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train advanced XGBoost model."""
        logger.info("Training advanced XGBoost model...")

        # Apply SMOTE for imbalanced learning
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_train_resampled.shape}, Stroke cases: {y_train_resampled.sum()}")

        # Optimized XGBoost parameters
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
            'eval_metric': 'logloss',
            'verbosity': 1
        }

        # Train XGBoost
        self.model = xgb.XGBClassifier(**xgb_params)
        self.model.fit(X_train_resampled, y_train_resampled)

        logger.info("XGBoost model trained successfully")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

        self.performance_metrics = metrics

        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def save_model(self):
        """Save the trained model."""
        logger.info("Saving model...")

        # Save XGBoost model
        joblib.dump(self.model, self.output_dir / 'xgboost_model.pkl')

        # Save performance metrics
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            import json
            json.dump(self.performance_metrics, f, indent=2)

        # Save model info
        model_info = {
            'model_type': 'XGBoost',
            'training_date': datetime.now().isoformat(),
            'performance': self.performance_metrics,
            'description': 'Advanced XGBoost stroke prediction model with SMOTE'
        }

        with open(self.output_dir / 'model_info.json', 'w') as f:
            import json
            json.dump(model_info, f, indent=2)

        logger.info(f"Model saved to {self.output_dir}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete XGBoost training pipeline...")

        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()

        # Train model
        model = self.train_model(X_train, X_test, y_train, y_test)

        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)

        # Save model
        self.save_model()

        logger.info("Complete XGBoost training pipeline finished!")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return self.model, metrics


def main():
    """Main function."""
    logger.info("Starting Simple Advanced XGBoost Stroke Predictor...")

    predictor = SimpleAdvancedXGBoost()

    try:
        model, metrics = predictor.run_complete_pipeline()

        logger.info("SUCCESS: Advanced XGBoost model created!")
        logger.info(f"Performance: {metrics}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == '__main__':
    main()
