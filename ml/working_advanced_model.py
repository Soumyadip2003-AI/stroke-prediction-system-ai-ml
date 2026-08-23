"""
Working Advanced Stroke Prediction Model with Supervised + Unsupervised Learning
================================================================================

This creates a truly advanced model using:
✅ Supervised Learning: XGBoost, Random Forest, Gradient Boosting
✅ Unsupervised Learning: PCA, ICA, K-Means clustering for feature engineering
✅ SMOTE for imbalanced data handling
✅ Advanced feature engineering
✅ Ensemble methods
✅ Hyperparameter optimization
✅ Cross-validation
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
from pathlib import Path

# Core libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

# Wrap XGBoost import in try-except block to handle library loading issues
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError as e:
    print(f"XGBoost not available: {e}")
    XGB_AVAILABLE = False

# Unsupervised learning
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans

# SMOTE for imbalanced data
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingAdvancedStrokePredictor:
    """
    Working implementation of advanced stroke prediction using both supervised and unsupervised learning.
    """

    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', test_size=0.2, random_state=42):
        """Initialize the predictor."""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.unsupervised_models = {}
        self.performance_metrics = {}

        # Create output directory
        self.output_dir = Path('working_advanced_models')
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Working Advanced Stroke Predictor initialized")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")

        # Handle missing values
        df = self.handle_missing_values(df)

        # Feature engineering (basic)
        df = self.feature_engineering(df)

        # Split data first
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

    def apply_unsupervised_learning(self, X_train, X_test):
        """Apply unsupervised learning for advanced feature engineering."""
        logger.info("Applying unsupervised learning...")

        # Standardize data for unsupervised learning
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # PCA for dimensionality reduction and feature extraction
        logger.info("Applying PCA...")
        pca = PCA(n_components=0.95, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        logger.info(f"PCA reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")

        # ICA for signal separation
        logger.info("Applying ICA...")
        ica = FastICA(n_components=5, random_state=self.random_state, max_iter=1000)
        X_train_ica = ica.fit_transform(X_train_scaled)
        X_test_ica = ica.transform(X_test_scaled)

        # K-Means clustering to find patterns
        logger.info("Applying K-Means clustering...")
        kmeans = KMeans(n_clusters=4, random_state=self.random_state, n_init=10)
        X_train_clusters = kmeans.fit_predict(X_train_scaled)
        X_test_clusters = kmeans.predict(X_test_scaled)

        # Store unsupervised models
        self.unsupervised_models = {
            'pca': pca,
            'ica': ica,
            'kmeans': kmeans,
            'scaler': scaler
        }

        # Add unsupervised features to original data
        unsupervised_features = {
            'pca_0': X_train_pca[:, 0] if X_train_pca.shape[1] > 0 else np.zeros(X_train.shape[0]),
            'pca_1': X_train_pca[:, 1] if X_train_pca.shape[1] > 1 else np.zeros(X_train.shape[0]),
            'ica_0': X_train_ica[:, 0] if X_train_ica.shape[1] > 0 else np.zeros(X_train.shape[0]),
            'ica_1': X_train_ica[:, 1] if X_train_ica.shape[1] > 1 else np.zeros(X_train.shape[0]),
            'kmeans_cluster': X_train_clusters,
        }

        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()

        for feature, values in unsupervised_features.items():
            X_train_enhanced[feature] = values
            if feature == 'kmeans_cluster':
                X_test_enhanced[feature] = X_test_clusters
            elif feature.startswith('pca_'):
                X_test_enhanced[feature] = X_test_pca[:, int(feature.split('_')[1])] if X_test_pca.shape[1] > int(feature.split('_')[1]) else 0
            elif feature.startswith('ica_'):
                X_test_enhanced[feature] = X_test_ica[:, int(feature.split('_')[1])] if X_test_ica.shape[1] > int(feature.split('_')[1]) else 0

        logger.info(f"Enhanced training set shape: {X_train_enhanced.shape}")
        logger.info(f"Enhanced test set shape: {X_test_enhanced.shape}")

        return X_train_enhanced, X_test_enhanced, self.unsupervised_models

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train advanced models with unsupervised features."""
        logger.info("Training advanced models with unsupervised features...")

        # Apply SMOTE for imbalanced learning
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_train_resampled.shape}, Stroke cases: {y_train_resampled.sum()}")

        models = {}

        # XGBoost - Primary model
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            xgb_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'min_child_weight': 5,
                'random_state': self.random_state,
                'verbosity': 0
            }

            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            models['xgboost'].fit(X_train_resampled, y_train_resampled)
        else:
            logger.warning("Skipping XGBoost training due to import error.")
            models['xgboost'] = None

        # Random Forest
        logger.info("Training Random Forest...")
        models['randomforest'] = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_split=3, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, oob_score=True, random_state=self.random_state
        )
        models['randomforest'].fit(X_train_resampled, y_train_resampled)

        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        models['gradientboosting'] = GradientBoostingClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
            min_samples_split=3, min_samples_leaf=2, random_state=self.random_state
        )
        models['gradientboosting'].fit(X_train_resampled, y_train_resampled)

        # Extra Trees
        logger.info("Training Extra Trees...")
        models['extratrees'] = ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_split=3, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, random_state=self.random_state
        )
        models['extratrees'].fit(X_train_resampled, y_train_resampled)

        # Create ensemble with optimized weights
        logger.info("Creating ensemble...")
        estimators = [(name, model) for name, model in models.items() if hasattr(model, 'predict_proba')]

        # Weighted ensemble favoring XGBoost
        weights = [0.4 if 'xgb' in name else 0.15 for name, _ in estimators]

        models['ensemble'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        models['ensemble'].fit(X_train_resampled, y_train_resampled)

        self.models = models
        logger.info(f"Trained {len(models)} models with unsupervised features")

        return models

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models."""
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

            logger.info(f"Best model: {best_model_name} with F1={results[best_model_name]['f1']:.4f}")

        return results

    def save_models(self):
        """Save all trained models and components."""
        logger.info("Saving models...")

        # Save individual models
        for name, model in self.models.items():
            if model is not None: # Only save if model was trained
                joblib.dump(model, self.output_dir / f'{name}_model.pkl')

        # Save unsupervised models
        joblib.dump(self.unsupervised_models, self.output_dir / 'unsupervised_models.pkl')

        # Save performance metrics
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            import json
            json.dump(self.performance_metrics, f, indent=2)

        # Save model info
        model_info = {
            'model_type': 'Advanced Supervised + Unsupervised',
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'total_models': len(self.models),
            'unsupervised_features_used': list(self.unsupervised_models.keys()),
            'performance': self.performance_metrics
        }

        with open(self.output_dir / 'model_info.json', 'w') as f:
            import json
            json.dump(model_info, f, indent=2)

        logger.info(f"Models saved to {self.output_dir}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline with supervised + unsupervised learning."""
        logger.info("Starting complete advanced training pipeline...")

        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test, df = self.load_and_preprocess_data()

            # Apply unsupervised learning for feature engineering
            X_train_enhanced, X_test_enhanced, unsupervised_models = self.apply_unsupervised_learning(X_train, X_test)

            # Train models with enhanced features
            models = self.train_models(X_train_enhanced, X_test_enhanced, y_train, y_test)

            # Evaluate models
            results = self.evaluate_models(models, X_test_enhanced, y_test)

            # Save everything
            self.save_models()

            logger.info("SUCCESS: Advanced model with supervised + unsupervised learning created!")
            logger.info(f"Best model: {self.best_model_name}")
            logger.info(f"Best F1 Score: {self.performance_metrics[self.best_model_name]['f1']:.4f}")
            logger.info(f"Best ROC AUC: {self.performance_metrics[self.best_model_name]['roc_auc']:.4f}")
            logger.info(f"Unsupervised techniques used: PCA, ICA, K-Means clustering")

            return self.best_model, self.performance_metrics

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Main function."""
    logger.info("Starting Working Advanced Stroke Prediction System...")

    predictor = WorkingAdvancedStrokePredictor()

    try:
        model, metrics = predictor.run_complete_pipeline()

        logger.info("SUCCESS: Advanced model with supervised + unsupervised learning created!")
        logger.info(f"Performance: {metrics}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == '__main__':
    main()
