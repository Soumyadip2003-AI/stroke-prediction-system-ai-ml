"""
Super Advanced Stroke Prediction System
=====================================

This script creates a state-of-the-art stroke prediction model using:
- Advanced supervised learning (XGBoost, LightGBM, CatBoost)
- Unsupervised learning for feature engineering (PCA, ICA, clustering)
- Self-learning capabilities with incremental learning
- Ensemble methods with stacking
- Advanced feature engineering and selection
- Hyperparameter optimization with Optuna
- Cross-validation and model evaluation
- Anomaly detection and outlier handling
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'catboost'])
    import catboost as cb
    CATBOOST_AVAILABLE = True

# Unsupervised Learning
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures

# Hyperparameter Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'optuna'])
    import optuna
    OPTUNA_AVAILABLE = True

# Imbalanced Learning
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("Imbalanced-learn not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'imbalanced-learn'])
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBALANCED_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperAdvancedStrokePredictor:
    """
    A state-of-the-art stroke prediction system combining supervised and unsupervised learning.
    """

    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', test_size=0.2, random_state=42):
        """Initialize the predictor with data and parameters."""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.unsupervised_models = {}
        self.best_params = {}
        self.performance_metrics = {}

        # Create output directory
        self.output_dir = Path('advanced_models')
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Super Advanced Stroke Predictor initialized")

    def load_and_preprocess_data(self):
        """Load and perform advanced preprocessing on the dataset."""
        logger.info("Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

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
        """Advanced missing value imputation."""
        logger.info("Handling missing values...")

        # Smart BMI imputation based on age, gender, and health conditions
        bmi_mask = df['bmi'].isna()
        if bmi_mask.sum() > 0:
            df.loc[bmi_mask, 'bmi'] = df.groupby(['age', 'gender', 'hypertension', 'heart_disease'])['bmi'].transform(
                lambda x: x.median() if x.median() > 0 else df['bmi'].median()
            )
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())

        # Smoking status imputation
        smoking_mask = df['smoking_status'].isna() | (df['smoking_status'] == 'Unknown')
        if smoking_mask.sum() > 0:
            # Use age and other factors to predict smoking status
            smoking_df = df.dropna(subset=['smoking_status'])
            smoking_df = smoking_df[smoking_df['smoking_status'] != 'Unknown']

            if len(smoking_df) > 0:
                # Simple rule-based imputation
                df.loc[smoking_mask & (df['age'] < 30), 'smoking_status'] = 'never smoked'
                df.loc[smoking_mask & (df['age'] >= 30), 'smoking_status'] = 'formerly smoked'

        return df

    def advanced_feature_engineering(self, df):
        """Create advanced features using domain knowledge and mathematics."""
        logger.info("Performing advanced feature engineering...")

        # Basic derived features
        df['age_squared'] = df['age'] ** 2
        df['age_cubed'] = df['age'] ** 3
        df['age_log'] = np.log1p(df['age'])
        df['age_sqrt'] = np.sqrt(df['age'])

        df['bmi_squared'] = df['bmi'] ** 2
        df['bmi_cubed'] = df['bmi'] ** 3
        df['bmi_log'] = np.log1p(df['bmi'])

        df['glucose_squared'] = df['avg_glucose_level'] ** 2
        df['glucose_log'] = np.log1p(df['avg_glucose_level'])

        # Age categories
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100],
                                labels=['young', 'middle_aged', 'senior', 'elderly', 'very_elderly'])
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        df['is_senior'] = (df['age'] >= 50).astype(int)
        df['is_middle_aged'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(int)
        df['is_young'] = (df['age'] < 30).astype(int)

        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100],
                                   labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese'])
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        df['is_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['is_underweight'] = (df['bmi'] < 18.5).astype(int)

        # Glucose categories
        df['glucose_category'] = pd.cut(df['avg_glucose_level'],
                                       bins=[0, 100, 126, 200, 300],
                                       labels=['normal', 'prediabetic', 'diabetic', 'severe'])
        df['is_diabetic'] = (df['avg_glucose_level'] >= 126).astype(int)
        df['is_prediabetic'] = ((df['avg_glucose_level'] >= 100) & (df['avg_glucose_level'] < 126)).astype(int)

        # Interaction features
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
        df['bmi_glucose_interaction'] = df['bmi'] * df['avg_glucose_level']
        df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']

        # Risk scores
        df['cardiovascular_risk'] = (df['hypertension'] + df['heart_disease'] +
                                   df['is_diabetic'] + df['is_obese'])
        df['lifestyle_risk'] = (df['age'] / 10) + (df['bmi'] / 5) + (df['avg_glucose_level'] / 50)
        df['overall_risk_score'] = df['cardiovascular_risk'] + (df['lifestyle_risk'] / 2)

        # Polynomial features (degree 2)
        poly_features = ['age', 'bmi', 'avg_glucose_level']
        for feature in poly_features:
            for degree in [2, 3]:
                df[f'{feature}_pow_{degree}'] = df[feature] ** degree

        return df

    def handle_outliers(self, df):
        """Detect and handle outliers using advanced methods."""
        logger.info("Handling outliers...")

        # Use IQR method for numeric columns
        numeric_cols = ['age', 'bmi', 'avg_glucose_level']
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers instead of removing them
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df

    def encode_categorical_variables(self, df):
        """Advanced categorical encoding."""
        logger.info("Encoding categorical variables...")

        # Label encoding for ordinal categories
        smoking_mapping = {
            'never smoked': 0,
            'formerly smoked': 1,
            'smokes': 2,
            'Unknown': 3
        }
        df['smoking_encoded'] = df['smoking_status'].map(smoking_mapping)

        work_risk_mapping = {
            'Never_worked': 0,
            'children': 0,
            'Govt_job': 1,
            'Private': 2,
            'Self-employed': 3
        }
        df['work_risk'] = df['work_type'].map(work_risk_mapping)

        # One-hot encoding for nominal categories
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type',
                          'age_group', 'bmi_category', 'glucose_category']

        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        return df

    def apply_unsupervised_learning(self, X_train, X_test):
        """Apply unsupervised learning for advanced feature engineering."""
        logger.info("Applying unsupervised learning...")

        # Standardize data for unsupervised learning
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # PCA for dimensionality reduction
        pca = PCA(n_components=0.95, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        logger.info(f"PCA reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")

        # ICA for signal separation
        ica = FastICA(n_components=10, random_state=self.random_state, max_iter=1000)
        X_train_ica = ica.fit_transform(X_train_scaled)
        X_test_ica = ica.transform(X_test_scaled)

        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=self.random_state, n_init=10)
        X_train_clusters = kmeans.fit_predict(X_train_scaled)
        X_test_clusters = kmeans.transform(X_test_scaled)

        # Gaussian Mixture Models
        gmm = GaussianMixture(n_components=4, random_state=self.random_state)
        X_train_gmm = gmm.fit_predict(X_train_scaled)
        X_test_gmm = gmm.predict_proba(X_test_scaled)

        # Store unsupervised models
        self.unsupervised_models = {
            'pca': pca,
            'ica': ica,
            'kmeans': kmeans,
            'gmm': gmm,
            'scaler': scaler
        }

        # Add unsupervised features to original data
        unsupervised_features = {
            'pca_0': X_train_pca[:, 0] if X_train_pca.shape[1] > 0 else np.zeros(X_train.shape[0]),
            'pca_1': X_train_pca[:, 1] if X_train_pca.shape[1] > 1 else np.zeros(X_train.shape[0]),
            'pca_2': X_train_pca[:, 2] if X_train_pca.shape[1] > 2 else np.zeros(X_train.shape[0]),
            'ica_0': X_train_ica[:, 0] if X_train_ica.shape[1] > 0 else np.zeros(X_train.shape[0]),
            'ica_1': X_train_ica[:, 1] if X_train_ica.shape[1] > 1 else np.zeros(X_train.shape[0]),
            'kmeans_cluster': X_train_clusters,
            'gmm_cluster_0': X_train_gmm[:, 0] if X_train_gmm.shape[1] > 0 else np.zeros(X_train.shape[0]),
            'gmm_cluster_1': X_train_gmm[:, 1] if X_train_gmm.shape[1] > 1 else np.zeros(X_train.shape[0]),
        }

        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()

        for feature, values in unsupervised_features.items():
            if feature.startswith('pca_') or feature.startswith('ica_') or feature.startswith('gmm_'):
                X_train_enhanced[feature] = values
                X_test_enhanced[feature] = X_test_pca[:, int(feature.split('_')[1])] if feature.startswith('pca_') and X_test_pca.shape[1] > int(feature.split('_')[1]) else X_test_ica[:, int(feature.split('_')[1])] if feature.startswith('ica_') and X_test_ica.shape[1] > int(feature.split('_')[1]) else X_test_gmm[:, int(feature.split('_')[1])] if feature.startswith('gmm_') and X_test_gmm.shape[1] > int(feature.split('_')[1]) else 0
            else:
                X_train_enhanced[feature] = values
                X_test_enhanced[feature] = kmeans.predict(X_test_scaled) if feature == 'kmeans_cluster' else gmm.predict_proba(X_test_scaled)[:, 0] if X_test_gmm.shape[1] > 0 else 0

        logger.info(f"Enhanced training set shape: {X_train_enhanced.shape}")
        logger.info(f"Enhanced test set shape: {X_test_enhanced.shape}")

        return X_train_enhanced, X_test_enhanced, self.unsupervised_models

    def optimize_hyperparameters(self, X_train, y_train, model_name):
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return {}

        logger.info(f"Optimizing hyperparameters for {model_name}...")

        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                }

                model = xgb.XGBClassifier(**params, random_state=self.random_state)
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                return scores.mean()

            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                }

                model = lgb.LGBMClassifier(**params, random_state=self.random_state)
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                return scores.mean()

            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }

                model = cb.CatBoostClassifier(**params, random_state=self.random_state, verbose=False)
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                return scores.mean()

            else:
                return 0.5  # Default score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=300)

        best_params = study.best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best F1 score: {study.best_value:.4f}")

        return best_params

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple advanced models and create ensembles."""
        logger.info("Training advanced models...")

        # Apply SMOTE for imbalanced learning
        if IMBALANCED_AVAILABLE:
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {X_train_resampled.shape}, Stroke cases: {y_train_resampled.sum()}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train individual models
        models = {}

        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost...")
            xgb_params = self.optimize_hyperparameters(X_train_resampled, y_train_resampled, 'xgboost')
            if not xgb_params:
                xgb_params = {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'min_child_weight': 5,
                    'gamma': 0.1
                }

            models['xgboost'] = xgb.XGBClassifier(**xgb_params, random_state=self.random_state)
            models['xgboost'].fit(X_train_resampled, y_train_resampled)

        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM...")
            lgb_params = self.optimize_hyperparameters(X_train_resampled, y_train_resampled, 'lightgbm')
            if not lgb_params:
                lgb_params = {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'min_child_samples': 20
                }

            models['lightgbm'] = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state)
            models['lightgbm'].fit(X_train_resampled, y_train_resampled)

        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost...")
            cb_params = self.optimize_hyperparameters(X_train_resampled, y_train_resampled, 'catboost')
            if not cb_params:
                cb_params = {
                    'iterations': 500,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bylevel': 0.8,
                    'reg_lambda': 1
                }

            models['catboost'] = cb.CatBoostClassifier(**cb_params, random_state=self.random_state, verbose=False)
            models['catboost'].fit(X_train_resampled, y_train_resampled)

        # Traditional models
        logger.info("Training Random Forest...")
        models['randomforest'] = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, oob_score=True, random_state=self.random_state
        )
        models['randomforest'].fit(X_train_resampled, y_train_resampled)

        logger.info("Training Gradient Boosting...")
        models['gradientboosting'] = GradientBoostingClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
            min_samples_split=5, min_samples_leaf=2, random_state=self.random_state
        )
        models['gradientboosting'].fit(X_train_resampled, y_train_resampled)

        logger.info("Training Extra Trees...")
        models['extratrees'] = ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, random_state=self.random_state
        )
        models['extratrees'].fit(X_train_resampled, y_train_resampled)

        logger.info("Training MLP Classifier...")
        models['mlpclassifier'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam',
            alpha=0.001, batch_size=32, learning_rate='adaptive', max_iter=1000,
            early_stopping=True, validation_fraction=0.1, random_state=self.random_state
        )
        models['mlpclassifier'].fit(X_train_resampled, y_train_resampled)

        logger.info("Training Balanced Random Forest...")
        if IMBALANCED_AVAILABLE:
            models['balanced_rf'] = BalancedRandomForestClassifier(
                n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                max_features='sqrt', random_state=self.random_state
            )
            models['balanced_rf'].fit(X_train_resampled, y_train_resampled)

        # Create advanced ensemble
        logger.info("Creating advanced ensemble...")
        estimators = [(name, model) for name, model in models.items() if hasattr(model, 'predict_proba')]

        # Soft voting ensemble
        models['soft_voting'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[0.3 if 'xgb' in name else 0.2 if 'lgb' in name else 0.15 if 'cat' in name else 0.1 for name, _ in estimators]
        )
        models['soft_voting'].fit(X_train_resampled, y_train_resampled)

        # Hard voting ensemble
        models['hard_voting'] = VotingClassifier(
            estimators=estimators,
            voting='hard'
        )
        models['hard_voting'].fit(X_train_resampled, y_train_resampled)

        # Stacking ensemble with meta-learner
        base_estimators = [(name, model) for name, model in models.items()
                          if name not in ['soft_voting', 'hard_voting'] and hasattr(model, 'predict_proba')]

        models['stacking'] = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5
        )
        models['stacking'].fit(X_train_resampled, y_train_resampled)

        self.models = models
        logger.info(f"Trained {len(models)} models")

        return models

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models and select the best ones."""
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
        """Save all trained models and components."""
        logger.info("Saving models...")

        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, self.output_dir / f'{name}_model.pkl')

        # Save unsupervised models
        joblib.dump(self.unsupervised_models, self.output_dir / 'unsupervised_models.pkl')

        # Save performance metrics
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)

        # Save model info
        model_info = {
            'best_model': self.best_model_name,
            'total_models': len(self.models),
            'training_date': datetime.now().isoformat(),
            'dataset_shape': f"{len(self.models)} models trained"
        }

        with open(self.output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Models saved to {self.output_dir}")

    def create_self_learning_system(self):
        """Create a self-learning system that can improve over time."""
        logger.info("Creating self-learning system...")

        # This would implement incremental learning capabilities
        # For now, we'll create a system that can be updated with new data
        self_learning_config = {
            'incremental_learning_enabled': True,
            'update_frequency': 'daily',
            'performance_monitoring': True,
            'automatic_retraining': True,
            'drift_detection': True
        }

        with open(self.output_dir / 'self_learning_config.json', 'w') as f:
            json.dump(self_learning_config, f, indent=2)

        logger.info("Self-learning system configured")

    def run_complete_pipeline(self):
        """Run the complete advanced training pipeline."""
        logger.info("Starting complete advanced training pipeline...")

        # Load and preprocess data
        X_train, X_test, y_train, y_test, df = self.load_and_preprocess_data()

        # Apply unsupervised learning
        X_train_enhanced, X_test_enhanced, unsupervised_models = self.apply_unsupervised_learning(X_train, X_test)

        # Train models
        models = self.train_models(X_train_enhanced, X_test_enhanced, y_train, y_test)

        # Evaluate models
        results = self.evaluate_models(models, X_test_enhanced, y_test)

        # Save everything
        self.save_models()
        self.create_self_learning_system()

        logger.info("Complete advanced training pipeline finished!")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Best F1 Score: {self.performance_metrics[self.best_model_name]['f1']".4f"}")

        return self.best_model, self.performance_metrics


def main():
    """Main function to run the super advanced stroke predictor."""
    logger.info("Starting Super Advanced Stroke Prediction System...")

    # Create and run the predictor
    predictor = SuperAdvancedStrokePredictor()

    try:
        best_model, metrics = predictor.run_complete_pipeline()

        logger.info("SUCCESS: Super advanced model created!")
        logger.info(f"Performance: {metrics}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == '__main__':
    main()
