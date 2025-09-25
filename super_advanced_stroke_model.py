import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperAdvancedStrokePredictor:
    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', output_dir='super_advanced_models', random_state=42):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=5, random_state=random_state)
        self.ensemble = None
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=25)

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with advanced feature engineering."""
        logger.info("Loading and preprocessing dataset...")
        df = pd.read_csv(self.data_path)

        # Handle missing values
        df['bmi'].fillna(df['bmi'].median(), inplace=True)

        # Advanced feature engineering
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['glucose_bmi_ratio'] = df['avg_glucose_level'] / df['bmi']
        df['age_squared'] = df['age'] ** 2
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['underweight', 'normal', 'overweight', 'obese'])
        df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 126, float('inf')], labels=['normal', 'prediabetes', 'diabetes'])
        df['age_risk'] = df['age'].apply(lambda x: 1 if x > 55 else 0)
        df['bmi_risk'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)
        df['glucose_risk'] = df['avg_glucose_level'].apply(lambda x: 1 if x > 100 else 0)

        # Encode categorical variables
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'bmi_category', 'glucose_category']
        df_processed = pd.get_dummies(df, columns=categorical_columns)

        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col not in ['stroke', 'id']]
        X = df_processed[feature_columns]
        y = df_processed['stroke']

        logger.info(f"Dataset loaded: {X.shape[0]} samples, {len(feature_columns)} features")
        logger.info(f"Stroke distribution: {y.value_counts().to_dict()}")

        return X, y, feature_columns

    def apply_unsupervised_learning(self, X):
        """Apply unsupervised learning for feature engineering."""
        logger.info("Applying unsupervised learning...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        logger.info(f"PCA reduced features to {X_pca.shape[1]} components")

        # KMeans clustering for additional features
        clusters = self.kmeans.fit_predict(X_pca)
        cluster_features = pd.get_dummies(clusters, prefix='cluster')

        # Combine PCA and cluster features
        X_unsupervised = np.hstack([X_pca, cluster_features.values])

        return X_unsupervised

    def train_models(self, X, y):
        """Train multiple advanced models with hyperparameter tuning."""
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

        logger.info("Applying SMOTE for imbalanced data...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        logger.info(f"After SMOTE: {X_train_resampled.shape}")

        # Apply unsupervised learning
        X_train_unsupervised = self.apply_unsupervised_learning(X_train_resampled)
        X_test_unsupervised = self.apply_unsupervised_learning(X_test)

        # Apply feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_unsupervised, y_train_resampled)
        X_test_selected = self.feature_selector.transform(X_test_unsupervised)

        logger.info(f"Final feature set: {X_train_selected.shape[1]} features")

        # Random Forest with hyperparameter tuning
        logger.info("Training Random Forest with hyperparameter optimization...")
        rf_params = {
            'n_estimators': [400, 500],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train_selected, y_train_resampled)
        self.models['randomforest'] = rf_grid.best_estimator_
        logger.info(f"Best RF params: {rf_grid.best_params_}")

        # Gradient Boosting with hyperparameter tuning
        logger.info("Training Gradient Boosting with hyperparameter optimization...")
        gb_params = {
            'n_estimators': [400, 500],
            'max_depth': [10, 15],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9]
        }
        gb_base = GradientBoostingClassifier(random_state=self.random_state)
        gb_grid = GridSearchCV(gb_base, gb_params, cv=5, scoring='f1', n_jobs=-1)
        gb_grid.fit(X_train_selected, y_train_resampled)
        self.models['gradientboosting'] = gb_grid.best_estimator_
        logger.info(f"Best GB params: {gb_grid.best_params_}")

        # Extra Trees with hyperparameter tuning
        logger.info("Training Extra Trees with hyperparameter optimization...")
        et_params = {
            'n_estimators': [400, 500],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5]
        }
        et_base = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
        et_grid = GridSearchCV(et_base, et_params, cv=5, scoring='f1', n_jobs=-1)
        et_grid.fit(X_train_selected, y_train_resampled)
        self.models['extratrees'] = et_grid.best_estimator_
        logger.info(f"Best ET params: {et_grid.best_params_}")

        # MLP Classifier with hyperparameter tuning
        logger.info("Training MLP Classifier with hyperparameter optimization...")
        mlp_params = {
            'hidden_layer_sizes': [(100, 50, 25), (150, 75, 25)],
            'max_iter': [1000],
            'learning_rate_init': [0.001, 0.01]
        }
        mlp_base = MLPClassifier(random_state=self.random_state)
        mlp_grid = GridSearchCV(mlp_base, mlp_params, cv=5, scoring='f1', n_jobs=-1)
        mlp_grid.fit(X_train_selected, y_train_resampled)
        self.models['mlpclassifier'] = mlp_grid.best_estimator_
        logger.info(f"Best MLP params: {mlp_grid.best_params_}")

        # Balanced Random Forest
        logger.info("Training Balanced Random Forest...")
        brf = BalancedRandomForestClassifier(n_estimators=500, random_state=self.random_state, n_jobs=-1)
        brf.fit(X_train_selected, y_train_resampled)
        self.models['balanced_rf'] = brf

        # AdaBoost
        logger.info("Training AdaBoost...")
        ada = AdaBoostClassifier(n_estimators=500, random_state=self.random_state)
        ada.fit(X_train_selected, y_train_resampled)
        self.models['adaboost'] = ada

        # Create Stacking Ensemble
        logger.info("Creating stacking ensemble...")
        base_estimators = [
            ('rf', self.models['randomforest']),
            ('gb', self.models['gradientboosting']),
            ('et', self.models['extratrees']),
            ('mlp', self.models['mlpclassifier']),
            ('brf', self.models['balanced_rf']),
            ('ada', self.models['adaboost'])
        ]

        meta_estimator = GradientBoostingClassifier(n_estimators=200, random_state=self.random_state)

        self.ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=5,
            n_jobs=-1
        )
        self.ensemble.fit(X_train_selected, y_train_resampled)

        # Evaluate all models
        logger.info("Evaluating models...")
        for name, model in self.models.items():
            y_pred = model.predict(X_test_selected)
            y_proba = model.predict_proba(X_test_selected)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }

            logger.info(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

        # Evaluate ensemble
        y_pred_ens = self.ensemble.predict(X_test_selected)
        y_proba_ens = self.ensemble.predict_proba(X_test_selected)[:, 1]

        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ens),
            'precision': precision_score(y_test, y_pred_ens, zero_division=0),
            'recall': recall_score(y_test, y_pred_ens, zero_division=0),
            'f1': f1_score(y_test, y_pred_ens, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_ens)
        }

        logger.info(f"Ensemble: Accuracy={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}, ROC-AUC={ensemble_metrics['roc_auc']:.4f}")

    def save_models(self):
        """Save all trained models and components."""
        logger.info("Saving models...")

        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, self.output_dir / f'{name}_model.pkl')

        # Save ensemble
        joblib.dump(self.ensemble, self.output_dir / 'stacking_ensemble.pkl')

        # Save preprocessing components
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        joblib.dump(self.pca, self.output_dir / 'pca.pkl')
        joblib.dump(self.kmeans, self.output_dir / 'kmeans.pkl')
        joblib.dump(self.feature_selector, self.output_dir / 'feature_selector.pkl')

        # Save feature columns
        feature_columns = [f'pca_component_{i}' for i in range(self.pca.n_components_)] + [f'cluster_{i}' for i in range(5)]
        joblib.dump(feature_columns, self.output_dir / 'feature_columns.pkl')

        logger.info("✅ All models saved successfully!")

    def run_training(self):
        """Run the complete advanced training pipeline."""
        try:
            # Load and preprocess data
            X, y, feature_columns = self.load_and_preprocess_data()

            # Train models
            self.train_models(X, y)

            # Save models
            self.save_models()

            logger.info("🎉 Super advanced training completed successfully!")
            logger.info(f"Models saved to: {self.output_dir}")
            logger.info("This is now the most advanced stroke prediction model using:")
            logger.info("- Supervised Learning: RF, GB, ET, MLP, BalancedRF, AdaBoost")
            logger.info("- Unsupervised Learning: PCA + KMeans clustering")
            logger.info("- Advanced Ensemble: StackingClassifier with meta-learner")
            logger.info("- Data Balancing: SMOTE")
            logger.info("- Feature Engineering: Interactions, ratios, polynomial features")
            logger.info("- Feature Selection: Mutual Information")
            logger.info("- Hyperparameter Optimization: GridSearchCV")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

if __name__ == '__main__':
    predictor = SuperAdvancedStrokePredictor()
    predictor.run_training()
