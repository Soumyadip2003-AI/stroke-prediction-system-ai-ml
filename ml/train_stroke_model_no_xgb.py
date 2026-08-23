import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrokeModelTrainer:
    def __init__(self, data_path='healthcare-dataset-stroke-data.csv', output_dir='models', random_state=42):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)
        self.ensemble = None

        logger.info(f"Training script initialized for {data_path}")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Handle missing BMI values
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # Convert categorical variables
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        df_processed = pd.get_dummies(df, columns=categorical_columns)
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != 'stroke' and col != 'id']
        X = df_processed[feature_columns]
        y = df_processed['stroke']
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {len(feature_columns)} features")
        logger.info(f"Stroke distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns

    def train_models(self, X, y):
        """Train multiple models and create ensemble."""
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        logger.info("Applying SMOTE for imbalanced data...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE: {X_train_resampled.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        logger.info(f"PCA components: {X_train_pca.shape[1]}")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=self.random_state)
        rf.fit(X_train_pca, y_train_resampled)
        self.models['randomforest'] = rf
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=8, random_state=self.random_state)
        gb.fit(X_train_pca, y_train_resampled)
        self.models['gradientboosting'] = gb
        
        # Train Extra Trees
        logger.info("Training Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=self.random_state)
        et.fit(X_train_pca, y_train_resampled)
        self.models['extratrees'] = et
        
        # Create and train ensemble
        logger.info("Creating ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('et', et)
            ],
            voting='soft'
        )
        self.ensemble.fit(X_train_pca, y_train_resampled)
        
        # Evaluate models
        logger.info("Evaluating models...")
        for name, model in self.models.items():
            y_pred = model.predict(X_test_pca)
            y_proba = model.predict_proba(X_test_pca)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            logger.info(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # Evaluate ensemble
        y_pred_ens = self.ensemble.predict(X_test_pca)
        y_proba_ens = self.ensemble.predict_proba(X_test_pca)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ens),
            'precision': precision_score(y_test, y_pred_ens, zero_division=0),
            'recall': recall_score(y_test, y_pred_ens, zero_division=0),
            'f1': f1_score(y_test, y_pred_ens, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_ens)
        }
        
        logger.info(f"Ensemble: Accuracy={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}")

    def save_models(self):
        """Save trained models."""
        logger.info("Saving models...")
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, self.output_dir / f'{name}_model.pkl')
            logger.info(f"Saved {name} model")
        
        # Save ensemble
        joblib.dump(self.ensemble, self.output_dir / 'voting_ensemble.pkl')
        logger.info("Saved ensemble model")
        
        # Save scaler and PCA
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        joblib.dump(self.pca, self.output_dir / 'pca.pkl')
        
        # Save feature columns
        feature_columns = [f'pca_component_{i}' for i in range(self.pca.n_components_)]
        joblib.dump(feature_columns, self.output_dir / 'feature_columns.pkl')
        
        logger.info("Models saved successfully!")

    def run_training(self):
        """Run the complete training pipeline."""
        try:
            # Load and preprocess data
            X, y, feature_columns = self.load_and_preprocess_data()
            
            # Train models
            self.train_models(X, y)
            
            # Save models
            self.save_models()
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"Models saved to: {self.output_dir}")
            logger.info("Use these models with the trained scaler and PCA for predictions.")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

if __name__ == '__main__':
    trainer = StrokeModelTrainer(data_path='healthcare-dataset-stroke-data.csv')
    trainer.run_training()
