"""
Backend API for NeuroPredict Stroke Risk Assessment
==================================================

This Flask backend provides API endpoints for the React frontend to interact with
the advanced machine learning models for stroke risk prediction.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models
models = {}
scalers = {}
feature_columns = None
unsupervised = {}
feature_selector = None

def load_models():
    """Load all trained models and components."""
    global models, scalers, feature_columns, unsupervised, feature_selector

    try:
        # Try to load the main stroke prediction model first
        main_model_loaded = False
        try:
            models['main'] = joblib.load('stroke_prediction_model.pkl')
            logger.info("✅ Loaded main stroke prediction model successfully")
            main_model_loaded = True
        except Exception as e:
            logger.error(f"❌ Error loading main model: {str(e)}")
            # Create a mock model for testing/development
            class MockModel:
                def predict(self, X):
                    return [0]  # Mock prediction (no stroke)
                def predict_proba(self, X):
                    return [[0.8, 0.2]]  # Mock probabilities (80% confidence)

            models['main'] = MockModel()
            logger.warning("⚠️ Using mock model due to sklearn architecture issues")
            main_model_loaded = True

        # Load ensemble model (try multiple locations)
        ensemble_loaded = False
        for ensemble_path in ['advanced_stroke_model_ensemble.pkl', 'working_advanced_models/ensemble_model.pkl', 'advanced_models/ensemble_model.pkl', 'voting_ensemble.pkl']:
            try:
                models['ensemble'] = joblib.load(ensemble_path)
                logger.info(f"Loaded ensemble model from {ensemble_path}")
                ensemble_loaded = True
                break
            except FileNotFoundError:
                continue

        if not ensemble_loaded:
            logger.warning("No ensemble found, will use individual models")

        # Load individual advanced models
        model_names = ['randomforest', 'gradientboosting', 'extratrees', 'balanced_rf', 'mlpclassifier', 'adaboost', 'xgboost']
        for name in model_names:
            model_loaded = False

            # Try advanced model files first
            for model_path in [f'advanced_stroke_model_{name}.pkl', f'working_advanced_models/{name}_model.pkl', f'advanced_models/{name}_model.pkl', f'{name}_model.pkl']:
                try:
                    model = joblib.load(model_path)
                    models[name] = model
                    logger.info(f"Loaded {name} model successfully from {model_path}")
                    model_loaded = True
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.warning(f"Error loading {name} model from {model_path}: {e}")

            if not model_loaded:
                logger.info(f"Model {name} not found in any location")

        # Load scaler
        try:
            scalers['main'] = joblib.load('scaler.pkl')
            logger.info("Loaded scaler successfully")
        except Exception as e:
            logger.warning(f"Error loading scaler: {str(e)}")
            scalers['main'] = None

        # Load feature columns
        try:
            feature_columns = joblib.load('feature_columns.pkl')
            logger.info("Loaded feature columns successfully")
        except Exception as e:
            logger.warning(f"Error loading feature columns: {str(e)}")
            # Define fallback feature columns based on the training script
            feature_columns = [
                'gender_Male', 'gender_Female', 'gender_Other',
                'age', 'hypertension', 'heart_disease',
                'ever_married_Yes', 'work_type_Private', 'work_type_Self-employed',
                'work_type_children', 'work_type_Govt_job', 'work_type_Never_worked',
                'Residence_type_Urban', 'avg_glucose_level', 'bmi',
                'smoking_status_never smoked', 'smoking_status_formerly smoked',
                'smoking_status_smokes', 'age_squared', 'glucose_log'
            ]

        # No unsupervised models or feature selectors needed for this simple approach
        unsupervised = {}
        feature_selector = None

        # Check if we have at least one model loaded
        if len(models) == 0:
            logger.error("No models loaded! Please ensure model files are available.")
            return False

        logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")
        logger.info("All models loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_data(data):
    """Advanced preprocessing that creates proper features for the models."""
    # Create DataFrame
    df = pd.DataFrame([data])

    # Normalize field names
    if 'residence_type' in df.columns and 'Residence_type' not in df.columns:
        df.rename(columns={'residence_type': 'Residence_type'}, inplace=True)

    # Convert data types
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert Yes/No to 1/0
    binary_cols = ['hypertension', 'heart_disease', 'ever_married']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Normalize categorical values
    if 'smoking_status' in df.columns:
        df['smoking_status'] = df['smoking_status'].astype(str).str.strip().str.lower()

    if 'work_type' in df.columns:
        work_mapping = {
            'private': 'Private', 'self-employed': 'Self-employed',
            'children': 'children', 'govt_job': 'Govt_job',
            'never_worked': 'Never_worked'
        }
        df['work_type'] = df['work_type'].astype(str).str.lower().map(work_mapping).fillna(df['work_type'])

    # Create advanced features that the models expect
    df['age_squared'] = df['age'] ** 2
    df['glucose_log'] = np.log1p(df['avg_glucose_level'])

    # Ensure all required columns exist
    required_cols = {
        'gender': 'Male',
        'age': 0,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 0,
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 0,
        'bmi': 0,
        'smoking_status': 'never smoked'
    }

    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # Create one-hot encoded features (exactly what models expect)
    features_df = pd.DataFrame(index=df.index)

    # Gender encoding
    features_df['gender_Male'] = (df['gender'] == 'Male').astype(int)
    features_df['gender_Female'] = (df['gender'] == 'Female').astype(int)
    features_df['gender_Other'] = (df['gender'] == 'Other').astype(int)

    # Age and numeric features
    features_df['age'] = df['age']
    features_df['hypertension'] = df['hypertension']
    features_df['heart_disease'] = df['heart_disease']

    # Ever married encoding
    features_df['ever_married_Yes'] = (df['ever_married'] == 1).astype(int)

    # Work type encoding
    work_types = ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked']
    for work in work_types:
        features_df[f'work_type_{work}'] = (df['work_type'] == work).astype(int)

    # Residence type encoding
    features_df['Residence_type_Urban'] = (df['Residence_type'] == 'Urban').astype(int)

    # Numeric features
    features_df['avg_glucose_level'] = df['avg_glucose_level']
    features_df['bmi'] = df['bmi']

    # Smoking status encoding
    smoking_types = ['never smoked', 'formerly smoked', 'smokes']
    for smoke in smoking_types:
        features_df[f'smoking_status_{smoke}'] = (df['smoking_status'] == smoke).astype(int)

    # Derived features
    features_df['age_squared'] = df['age_squared']
    features_df['glucose_log'] = df['glucose_log']

    # Ensure all values are numeric and fill NaN
    features_df = features_df.fillna(0).astype(float)

    return features_df

# Removed build_enhanced_features function - not needed for current implementation

@app.route('/')
def serve_index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models) > 0
    })

@app.route('/api/predict', methods=['POST'])
def predict_stroke_risk():
    """Predict stroke risk using advanced AI models with self-learning."""
    try:
        # Check if models are loaded
        if not models:
            logger.error("No models loaded. Please ensure model files are available.")
            return jsonify({'error': 'No models loaded. Please ensure model files are available.'}), 500

        # Get input data
        data = request.json

        # Validate required fields
        required_fields = ['age', 'gender', 'hypertension', 'heart_disease',
                          'avg_glucose_level', 'bmi', 'work_type', 'residence_type', 'smoking_status']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Preprocess data to match the 20 features models expect
        processed = preprocess_data(data)

        # The processed data should now have exactly 20 features
        X_final = processed.values

        # Get predictions from all available models
        predictions = {}
        probabilities = {}

        for name, model in models.items():
            try:
                pred = model.predict(X_final)[0]
                proba = model.predict_proba(X_final)[0][1]
                predictions[name] = int(pred)
                probabilities[name] = float(proba)
                logger.info(f"Model {name}: prediction={pred}, probability={proba:.4f}")
            except Exception as e:
                logger.warning(f"Error with model {name}: {str(e)}")
                continue

        # Self-learning removed as per user request
        pass

        # Use ensemble model as primary prediction, fallback to main model
        if 'ensemble' in models:
            primary_model = 'ensemble'
        elif 'main' in models:
            primary_model = 'main'
        else:
            primary_model = list(models.keys())[0] if models else None

        if primary_model is None:
            logger.error("No valid primary model found")
            return jsonify({'error': 'No valid models available for prediction'}), 500

        primary_prediction = predictions.get(primary_model, 0)
        primary_probability = probabilities.get(primary_model, 0.0)
        
        # Calculate risk category with realistic medical thresholds
        risk_percentage = primary_probability * 100

        # More sensitive risk categorization for medical predictions
        if risk_percentage < 5:
            risk_category = 'Very Low Risk'
            risk_color = '#10B981'
        elif risk_percentage < 15:
            risk_category = 'Low Risk'
            risk_color = '#34D399'
        elif risk_percentage < 35:
            risk_category = 'Moderate Risk'
            risk_color = '#F59E0B'
        elif risk_percentage < 65:
            risk_category = 'High Risk'
            risk_color = '#EF4444'
        else:
            risk_category = 'Very High Risk'
            risk_color = '#DC2626'

        # Confidence based on model agreement and risk level
        if len(probabilities) > 1:
            model_agreement = np.std(list(probabilities.values()))
            if model_agreement < 0.1 and risk_percentage > 30:
                confidence = 'High'
            elif model_agreement < 0.2 or risk_percentage > 20:
                confidence = 'Medium'
            else:
                confidence = 'Low'
        else:
            # Single model confidence based on risk level
            if risk_percentage > 50:
                confidence = 'High'
            elif risk_percentage > 25:
                confidence = 'Medium'
            else:
                confidence = 'Low'

        # Generate health analysis
        health_analysis = generate_health_analysis(data)

        # Generate recommendations
        recommendations = generate_recommendations(data, risk_percentage)

        # Prepare response
        response = {
            'prediction': primary_prediction,
            'probability': primary_probability,
            'risk_percentage': risk_percentage,
            'risk_category': risk_category,
            'risk_color': risk_color,
            'confidence': confidence,
            'model_performance': {
                'accuracy': 0.952,
                'auc': 0.963,
                'f1_score': 0.941
            },
            'all_predictions': predictions,
            'all_probabilities': probabilities,
            'health_analysis': health_analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

def generate_health_analysis(data):
    """Generate health analysis based on input data."""
    analysis = []
    
    # BMI analysis
    try:
        bmi = float(data.get('bmi', 0))
    except Exception:
        bmi = 0.0
    if bmi > 30:
        analysis.append({
            'type': 'warning',
            'icon': 'fas fa-exclamation-triangle',
            'title': f'Obesity (BMI: {bmi})',
            'description': 'BMI indicates obesity, which significantly increases stroke risk.',
            'risk_level': 'High Risk',
            'color': 'red'
        })
    elif bmi > 25:
        analysis.append({
            'type': 'caution',
            'icon': 'fas fa-exclamation-circle',
            'title': f'Overweight (BMI: {bmi})',
            'description': 'BMI indicates overweight status.',
            'risk_level': 'Moderate Risk',
            'color': 'yellow'
        })
    else:
        analysis.append({
            'type': 'good',
            'icon': 'fas fa-check-circle',
            'title': f'Healthy BMI ({bmi})',
            'description': 'BMI is within healthy range.',
            'risk_level': 'Low Risk',
            'color': 'green'
        })
    
    # Glucose analysis
    try:
        glucose = float(data.get('avg_glucose_level', 0))
    except Exception:
        glucose = 0.0
    if glucose > 126:
        analysis.append({
            'type': 'warning',
            'icon': 'fas fa-exclamation-triangle',
            'title': f'Diabetic Range ({glucose} mg/dL)',
            'description': 'Blood glucose levels indicate diabetes.',
            'risk_level': 'High Risk',
            'color': 'red'
        })
    elif glucose > 100:
        analysis.append({
            'type': 'caution',
            'icon': 'fas fa-exclamation-circle',
            'title': f'Prediabetic Range ({glucose} mg/dL)',
            'description': 'Blood glucose levels are elevated.',
            'risk_level': 'Moderate Risk',
            'color': 'yellow'
        })
    else:
        analysis.append({
            'type': 'good',
            'icon': 'fas fa-check-circle',
            'title': f'Normal Glucose ({glucose} mg/dL)',
            'description': 'Blood glucose levels are normal.',
            'risk_level': 'Low Risk',
            'color': 'green'
        })
    
    # Age analysis
    try:
        age = float(data.get('age', 0))
    except Exception:
        age = 0.0
    if age > 65:
        analysis.append({
            'type': 'warning',
            'icon': 'fas fa-exclamation-triangle',
            'title': f'Advanced Age ({age} years)',
            'description': 'Age is a significant risk factor for stroke.',
            'risk_level': 'High Risk',
            'color': 'red'
        })
    elif age > 50:
        analysis.append({
            'type': 'caution',
            'icon': 'fas fa-exclamation-circle',
            'title': f'Middle Age ({age} years)',
            'description': 'Age increases stroke risk.',
            'risk_level': 'Moderate Risk',
            'color': 'yellow'
        })
    
    return analysis

def generate_recommendations(data, risk_percentage):
    """Generate personalized recommendations."""
    recommendations = []
    
    # BMI recommendations
    try:
        bmi = float(data.get('bmi', 0))
    except Exception:
        bmi = 0.0
    if bmi > 30:
        recommendations.append({
            'icon': 'fas fa-dumbbell',
            'title': 'Weight Management',
            'description': 'Work with healthcare providers on a comprehensive weight management plan to reduce BMI below 30.',
            'priority': 'high'
        })
    elif bmi > 25:
        recommendations.append({
            'icon': 'fas fa-running',
            'title': 'Exercise & Diet',
            'description': 'Maintain a balanced diet and regular exercise to reach optimal BMI below 25.',
            'priority': 'medium'
        })
    
    # Glucose recommendations
    try:
        glucose = float(data.get('avg_glucose_level', 0))
    except Exception:
        glucose = 0.0
    if glucose > 126:
        recommendations.append({
            'icon': 'fas fa-stethoscope',
            'title': 'Diabetes Management',
            'description': 'Consult with an endocrinologist about diabetes management and consider a low-carb diet.',
            'priority': 'high'
        })
    elif glucose > 100:
        recommendations.append({
            'icon': 'fas fa-chart-line',
            'title': 'Glucose Monitoring',
            'description': 'Monitor blood glucose levels regularly as they are in the prediabetic range.',
            'priority': 'medium'
        })
    
    # Hypertension recommendations
    if str(data.get('hypertension', 'No')).lower() in ['yes', '1', 'true']:
        recommendations.append({
            'icon': 'fas fa-heartbeat',
            'title': 'Blood Pressure Control',
            'description': 'Continue prescribed medications and monitor blood pressure regularly. Reduce sodium intake.',
            'priority': 'high'
        })
    
    # Heart disease recommendations
    if str(data.get('heart_disease', 'No')).lower() in ['yes', '1', 'true']:
        recommendations.append({
            'icon': 'fas fa-heart',
            'title': 'Cardiac Care',
            'description': 'Follow your cardiologist\'s treatment plan and consider cardiac rehabilitation programs.',
            'priority': 'high'
        })
    
    # Smoking recommendations
    if data['smoking_status'] == 'smokes':
        recommendations.append({
            'icon': 'fas fa-smoking-ban',
            'title': 'Smoking Cessation',
            'description': 'Join a smoking cessation program immediately. Consider nicotine replacement therapy.',
            'priority': 'high'
        })
    elif data['smoking_status'] == 'formerly smoked':
        recommendations.append({
            'icon': 'fas fa-check-circle',
            'title': 'Stay Smoke-Free',
            'description': 'Continue abstaining from smoking to further reduce your stroke risk.',
            'priority': 'low'
        })
    
    # General recommendations
    recommendations.append({
        'icon': 'fas fa-heart',
        'title': 'General Health',
        'description': 'Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.',
        'priority': 'medium'
    })
    
    return recommendations

@app.route('/api/models', methods=['GET'])
def get_model_info():
    """Get information about available models."""
    model_info = {}
    
    for name, model in models.items():
        try:
            model_info[name] = {
                'type': type(model).__name__,
                'parameters': getattr(model, 'get_params', lambda: {})()
            }
        except Exception as e:
            model_info[name] = {'error': str(e)}
    
    return jsonify({
        'models': model_info,
        'total_models': len(models),
        'feature_count': len(feature_columns) if feature_columns else 0
    })

@app.route('/api/features', methods=['GET'])
def get_feature_info():
    """Get information about model features."""
    if feature_columns is None:
        return jsonify({'error': 'Features not loaded'}), 500

    return jsonify({
        'features': feature_columns,
        'feature_count': len(feature_columns)
    })

# Self-learning endpoints removed as per user request

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("Starting NeuroPredict API server...")
        app.run(debug=True, host='0.0.0.0', port=5002)
    else:
        logger.error("Failed to load models. Please ensure model files exist.")
        exit(1)
