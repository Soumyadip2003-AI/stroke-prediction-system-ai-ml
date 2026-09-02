"""
Backend API for NeuroPredict Stroke Risk Assessment
==================================================

This Flask backend provides API endpoints for the React frontend to interact with
the advanced machine learning models for stroke risk prediction.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ Pandas not available - using numpy only")

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

@app.route('/', methods=['GET'])
def home():
    """Simple root route for deployment health checks."""
    return jsonify({'status': 'Backend is running'})

@app.route('/health', methods=['GET'])
def health():
    """Health endpoint used by deployment checks."""
    return jsonify({'status': 'healthy'})

# Global variables for models
models = {}
feature_columns = None

# True only when every real model failed to load and a stub is standing in.
# /api/health reports this, so a dead deployment cannot look healthy.
using_mock_model = False

# Written by ml/train_stroke_model.py. Carries the decision threshold fitted
# out-of-fold and the held-out metrics, so nothing here is hardcoded.
def _load_model_metadata():
    try:
        with open('model_metadata.json') as fh:
            return json.load(fh)
    except Exception:
        return {}

MODEL_METADATA = _load_model_metadata()

# model_metadata.json is required, not optional: it carries the fitted decision
# threshold and the population rate that risk_category and flagged are derived
# from. Without it the threshold silently fell back to 0.5, which on calibrated
# probabilities that top out near 50% under-flags genuinely high-risk cases
# while /api/health still reported "healthy". Predictions are refused instead.
METADATA_OK = bool(MODEL_METADATA.get('threshold') and MODEL_METADATA.get('features'))
if not METADATA_OK:
    logger.error(
        "model_metadata.json missing or incomplete. Predictions are disabled: "
        "risk categories and flagging would be wrong. Run `npm run train:model`."
    )
# 0.5 is the wrong default on a 4.9% positive class; the trained threshold
# is far lower. Fall back to it only if the metadata file is missing.
DECISION_THRESHOLD = float(MODEL_METADATA.get('threshold', 0.5))

def load_models():
    """Load the served model and the feature order it was trained on.

    Exactly one model ships: the artifact written by ml/train_stroke_model.py,
    which is itself a soft-voting ensemble. The previous version searched about
    forty paths for eight more models that do not exist, and any that had
    loaded would have been handed the 21 features this module produces
    regardless of what they were trained on, then folded into the confidence
    calculation.
    """
    global models, feature_columns, using_mock_model

    try:
        models['main'] = joblib.load('stroke_prediction_model.pkl')
        logger.info("Loaded model: %s", MODEL_METADATA.get('model', 'unknown'))
    except Exception as exc:
        logger.error("Could not load stroke_prediction_model.pkl: %s", exc)

        class MockModel:
            """Stand-in so the process still boots. /api/health reports it."""

            def predict(self, X):
                return [0]

            def predict_proba(self, X):
                return [[0.8, 0.2]]

        models['main'] = MockModel()
        using_mock_model = True
        logger.warning("Serving a mock model. Predictions are meaningless until this is fixed.")

    # Authoritative feature order, written by the trainer from this same
    # preprocess_data.
    # model_metadata.json is authoritative: the trainer writes it from the same
    # preprocess_data this module serves with.
    feature_columns = MODEL_METADATA.get('features') or []
    if not feature_columns:
        logger.error("No feature order in model_metadata.json")

    return len(models) > 0


def preprocess_data(data):
    """Advanced preprocessing that creates proper features for the models."""
    # Create DataFrame if pandas is available, otherwise use dict
    if PANDAS_AVAILABLE:
        df = pd.DataFrame([data])
    else:
        df = data  # Use data directly as dict

    # Helper function to safely get values from either DataFrame or dict
    def get_value(key):
        if PANDAS_AVAILABLE and hasattr(df, 'columns'):
            return df[key].iloc[0] if key in df.columns else None
        else:
            return df.get(key)

    def set_value(key, value):
        if PANDAS_AVAILABLE and hasattr(df, 'columns'):
            df[key] = value
        else:
            df[key] = value

    # Normalize field names (pandas-specific)
    if PANDAS_AVAILABLE and hasattr(df, 'columns'):
        if 'residence_type' in df.columns and 'Residence_type' not in df.columns:
            df.rename(columns={'residence_type': 'Residence_type'}, inplace=True)

    # Convert data types and normalize values
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    for col in numeric_cols:
        value = get_value(col)
        if value is not None:
            try:
                set_value(col, float(value))
            except (ValueError, TypeError):
                set_value(col, 0.0)

    # Convert Yes/No to 1/0
    binary_cols = ['hypertension', 'heart_disease', 'ever_married']
    for col in binary_cols:
        value = get_value(col)
        if value is not None:
            if str(value).lower() in ['yes', '1', 'true']:
                set_value(col, 1)
            else:
                set_value(col, 0)

    # Normalize categorical values
    smoking_status = get_value('smoking_status')
    if smoking_status is not None:
        set_value('smoking_status', str(smoking_status).strip().lower())

    work_type = get_value('work_type')
    if work_type is not None:
        work_mapping = {
            'private': 'Private', 'self-employed': 'Self-employed',
            'children': 'children', 'govt_job': 'Govt_job',
            'never_worked': 'Never_worked'
        }
        normalized_work = work_mapping.get(str(work_type).lower(), work_type)
        set_value('work_type', normalized_work)

    # Create advanced features that the models expect
    age = get_value('age') or 0
    avg_glucose_level = get_value('avg_glucose_level') or 0
    bmi = get_value('bmi') or 0

    set_value('age_squared', age ** 2)
    set_value('glucose_log', np.log1p(avg_glucose_level))

    # Advanced BMI features
    set_value('bmi_category_normal', 1 if 18.5 <= bmi < 25 else 0)
    set_value('bmi_category_overweight', 1 if 25 <= bmi < 30 else 0)
    set_value('bmi_category_obese', 1 if 30 <= bmi < 35 else 0)
    set_value('bmi_category_severely_obese', 1 if bmi >= 35 else 0)
    set_value('bmi_category_underweight', 1 if bmi < 18.5 else 0)

    # Advanced glucose features
    set_value('glucose_category_normal', 1 if avg_glucose_level < 100 else 0)
    set_value('glucose_category_prediabetic', 1 if 100 <= avg_glucose_level < 126 else 0)
    set_value('glucose_category_diabetic', 1 if 126 <= avg_glucose_level < 200 else 0)
    set_value('glucose_category_severe', 1 if avg_glucose_level >= 200 else 0)

    # Interaction features
    set_value('age_bmi_interaction', age * bmi)
    set_value('age_glucose_interaction', age * avg_glucose_level)
    set_value('bmi_glucose_interaction', bmi * avg_glucose_level)

    # Risk indicators
    set_value('is_elderly', 1 if age > 65 else 0)
    set_value('is_obese', 1 if bmi >= 30 else 0)
    set_value('is_diabetic', 1 if avg_glucose_level > 126 else 0)
    set_value('is_prediabetic', 1 if 100 <= avg_glucose_level <= 126 else 0)

    # Risk scores
    hypertension = get_value('hypertension') or 0
    heart_disease = get_value('heart_disease') or 0
    set_value('cardiovascular_risk', hypertension + heart_disease)
    set_value('metabolic_risk', (1 if avg_glucose_level > 126 else 0) + (1 if bmi >= 30 else 0))
    set_value('total_risk_score', hypertension + heart_disease + (1 if age > 65 else 0) + (1 if avg_glucose_level > 150 else 0))

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
        if get_value(col) is None:
            set_value(col, default_val)

    # Create one-hot encoded features (exactly what models expect)
    if PANDAS_AVAILABLE:
        features_df = pd.DataFrame(index=df.index)
    else:
        features_df = {}

    # Helper function to set feature values
    def set_feature(key, value):
        if PANDAS_AVAILABLE:
            features_df[key] = value
        else:
            features_df[key] = value

    # Gender encoding
    gender = get_value('gender') or 'Male'
    set_feature('gender_Male', 1 if gender == 'Male' else 0)
    set_feature('gender_Female', 1 if gender == 'Female' else 0)
    set_feature('gender_Other', 1 if gender == 'Other' else 0)

    # Age and numeric features
    set_feature('age', age)
    set_feature('hypertension', get_value('hypertension') or 0)
    set_feature('heart_disease', get_value('heart_disease') or 0)

    # Ever married encoding
    set_feature('ever_married_Yes', 1 if get_value('ever_married') == 1 else 0)

    # Work type encoding
    work_types = ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked']
    work_val = get_value('work_type') or 'Private'
    for work in work_types:
        set_feature(f'work_type_{work}', 1 if work_val == work else 0)

    # Residence type encoding
    residence = get_value('Residence_type') or 'Urban'
    set_feature('Residence_type_Urban', 1 if residence == 'Urban' else 0)
    set_feature('Residence_type_Rural', 1 if residence == 'Rural' else 0)

    # Numeric features
    set_feature('avg_glucose_level', avg_glucose_level)
    set_feature('bmi', get_value('bmi') or 0)

    # Smoking status encoding
    smoking_types = ['never smoked', 'formerly smoked', 'smokes']
    smoking_val = get_value('smoking_status') or 'never smoked'
    for smoke in smoking_types:
        set_feature(f'smoking_status_{smoke}', 1 if smoking_val == smoke else 0)

    # Derived features
    set_feature('age_squared', get_value('age_squared') or 0)
    set_feature('glucose_log', get_value('glucose_log') or 0)

    # Convert to numpy array if pandas is not available
    if not PANDAS_AVAILABLE:
        # Create feature array from dict
        feature_list = []
        for key in feature_columns:
            feature_list.append(features_df.get(key, 0.0))
        return np.array([feature_list])
    else:
        # Ensure all values are numeric and fill NaN
        features_df = features_df.fillna(0).astype(float)
        return features_df

# Removed build_enhanced_features function - not needed for current implementation

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    real_models = len(models) > 0 and not using_mock_model
    serving = real_models and METADATA_OK
    return jsonify({
        'status': 'healthy' if serving else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': real_models,
        'metadata_loaded': METADATA_OK,
        'using_mock_model': using_mock_model,
        'model': MODEL_METADATA.get('model'),
        'decision_threshold': DECISION_THRESHOLD
    })

# Accepted values per field. Anything outside these silently produced a
# plausible-looking risk score before: preprocess_data coerces with
# `get_value(x) or 0`, so None, [] and "abc" all became 0, and an unknown
# category became an all-zero one-hot the model never saw in training.
# On a health endpoint that is worse than an error, so this rejects instead.
NUMERIC_RANGES = {
    'age': (1, 100),
    'avg_glucose_level': (50, 300),
    'bmi': (10, 50),
}

CATEGORICAL_VALUES = {
    'gender': {'male', 'female', 'other'},
    'ever_married': {'yes', 'no', '0', '1'},
    'hypertension': {'yes', 'no', '0', '1', 'true', 'false'},
    'heart_disease': {'yes', 'no', '0', '1', 'true', 'false'},
    'work_type': {'private', 'self-employed', 'children', 'govt_job', 'never_worked'},
    'residence_type': {'urban', 'rural'},
    'smoking_status': {'never smoked', 'formerly smoked', 'smokes', 'unknown'},
}


def validate_payload(data):
    """Return an error string, or None when the payload is usable."""
    for field, (low, high) in NUMERIC_RANGES.items():
        raw = data.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            return f"'{field}' must be a number, got {type(raw).__name__}"
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return f"'{field}' must be a number, got {raw!r}"
        if value != value or value in (float('inf'), float('-inf')):
            return f"'{field}' must be a finite number"
        if not low <= value <= high:
            return f"'{field}' must be between {low} and {high}, got {value:g}"

    for field, allowed in CATEGORICAL_VALUES.items():
        raw = data.get(field)
        if raw is None:
            continue  # presence is enforced by the required-field check
        if str(raw).strip().lower() not in allowed:
            return f"'{field}' must be one of {sorted(allowed)}, got {raw!r}"

    return None


@app.route('/api/predict', methods=['POST'])
def predict_stroke_risk():
    """Predict stroke risk using advanced AI models with self-learning."""
    try:
        if not METADATA_OK:
            return jsonify({
                'error': 'Model metadata unavailable; predictions are disabled '
                         'because the decision threshold is unknown.'
            }), 503

        # Check if models are loaded
        if not models:
            logger.error("No models loaded. Please ensure model files are available.")
            return jsonify({'error': 'No models loaded. Please ensure model files are available.'}), 500

        # request.json raises on malformed bodies, which the generic handler
        # below turned into a 500. A bad body is a client error.
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400

        # Validate required fields
        required_fields = ['age', 'gender', 'hypertension', 'heart_disease',
                          'avg_glucose_level', 'bmi', 'work_type', 'residence_type', 'smoking_status']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        validation_error = validate_payload(data)
        if validation_error:
            return jsonify({'error': validation_error}), 400

        # Preprocess data to match the 20 features models expect
        processed = preprocess_data(data)

        # The processed data should now have exactly 20 features
        # Handle both pandas DataFrame and numpy array cases
        if hasattr(processed, 'values'):
            # pandas DataFrame case
            X_final = processed.values
        else:
            # numpy array case
            X_final = processed

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

        # load_models() loads exactly one model, under the key 'main'. This
        # used to prefer an 'ultimate_xgboost' key annotated "95%+ accuracy";
        # no such model has ever existed in this repository, and the 95% was
        # the always-predict-no baseline for a 4.87% positive class.
        primary_model = 'main' if 'main' in models else (
            next(iter(models), None)
        )

        if primary_model is None:
            logger.error("No valid primary model found")
            return jsonify({'error': 'No valid models available for prediction'}), 500
        primary_prediction = predictions.get(primary_model, 0)
        primary_probability = probabilities.get(primary_model, 0.0)
        
        # Calculate risk category with realistic medical thresholds
        risk_percentage = primary_probability * 100

        # Probabilities are calibrated, so risk_percentage is a real estimated
        # probability and tops out well below 100. Two anchors matter and they
        # must not disagree:
        #
        #   the fitted threshold  -> whether the case is flagged
        #   the population rate   -> how the number reads to a person
        #
        # Anchoring the bands purely on base-rate multiples put the threshold
        # (4.02%) inside the Low band (2.44% to 4.87%), so a case at 4.7% was
        # labelled "Low Risk" while flagged=True. The Low/Moderate boundary is
        # therefore the threshold itself: Moderate and above is exactly the set
        # the model flags. Bands above it are multiples of the population rate.
        base_rate = MODEL_METADATA.get('positive_rate', 0.0487) * 100
        risk_multiple = risk_percentage / base_rate if base_rate else 0.0
        t = DECISION_THRESHOLD * 100

        # sorted() keeps the bands ordered even if a future threshold lands
        # above 2x the base rate.
        b1, b2, b3, b4 = sorted([t * 0.5, t, base_rate * 2, base_rate * 4])

        if risk_percentage < b1:
            risk_category = 'Very Low Risk'
            risk_color = '#10B981'
        elif risk_percentage < b2:
            risk_category = 'Low Risk'
            risk_color = '#34D399'
        elif risk_percentage < b3:
            risk_category = 'Moderate Risk'
            risk_color = '#F59E0B'
        elif risk_percentage < b4:
            risk_category = 'High Risk'
            risk_color = '#EF4444'
        else:
            risk_category = 'Very High Risk'
            risk_color = '#DC2626'

        # 'confidence' used to bin by risk magnitude: >50 High, >25 Medium,
        # else Low. That is not confidence, it is the risk number again under
        # another name, and once probabilities were calibrated (they top out
        # near 26%) it returned 'Low' for every single user, implying the
        # model was unsure about everyone. A single model has no honest
        # per-prediction confidence without conformal intervals or ensemble
        # variance, so the field is gone rather than invented.

        # Generate health analysis
        health_analysis = generate_health_analysis(data)

        # Generate recommendations
        recommendations = generate_recommendations(data, risk_percentage)

        # Prepare response
        response = {
            'probability': primary_probability,
            'risk_percentage': risk_percentage,
            'risk_category': risk_category,
            'risk_color': risk_color,
            # model.predict() thresholds at 0.5, which is meaningless on a
            # 4.87% positive class and is the entire reason a threshold is
            # fitted. Left as-is, 'prediction' said 0 for people 'flagged'
            # said were at risk, for every case between 4% and 50%.
            'prediction': int(primary_probability >= DECISION_THRESHOLD),
            'flagged': bool(primary_probability >= DECISION_THRESHOLD),
            'risk_multiple': round(risk_multiple, 1),
            'population_base_rate': round(base_rate, 2),
            'calibrated': bool(MODEL_METADATA.get('calibrated', False)),
            'decision_threshold': DECISION_THRESHOLD,
            # Measured on a held-out split by ml/train_stroke_model.py, not typed in.
            'model_performance': MODEL_METADATA.get('metrics_holdout', {}),
            # 'all_predictions' / 'all_probabilities' are gone. They existed to
            # report per-model output back when the response claimed nine
            # models; with one model they were {'main': x}, duplicating
            # 'probability' exactly.
            'health_analysis': health_analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

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
                'loaded': True,
                'available': True
            }
        except Exception as e:
            model_info[name] = {'error': str(e), 'loaded': False, 'available': False}
    
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

# Load models when module is imported (for both direct execution and gunicorn)
models_loaded = load_models()

if __name__ == '__main__':
    if models_loaded:
        logger.info("Starting NeuroPredict API server...")
        app.run(debug=True, host='0.0.0.0', port=5002)
    else:
        logger.error("Failed to load models. Please ensure model files exist.")
        exit(1)
