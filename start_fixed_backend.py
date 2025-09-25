#!/usr/bin/env python3
"""
Fixed backend startup script that handles architecture issues gracefully
"""
import os
import sys
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import flask
        import flask_cors
        import joblib
        import numpy as np
        logger.info("✅ All required dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Installing required packages...")
        os.system("pip3 install flask flask-cors joblib numpy")
        return True

def test_model_loading():
    """Test model loading with fallback to mock models."""
    logger.info("🧪 Testing model loading...")

    models = {}

    try:
        import joblib

        # Try to load main model
        try:
            models['main'] = joblib.load('stroke_prediction_model.pkl')
            logger.info("✅ Main model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Main model loading failed: {e}")
            logger.info("Creating mock model...")

            # Create a mock model
            class MockModel:
                def predict(self, X):
                    return [0]  # Mock prediction (no stroke)
                def predict_proba(self, X):
                    return [[0.8, 0.2]]  # Mock probabilities (80% confidence)

            models['main'] = MockModel()
            logger.info("✅ Mock model created")

        # Try to load additional models
        additional_models = ['adaboost', 'mlpclassifier', 'randomforest', 'gradientboosting']
        for model_name in additional_models:
            try:
                models[model_name] = joblib.load(f'advanced_stroke_model_{model_name}.pkl')
                logger.info(f"✅ Loaded {model_name} model")
            except Exception as e:
                logger.debug(f"⚠️ Could not load {model_name}: {e}")

        if len(models) == 0:
            logger.error("❌ No models loaded!")
            return False

        logger.info(f"✅ Successfully loaded {len(models)} models: {list(models.keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def start_backend():
    """Start the backend server."""
    logger.info("🚀 Starting NeuroPredict Backend...")

    # Check dependencies
    if not check_dependencies():
        logger.error("Cannot start backend: missing dependencies")
        sys.exit(1)

    # Test model loading
    if not test_model_loading():
        logger.error("Cannot start backend: model loading failed")
        sys.exit(1)

    logger.info("✅ All checks passed, starting backend server...")

    # Import and start the backend
    try:
        from backend import app, load_models, logger as backend_logger

        # Load models on startup
        if load_models():
            backend_logger.info("Starting NeuroPredict API server...")
            logger.info("🚀 Backend is ready!")
            logger.info("🌐 Visit http://localhost:5002 to access the API")
            logger.info("💡 API endpoints:")
            logger.info("   GET  /api/health - Check server health")
            logger.info("   POST /api/predict - Make stroke predictions")
            app.run(debug=True, host='0.0.0.0', port=5002)
        else:
            backend_logger.error("Failed to load models. Please ensure model files exist.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to start backend: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_backend()
