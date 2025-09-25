#!/usr/bin/env python3
"""
Simple script to start the backend with model loading verification
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_models_exist():
    """Verify that all required model files exist."""
    required_files = ['stroke_prediction_model.pkl', 'scaler.pkl', 'feature_columns.pkl']
    missing_files = []

    logger.info("🔍 Checking for required model files...")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            logger.info(f"✅ {file} found ({size} bytes)")
        else:
            logger.error(f"❌ {file} missing")
            missing_files.append(file)

    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False

    return True

def test_model_loading():
    """Test loading models manually."""
    logger.info("🧪 Testing model loading...")

    models = {}
    scalers = {}
    feature_columns = None

    try:
        import joblib

        # Try to load main model
        try:
            models['main'] = joblib.load('stroke_prediction_model.pkl')
            logger.info("✅ Main model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load main model: {str(e)}")
            return False

        # Load scaler
        try:
            scalers['main'] = joblib.load('scaler.pkl')
            logger.info("✅ Scaler loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Scaler loading failed: {str(e)}")

        # Load feature columns
        try:
            feature_columns = joblib.load('feature_columns.pkl')
            logger.info("✅ Feature columns loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Feature columns loading failed: {str(e)}")

        # Check if we have models
        if len(models) == 0:
            logger.error("❌ No models loaded!")
            return False

        logger.info(f"✅ Successfully loaded {len(models)} models: {list(models.keys())}")
        logger.info(f"✅ Feature columns: {len(feature_columns) if feature_columns else 0} features")
        return True

    except ImportError as e:
        logger.error(f"❌ Failed to import joblib: {str(e)}")
        return False

def start_backend():
    """Start the backend server."""
    logger.info("🚀 Starting NeuroPredict Backend...")

    # Verify models exist
    if not verify_models_exist():
        logger.error("Cannot start backend: missing model files")
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
            app.run(debug=True, host='0.0.0.0', port=5002)
        else:
            backend_logger.error("Failed to load models. Please ensure model files exist.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to start backend: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_backend()
