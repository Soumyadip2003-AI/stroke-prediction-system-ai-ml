#!/usr/bin/env python3
"""
Demonstration of Self-Learning Stroke Prediction System
=======================================================

This script demonstrates how the AI model can learn and improve over time
through user feedback and new data.

Usage:
    python demo_self_learning.py
"""

import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_self_learning():
    """Demonstrate the self-learning capabilities."""

    logger.info("🤖 Starting Self-Learning Stroke Prediction Demo")
    logger.info("=" * 60)

    try:
        from self_learning_system import SelfLearningStrokePredictor

        # Initialize the self-learning system
        logger.info("📚 Initializing Self-Learning System...")
        self_learner = SelfLearningStrokePredictor()

        # Load current models
        logger.info("🔄 Loading current models...")
        models = self_learner.load_current_models()

        logger.info(f"✅ Loaded {len(models)} models: {list(models.keys())}")

        # Show initial system status
        status = self_learner.get_system_status()
        logger.info("📊 Initial System Status:")
        logger.info(f"   Learning Data: {status['learning_data_count']} samples")
        logger.info(f"   Feedback Data: {status['feedback_count']} entries")
        logger.info(f"   Model Versions: {status['model_versions']}")
        logger.info(f"   System Health: {status['system_health']}")

        # Simulate adding new patient data
        logger.info("\n📝 Adding New Patient Data for Learning...")

        # Sample patient data (simulated)
        new_patients = [
            {
                'age': 45,
                'gender': 'Male',
                'ever_married': 'Yes',
                'hypertension': 'No',
                'heart_disease': 'No',
                'avg_glucose_level': 85,
                'bmi': 28,
                'work_type': 'Private',
                'residence_type': 'Urban',
                'smoking_status': 'never smoked'
            },
            {
                'age': 67,
                'gender': 'Female',
                'ever_married': 'Yes',
                'hypertension': 'Yes',
                'heart_disease': 'No',
                'avg_glucose_level': 140,
                'bmi': 32,
                'work_type': 'Self-employed',
                'residence_type': 'Rural',
                'smoking_status': 'formerly smoked'
            },
            {
                'age': 72,
                'gender': 'Male',
                'ever_married': 'Yes',
                'hypertension': 'Yes',
                'heart_disease': 'Yes',
                'avg_glucose_level': 180,
                'bmi': 35,
                'work_type': 'Private',
                'residence_type': 'Urban',
                'smoking_status': 'smokes'
            }
        ]

        # Add patient data to learning system
        for i, patient in enumerate(new_patients, 1):
            success = self_learner.add_learning_data(patient)
            logger.info(f"   Added patient {i}: {'✅ Success' if success else '❌ Failed'}")

        # Simulate adding feedback from predictions
        logger.info("\n💬 Adding Prediction Feedback...")

        feedback_scenarios = [
            {
                'patient_data': new_patients[0],
                'predicted_risk': 0.12,
                'actual_outcome': 0,  # No stroke occurred
                'user_feedback': 'Model correctly predicted low risk'
            },
            {
                'patient_data': new_patients[1],
                'predicted_risk': 0.35,
                'actual_outcome': 1,  # Stroke did occur
                'user_feedback': 'Model correctly identified moderate risk'
            },
            {
                'patient_data': new_patients[2],
                'predicted_risk': 0.28,
                'actual_outcome': 0,  # No stroke occurred
                'user_feedback': 'Model was conservative, which is good for medical predictions'
            }
        ]

        for i, feedback in enumerate(feedback_scenarios, 1):
            success = self_learner.add_prediction_feedback(**feedback)
            logger.info(f"   Added feedback {i}: {'✅ Success' if success else '❌ Failed'}")

        # Show updated system status
        logger.info("\n📊 Updated System Status:")
        status = self_learner.get_system_status()
        logger.info(f"   Learning Data: {status['learning_data_count']} samples")
        logger.info(f"   Feedback Data: {status['feedback_count']} entries")
        logger.info(f"   System Health: {status['system_health']}")

        # Check if retraining is needed
        logger.info("\n🔍 Checking if retraining is needed...")

        # Get current performance (using dummy data for demo)
        dummy_performance = {
            'accuracy': 0.87,
            'f1': 0.75,
            'precision': 0.78,
            'recall': 0.72
        }

        needs_retraining = self_learner.should_retrain(dummy_performance)

        if needs_retraining:
            logger.info("   ✅ Retraining recommended!")

            # Run self-learning cycle
            logger.info("🚀 Running Self-Learning Cycle...")

            results = self_learner.run_self_learning_cycle(models)

            logger.info("   📈 Results:")
            logger.info(f"      Status: {results.get('status', 'unknown')}")
            logger.info(f"      Models Updated: {results.get('models_updated', False)}")
            logger.info(f"      Reason: {results.get('reason', 'unknown')}")

        else:
            logger.info("   ⏸️  No retraining needed at this time")

        # Show final status
        logger.info("\n🎯 Final System Status:")
        status = self_learner.get_system_status()
        for key, value in status.items():
            logger.info(f"   {key}: {value}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Self-Learning Demo Completed Successfully!")
        logger.info("\n🔬 Key Self-Learning Features Demonstrated:")
        logger.info("   ✅ Incremental Learning - Models learn from new data")
        logger.info("   ✅ Performance Monitoring - Track model accuracy over time")
        logger.info("   ✅ Drift Detection - Detect when models need retraining")
        logger.info("   ✅ User Feedback - Learn from prediction feedback")
        logger.info("   ✅ Automated Decision Making - System decides when to retrain")
        logger.info("   ✅ Model Versioning - Track improvements over time")
        logger.info("   ✅ Continuous Improvement - System gets better with more data")

        return True

    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("Make sure all required packages are installed:")
        logger.error("pip install xgboost scikit-learn imbalanced-learn pandas numpy")
        return False

    except Exception as e:
        logger.error(f"❌ Error in self-learning demo: {e}")
        return False

def show_learning_capabilities():
    """Show what the self-learning system can do."""
    logger.info("\n🧠 Self-Learning Capabilities:")
    logger.info("=" * 40)

    capabilities = [
        "📈 Incremental Learning - Update models with new patient data",
        "🔍 Performance Monitoring - Track accuracy, F1, precision, recall over time",
        "⚡ Drift Detection - Automatically detect when models need retraining",
        "💬 User Feedback - Learn from doctor/clinician feedback on predictions",
        "🔄 Automated Retraining - Retrain models when performance degrades",
        "📊 Model Versioning - Keep track of model improvements and changes",
        "🎯 Smart Thresholds - Know when enough data exists for meaningful retraining",
        "⚕️ Medical Focus - Conservative predictions with medical-grade reliability",
        "🔒 Data Privacy - Secure storage of learning data and feedback",
        "📱 API Integration - RESTful endpoints for easy integration"
    ]

    for capability in capabilities:
        logger.info(f"   {capability}")

    logger.info("\n🚀 How to Use Self-Learning:")
    logger.info("   1. Make predictions → API automatically adds data for learning")
    logger.info("   2. Provide feedback → Use /api/self-learning/add-feedback endpoint")
    logger.info("   3. Monitor status → Check /api/self-learning/status")
    logger.info("   4. Trigger learning → Call /api/self-learning/retrain")
    logger.info("   5. System improves → Models get better over time automatically")

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🤖 SELF-LEARNING STROKE PREDICTION SYSTEM DEMO")
    print("=" * 70)

    success = demonstrate_self_learning()

    if success:
        show_learning_capabilities()
        print("\n" + "=" * 70)
        print("✅ SELF-LEARNING DEMO COMPLETED SUCCESSFULLY!")
        print("🎯 The AI model can now learn and improve over time!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ SELF-LEARNING DEMO FAILED")
        print("🔧 Check the error messages above and fix any issues")
        print("=" * 70)
