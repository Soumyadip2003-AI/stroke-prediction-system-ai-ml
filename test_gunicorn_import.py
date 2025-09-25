#!/usr/bin/env python3
"""
Test script to verify that models are loaded when backend is imported (gunicorn style)
"""
import sys
import os

def test_gunicorn_import():
    """Test importing backend module like gunicorn does."""
    print("🧪 Testing gunicorn-style import...")

    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    try:
        # Import backend like gunicorn does
        print("📥 Importing backend module...")
        from backend import app, models, models_loaded, logger

        print("✅ Backend imported successfully!")
        print(f"✅ Models loaded: {models_loaded}")
        print(f"✅ Available models: {list(models.keys())}")

        if models_loaded and len(models) > 0:
            print("🎉 SUCCESS: Backend is ready for gunicorn deployment!")
            return True
        else:
            print("❌ FAILURE: Models not loaded properly")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gunicorn_import()
    sys.exit(0 if success else 1)
