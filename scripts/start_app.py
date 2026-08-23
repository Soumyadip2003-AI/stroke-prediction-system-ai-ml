#!/usr/bin/env python3
"""
NeuroPredict Application Startup Script
======================================

This script starts both the Flask backend and serves the React frontend
for the NeuroPredict stroke risk assessment system.
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread
import signal

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'numpy', 
        'scikit_learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_models():
    """Check if trained models exist."""
    model_files = [
        'advanced_stroke_model_ensemble.pkl',
        'advanced_stroke_model_scaler.pkl',
        'advanced_stroke_model_features.pkl'
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("❌ Missing trained models:")
        for model_file in missing_models:
            print(f"   - {model_file}")
        print("\n💡 Train models first with:")
        print("   python train_and_save_model.py")
        return False
    
    return True

def start_backend():
    """Start the Flask backend server."""
    print("🚀 Starting Flask backend server...")
    try:
        subprocess.run([sys.executable, 'backend.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped.")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def start_frontend():
    """Start the frontend server."""
    print("🎨 Starting frontend server...")
    try:
        # Use Python's built-in HTTP server
        subprocess.run([sys.executable, '-m', 'http.server', '8000'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped.")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def open_browser():
    """Open browser after a short delay."""
    time.sleep(3)
    print("🌐 Opening browser...")
    webbrowser.open('http://localhost:8000')

def main():
    """Main startup function."""
    print("🧠 NeuroPredict - Advanced Stroke Risk Assessment")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found!")
    
    # Check models
    print("🤖 Checking trained models...")
    if not check_models():
        print("\n💡 Would you like to train models now? (y/n): ", end="")
        response = input().lower().strip()
        if response == 'y':
            print("🚀 Training models...")
            try:
                subprocess.run([sys.executable, 'train_and_save_model.py'], check=True)
                print("✅ Models trained successfully!")
            except subprocess.CalledProcessError:
                print("❌ Error training models. Please run manually:")
                print("   python train_and_save_model.py")
                sys.exit(1)
        else:
            sys.exit(1)
    print("✅ All models found!")
    
    print("\n🎯 Starting NeuroPredict application...")
    print("📡 Backend API: http://localhost:5000")
    print("🎨 Frontend: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop both servers")
    print("-" * 60)
    
    try:
        # Start backend in a separate thread
        backend_thread = Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(2)
        
        # Start frontend in a separate thread
        frontend_thread = Thread(target=start_frontend, daemon=True)
        frontend_thread.start()
        
        # Open browser
        browser_thread = Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down NeuroPredict...")
        print("👋 Thank you for using NeuroPredict!")

if __name__ == "__main__":
    main()
