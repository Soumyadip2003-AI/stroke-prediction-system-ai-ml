#!/usr/bin/env python3
"""
NeuroPredict React Application Startup Script
============================================

This script starts both the Flask backend and the React frontend
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
    # Check Python dependencies
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
        print("❌ Missing Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    # Check Node.js and npm
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js or npm not found.")
        print("💡 Install Node.js from: https://nodejs.org/")
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

def install_react_dependencies():
    """Install React dependencies."""
    print("📦 Installing React dependencies...")
    try:
        subprocess.run(['npm', 'install'], 
                      cwd='neuropredict-frontend', 
                      check=True)
        print("✅ React dependencies installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing React dependencies.")
        return False

def start_backend():
    """Start the Flask backend server."""
    print("🚀 Starting Flask backend server...")
    try:
        subprocess.run([sys.executable, 'backend.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped.")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def start_react():
    """Start the React development server."""
    print("🎨 Starting React development server...")
    try:
        subprocess.run(['npm', 'start'], 
                      cwd='neuropredict-frontend', 
                      check=True)
    except KeyboardInterrupt:
        print("\n🛑 React server stopped.")
    except Exception as e:
        print(f"❌ Error starting React: {e}")

def open_browser():
    """Open browser after a short delay."""
    time.sleep(5)
    print("🌐 Opening browser...")
    webbrowser.open('http://localhost:3000')

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
    
    # Install React dependencies
    if not install_react_dependencies():
        sys.exit(1)
    
    print("\n🎯 Starting NeuroPredict application...")
    print("📡 Backend API: http://localhost:5000")
    print("🎨 React Frontend: http://localhost:3000")
    print("🛑 Press Ctrl+C to stop both servers")
    print("-" * 60)
    
    try:
        # Start backend in a separate thread
        backend_thread = Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(2)
        
        # Start React in a separate thread
        react_thread = Thread(target=start_react, daemon=True)
        react_thread.start()
        
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
