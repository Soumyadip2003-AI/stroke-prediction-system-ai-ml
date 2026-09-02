# 🧠 NeuroPredict - Interactive AI Stroke Risk Assessment System

🅻🅸🅽🅺:- [https://stroke-prediction-system-ai-ml-xshe.vercel.app/](https://stroke-prediction-system-ai-ml-xshe.vercel.app)

![React Frontend](https://img.shields.io/badge/React-Frontend-blue)
![Flask Backend](https://img.shields.io/badge/Flask-Backend-green)
![Mobile Responsive](https://img.shields.io/badge/Mobile-Responsive-purple)
![Interactive Animations](https://img.shields.io/badge/Interactive-Animations-orange)


## 🎯 Overview

**NeuroPredict** is a revolutionary **AI-powered stroke risk assessment system** featuring an ultra-interactive React frontend with stunning animations and comprehensive mobile responsiveness. Built with cutting-edge machine learning models and advanced web technologies, it provides accurate stroke risk predictions with an engaging, futuristic user experience.

## 🧭 Repository management

This repo mixes a Flask backend, a React frontend, trained model artifacts, and several experimental training scripts. To make it easier to work with, the project now has a simple developer workflow:

- Use `npm run setup` for initial environment setup
- Use `npm run dev` to launch both backend and frontend together
- Use `npm run check` before shipping changes
- Keep model/training files separate from the app runtime files

### Common commands

```bash
# one-time environment setup
npm run setup

# start both services together
npm run dev

# backend only
npm run dev:backend

# frontend only
npm run dev:frontend

# validate repo health
npm run check
```

> The repo still contains research files and model artifacts alongside the app runtime, so the goal is to keep the main application flow predictable while preserving the experimentation scripts in place.

### ✨ What Makes NeuroPredict Special?

- 🎨 **Ultra-Interactive Animations**: 150+ particles responding to touch/mouse movements
- 📱 **100% Mobile Responsive**: Perfect experience on phones, tablets, and desktops
- 🧠 **One Tuned Model**: Histogram gradient boosting over 21 features, ROC-AUC 0.85 on a held-out split
- ⚡ **Real-time Interactions**: Dynamic neural network visualizations
- 🎯 **Touch-Optimized**: Full gesture support for mobile devices
- 🚀 **Performance Optimized**: 60fps animations on all devices

## ✨ Key Features

### 🎨 Ultra-Interactive Frontend
- **React with TypeScript**: Modern, type-safe frontend development
- **Tailwind CSS**: Mobile-first responsive design system
- **Advanced Animations**: 150+ particles with real-time interactions
- **Neural Network Visualization**: Dynamic SVG-based neural connections
- **Touch-Optimized**: Full gesture support for mobile devices
- **Responsive Design**: Perfect scaling from 320px to 4K displays

### 🤖 The Model
- **One Model**: Histogram gradient boosting, chosen by cross-validation over 41 tuned configurations
- **Balanced Class Weights**: The dataset is 4.87% positive; without this, "always no" wins
- **Fitted Decision Threshold**: 0.4796, chosen out-of-fold via Youden's J, not left at 0.5
- **21 Features**: Built by the same `preprocess_data` the API serves with, so training and serving cannot drift
- **ROC-AUC 0.8517, recall 0.82**: Measured on a held-out split, not on training data

### 📱 Mobile Excellence
- **100% Mobile Responsive**: Perfect experience on all devices
- **Touch Interactions**: Multi-zone particle interactions
- **Mobile Performance**: Optimized for 60fps on mobile hardware
- **PWA Ready**: Can be installed as a mobile app
- **Gesture Support**: Swipe, tap, and multi-touch interactions

### ⚡ Interactive Animations
- **Particle Physics**: Advanced multi-zone interaction system
- **Neural Network**: Real-time connections responding to mouse/touch
- **Dynamic Effects**: Color transitions, glows, and ripple animations
- **Performance Optimized**: Hardware-accelerated animations
- **Keyboard Interactions**: Spacebar triggers special effects

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** for the backend
- **Node.js 14+** for the frontend
- **Git** for version control

### 1. Clone the Repository
```bash
git clone https://github.com/Soumyadip2003-AI/stroke-prediction-system-ai-ml.git
cd stroke-prediction-system-ai-ml
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Train the model (writes stroke_prediction_model.pkl + model_metadata.json)
python ml/train_stroke_model.py
```

### 3. Frontend Setup
```bash
cd neuropredict-frontend

# Install Node.js dependencies
npm install

# Build the frontend
npm run build

# Start the development server
npm start
```

### 4. Start the Backend Server
```bash
# In a new terminal, activate the virtual environment
source venv/bin/activate

# Start the Flask API server
python backend.py
```

### 5. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5002
- **Interactive Features**: Move your mouse, touch the neural network, try mobile gestures!

## 📁 Project Structure

```
stroke-prediction-system-ai-ml/
├── 🧠 Model
│   ├── stroke_prediction_model.pkl     # The served model (calibrated gradient boosting)
│   ├── model_metadata.json             # Fitted threshold + measured held-out metrics
│   └── healthcare-dataset-stroke-data.csv   # Public stroke dataset, 5,110 rows
│
├── 🖥️  Backend (Flask API)
│   ├── backend.py                      # Gunicorn entrypoint (backend:app)
│   ├── app/server.py                   # Routes, preprocessing, validation, prediction
│   ├── requirements.txt                # Serving dependencies only
│   └── requirements-ml.txt             # Optional training extras
│
├── 🔬 ml/
│   ├── train_stroke_model.py           # Trains and saves the served model
│   ├── verify_model.py                 # Does the model actually predict? (out-of-fold gate)
│   └── test_model_sanity.py            # Fast regression guard, wired into `npm run check`
│
└── ⚛️  Frontend (React + TypeScript)
    └── neuropredict-frontend/
        ├── src/components/             # Navigation, Hero, Assessment, Results,
        │                               #   ResultsSkeleton, Insights, About, Footer
        ├── src/App.tsx                 # Section layout + scroll reveal
        ├── src/index.css               # Design tokens, reduced-motion, focus states
        └── public/index.html           # Metadata, Open Graph, JSON-LD
```

## 🔬 Modelling Notes

Everything below describes what the shipped model actually does. Techniques that
appear in the legacy scripts under `ml/` but are **not** used by the served model
are called out as such, because claiming them was how this project ended up
advertising a 0.982 ROC-AUC it never had.

### 🧠 What the served model does
- **One estimator**: histogram gradient boosting, selected by cross-validation against a
  tuned random forest and logistic regression
- **Balanced class weights**: the dataset is 4.87% positive, so without this the
  loss-minimising answer is "nobody has a stroke"
- **Fitted decision threshold**: 0.5308, chosen out-of-fold via Youden's J.
  Leaving it at 0.5 is what produced the original zero-recall model
- **Stratified k-fold cross-validation**: model selection and threshold fitting both
  happen out-of-fold; the reported metrics come from a held-out split touched by neither

### ⚡ Features
- **21 features**, built by calling the API's own `preprocess_data`, so training and
  serving cannot drift apart
- Age, glucose and BMI as-is, plus age squared and log-glucose, plus one-hot gender,
  marital status, work type, residence and smoking status
- **Not used**: BMI bands, glucose bands, interaction terms (age x BMI, age x glucose,
  BMI x glucose) and composite risk scores. All 19 were tested. They made the model
  slightly *worse* (0.8423 vs 0.8429 AUC), so they were dropped

### 🚫 Not used by the served model
Present in the legacy `ml/` scripts, deliberately absent from what ships:
- **Optuna hyperparameter search** - a fixed grid of 41 configurations was enough, and
  the accuracy curve is flat across most of it
- **SMOTE and other resampling** - balanced class weights achieve the same end without
  synthesising minority rows. Resampling before splitting also inflates CV scores, which
  is the most likely origin of the 0.96+ figures those scripts report
- **RFE / mutual-information feature selection** - 21 features on 249 positives does not
  need pruning
- **Isotonic calibration, stacking, GPU training** - no measurable benefit at this size

### 🎯 Calibrated probabilities
The number the API returns is a real estimated probability, not a ranking score.
Of people scored around 15%, about 15% went on to have a stroke.

Balanced class weights deliberately distort probabilities so the model takes a
4.87% positive class seriously. That is right for ranking and wrong for anything
shown to a person: uncalibrated, the model told people scoring 0.80+ that their
risk was 80% when the real rate in that group was 20.6%. Sigmoid calibration
fixes it at no cost to ranking:

| | Uncalibrated | Calibrated |
|---|---|---|
| Mean gap between shown % and reality | 36.93 pts | **0.83 pts** |
| Brier score | 0.1503 | **0.0416** |
| ROC-AUC | 0.8361 | **0.8486** |

Because the outputs are honest, they top out near 26% rather than 100%. Risk
bands are therefore multiples of the population base rate (4.87%),
which is also what a reader wants: not "24%" alone, but "five times average".

### 📉 The accuracy ceiling
Every architecture tested lands between 0.83 and 0.85 ROC-AUC. That flatness is an
information ceiling in the data, not a modelling failure. Age dominates. Moving past it
requires inputs this dataset does not contain: atrial fibrillation, actual blood-pressure
readings, prior stroke or TIA, cholesterol, family history.

## 🔬 Advanced Interactive Features

### 🎨 Ultra-Interactive Particle System
- **Multi-Zone Physics**: 3 interaction zones (close/medium/far proximity)
- **Touch-Responsive**: Full gesture support for mobile devices
- **150+ Particles**: Dynamic particle count based on device performance
- **Real-time Tracking**: Particles follow mouse/finger movements
- **Advanced Effects**: Grab, bubble, repulse, push, and remove interactions

### 🧠 Dynamic Neural Network Visualization
- **SVG-Based Graphics**: Scalable vector graphics for crisp visuals
- **Real-time Connections**: Lines appear dynamically based on mouse position
- **Touch Interactions**: Neural network responds to touch gestures
- **Color-Coded Neurons**: Each neuron has unique colors and animations
- **Performance Optimized**: Hardware-accelerated animations

### 📱 Mobile Excellence
- **Responsive Design**: Perfect scaling from 320px to 4K displays
- **Touch Optimization**: Proper touch target sizing (44px minimum)
- **Performance Scaling**: Adaptive particle counts for mobile devices
- **Gesture Support**: Swipe, tap, and multi-touch interactions
- **PWA Ready**: Can be installed as a mobile application

## 📈 Performance Metrics

### 🤖 Model Performance

Measured by `ml/train_stroke_model.py` on a held-out 20% split never seen during
training or threshold fitting. Regenerate with `npm run train:model`.

| Metric | Value | Why it is here |
|--------|-------|----------------|
| **ROC-AUC** | **0.8486** | The headline number. 0.50 is a coin flip. |
| **Recall** | **0.82** | Share of real strokes caught. This is what the model is tuned for. |
| **Precision** | **0.1444** | Roughly 1 flagged case in 7 is a real stroke. |
| **Average precision** | **0.2744** | Better than accuracy on a 4.9% positive class. |
| **Specificity** | **0.7500** | Share of non-strokes correctly cleared. |
| **Accuracy** | **0.7534** | Reported last, deliberately. See below. |

**Accuracy is lower than the baseline, and that is the point.** Only 4.87% of the
dataset had a stroke, so a model that always answers "no stroke" scores
95.13% accuracy while catching zero of them. An earlier version of
this project did exactly that: 95.13% accuracy, ROC-AUC 0.5605, recall 0.0000.
Models here are selected on ROC-AUC with balanced class weights, and the decision
threshold (0.0666) is fitted out-of-fold rather than left at 0.5.

**Architecture:** one histogram gradient boosting model over 21 features. A grid of 41
configurations showed a tuned single model reaches ROC-AUC 0.8425 against 0.8429 for a
three-model soft-voting ensemble. That 0.0004 gap sits far inside the +/- 0.019 fold
spread, so extra ensemble members buy nothing measurable while tripling inference cost
and failure surface. Ensemble size peaks at three and then *declines*: 1 model 0.8392,
3 models 0.8429, 4 models 0.8410, 6 models 0.8395. The nine-model claim this project
once made would have been worse than one.

### 🎨 Frontend Performance
- **60 FPS Animations**: Smooth performance on all devices
- **Mobile Optimized**: 80 particles on mobile vs 150 on desktop
- **Responsive Design**: Perfect scaling from 320px to 4K displays
- **Touch Interactions**: 100ms response time for touch events
- **PWA Ready**: Can be installed as a mobile app

### 📱 Mobile Responsiveness
- **Breakpoints**: Mobile (320px+), Tablet (768px+), Desktop (1024px+)
- **Touch Targets**: Minimum 44px for accessibility
- **Performance**: Adaptive particle counts based on device capabilities
- **Gestures**: Full support for swipe, tap, and multi-touch
- **Orientation**: Works in both portrait and landscape modes

## 🎯 Usage Examples

### 🔧 Backend API Usage
```python
import requests
import json

# Prepare patient data
patient_data = {
    "gender": "Female",
    "age": 65,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 200.5,
    "bmi": 28.7,
    "smoking_status": "formerly smoked"
}

# Make prediction request to Flask API
response = requests.post('http://localhost:5002/api/predict',
                        json=patient_data)

if response.status_code == 200:
    result = response.json()
    print(f"Stroke Risk: {result['risk_percentage']:.1f}%")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Confidence: {result['confidence']}")
```

### ⚛️ Frontend Integration
```javascript
// React component making API call
const getStrokePrediction = async (patientData) => {
    try {
        const response = await fetch('http://localhost:5002/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
    }
};
```

## 🔧 Advanced Configuration

### 🎨 Frontend Customization
Customize the interactive animations and responsive design:

```javascript
// particles.js configuration in App.tsx
const particleConfig = {
  particles: {
    number: {
      value: isMobile ? 80 : 150,  // Adaptive particle count
      density: { enable: true, value_area: isMobile ? 600 : 1000 }
    },
    interactivity: {
      events: {
        onhover: { enable: true, mode: ['grab', 'bubble', 'repulse'] },
        onclick: { enable: true, mode: ['push', 'remove', 'bubble'] }
      }
    }
  }
};
```

### 🖥️ Backend Configuration
Configure the Flask API server:

```python
# backend.py configuration
app.config['DEBUG'] = True
app.config['HOST'] = '0.0.0.0'
app.config['PORT'] = 5002

# CORS configuration for frontend access
CORS(app, origins=['http://localhost:3000'])
```

## 📊 Interactive Features Guide

### 🎮 Particle Interactions
- **Mouse/Finger Tracking**: Particles respond to cursor/touch movement
- **Multi-Zone Physics**: Different effects at different distances
- **Touch Gestures**: Tap, swipe, and multi-touch support
- **Click Effects**: Ripple animations and particle bursts
- **Keyboard Shortcuts**: Spacebar triggers special effects

### 🧠 Neural Network Interactions
- **Dynamic Connections**: SVG lines follow cursor position
- **Hover Effects**: Neurons scale and glow on interaction
- **Click Animations**: Special burst effects on clicks
- **Touch-Responsive**: Optimized for mobile touch interactions
- **Performance Scaling**: Adaptive complexity based on device

## 🎨 User Interface Features

### ⚛️ React Frontend with TypeScript
- **Modern Component Architecture**: Modular, reusable React components
- **TypeScript Integration**: Full type safety and IntelliSense support
- **Tailwind CSS**: Utility-first responsive styling
- **State Management**: React hooks for dynamic interactions
- **Real-time Updates**: Live API integration with Flask backend

### 🎯 Interactive Components
- **Hero Section**: Animated neural network with real-time interactions
- **Assessment Form**: Multi-step form with validation
- **Results Display**: Dynamic risk visualization with recommendations
- **Navigation**: Smooth scrolling navigation between sections
- **Loading States**: Beautiful loading animations and overlays

### 📱 Mobile-First Design
- **Responsive Layouts**: Perfect scaling across all device sizes
- **Touch Interactions**: Optimized for mobile gestures
- **PWA Capabilities**: Can be installed as a mobile app
- **Performance Optimized**: Hardware-accelerated animations
- **Accessibility**: WCAG-compliant interface design

## 🚀 Deployment

### 🖥️ Local Development
```bash
# 1. Backend Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the model (if needed)
python ml/train_stroke_model.py

# 3. Start Backend Server
python backend.py

# 4. Frontend Setup (in another terminal)
cd neuropredict-frontend
npm install
npm start

# 5. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:5002
```

### 🌐 Production Deployment Options

#### Option 1: Local Production
```bash
# Build frontend for production
cd neuropredict-frontend
npm run build

# Serve frontend with a static server
npm install -g serve
serve -s build -p 3000

# Run backend server
python backend.py
```

#### Option 2: Docker Deployment
```dockerfile
# Create Dockerfile for backend
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5002
CMD ["python", "backend.py"]

# Create Dockerfile for frontend
FROM node:16-alpine
WORKDIR /app
COPY neuropredict-frontend/package*.json ./
RUN npm install
COPY neuropredict-frontend/ .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

#### Option 3: Cloud Platforms
- **Vercel** (Frontend): Deploy React app with zero configuration
- **Heroku** (Backend): Deploy Flask API with Gunicorn
- **Netlify** (Frontend): Static site deployment
- **AWS/GCP/Azure**: Full cloud deployment with containers

## 🔬 Technical Architecture

### 🏗️ System Architecture
- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: Flask + Python + XGBoost + Optuna
- **Database**: No database required (stateless API)
- **ML Model**: A single histogram gradient boosting classifier (scikit-learn)
- **Animations**: CSS3 + SVG + Particles.js

### 📱 Mobile Optimization
- **Responsive Design**: Mobile-first approach with breakpoints
- **Touch Interactions**: Comprehensive gesture support
- **Performance Scaling**: Adaptive resource usage based on device
- **PWA Features**: App-like experience with offline capabilities

### 🔧 Development Stack
- **React 18**: Latest React with hooks and concurrent features
- **TypeScript**: Full type safety and IntelliSense
- **Flask 2.3**: Modern Python web framework
- **Scikit-learn**: Comprehensive ML library
- **Tailwind CSS**: Utility-first CSS framework

## 📚 Dependencies

### 🔧 Backend Dependencies
```txt
Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.1.1
numpy==1.25.2
scikit-learn==1.3.0
joblib==1.3.2
xgboost==1.7.6  # Ultimate model primary library
lightgbm==3.3.5  # Ensemble stacking component
catboost==1.2     # Ensemble stacking component
optuna==3.4.0     # Hyperparameter optimization
imbalanced-learn==0.11.0  # Advanced data balancing
```

### ⚛️ Frontend Dependencies
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "typescript": "^4.9.5",
  "tailwindcss": "^3.3.3",
  "@types/react": "^18.2.15",
  "@fortawesome/react-fontawesome": "^0.2.0",
  "particles.js": "^2.0.0"
}
```

### 🎨 Animation Libraries
- **Particles.js**: Interactive particle animations
- **CSS3 Animations**: Hardware-accelerated transitions
- **SVG Graphics**: Scalable vector neural network
- **Tailwind CSS**: Utility-first responsive design

## 🤝 Contributing

We welcome contributions to NeuroPredict! Here's how you can help:

### 🚀 Getting Started
1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/stroke-prediction-system-ai-ml.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes and test thoroughly
5. **Commit** with descriptive messages: `git commit -m "Add amazing feature"`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Submit** a Pull Request with detailed description

### 🎯 Development Guidelines
- Follow **mobile-first responsive design** principles
- Ensure **touch interactions** work properly
- Test **animations** at 60fps on various devices
- Maintain **TypeScript** type safety
- Write **descriptive commit messages**

### 🐛 Reporting Issues
- Use **GitHub Issues** for bug reports and feature requests
- Include **detailed reproduction steps**
- Specify **device/browser** information for UI issues
- Add **screenshots/videos** for visual problems


## 🙏 Acknowledgments

### 🧠 Machine Learning
- **Scikit-learn**: Comprehensive ML library
- **Healthcare Dataset**: Stroke prediction research data
- **Open Source Community**: ML and data science communities

### ⚛️ Frontend Development
- **React Team**: Modern JavaScript framework
- **Tailwind CSS**: Utility-first CSS framework
- **Particles.js**: Interactive animation library
- **TypeScript**: Type-safe JavaScript development

### 🎨 Design & UX
- **Font Awesome**: Icon library
- **Google Fonts**: Inter font family
- **Responsive Design**: Mobile-first principles

## 📞 Support & Contact

### 💬 Getting Help
- **📖 Documentation**: This comprehensive README
- **🐛 Issues**: [GitHub Issues](https://github.com/Soumyadipsarkar/stroke-prediction-system-ai-ml/issues)
- **📧 Email**: soumyadipsarkar.0202@gmail.com

### 🆘 Troubleshooting
- **Mobile Issues**: Check responsive design breakpoints
- **Performance**: Verify particle count optimization
- **API Errors**: Check Flask server logs
- **Build Issues**: Clear node_modules and reinstall

### 🎓 Learning Resources
- [React Documentation](https://react.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Tailwind CSS Guide](https://tailwindcss.com/docs)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)

---

## ⚠️ Important Disclaimers

### 🏥 Medical Disclaimer
**This tool is for educational and research purposes only.** The stroke risk predictions provided by NeuroPredict should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

### 🔬 Research Use
This system is intended for **research and educational purposes**. Held-out ROC-AUC is 0.85 with recall 0.82 and precision 0.15, meaning roughly 6 in every 7 flagged cases is a false alarm. It is trained on one public dataset of 5,110 records and has not been clinically validated. Real-world performance will differ.

### 👥 No Medical Advice
The predictions and recommendations provided are **not medical advice**. Users should not make health decisions based on this tool's output without consulting healthcare professionals.

### 📱 Beta Status
This is an **experimental system** with advanced interactive features. Some animations and interactions may not work perfectly on all devices or browsers.

---

## 🎉 Enjoy NeuroPredict!

Experience the future of **interactive AI-powered healthcare assessment** with stunning animations, mobile responsiveness, and cutting-edge machine learning! 🚀✨
