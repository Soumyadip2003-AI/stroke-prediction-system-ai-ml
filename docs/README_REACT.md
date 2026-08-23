# 🧠 NeuroPredict - React + Tailwind CSS Frontend

[![React](https://img.shields.io/badge/React-18.0-blue)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9-blue)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC)](https://tailwindcss.com/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com/)

## 🎯 Overview

This is the **React + Tailwind CSS frontend** for NeuroPredict, featuring:

- **Modern React 18** with TypeScript
- **Tailwind CSS** for beautiful styling
- **Component-based Architecture** with reusable components
- **Real-time API Integration** with Flask backend
- **Responsive Design** for all devices
- **Beautiful Animations** and interactive elements

## 📁 Project Structure

```
neuropredict-frontend/
├── 📦 Package Files
│   ├── package.json              # Dependencies and scripts
│   ├── package-lock.json         # Locked dependency versions
│   ├── tailwind.config.js        # Tailwind CSS configuration
│   └── postcss.config.js         # PostCSS configuration
│
├── 🎨 Source Code
│   ├── src/
│   │   ├── components/           # React components
│   │   │   ├── Navigation.tsx    # Navigation component
│   │   │   ├── Hero.tsx          # Hero section
│   │   │   ├── Assessment.tsx    # Assessment form
│   │   │   ├── Results.tsx        # Results display
│   │   │   ├── Insights.tsx       # Model insights
│   │   │   ├── About.tsx         # About section
│   │   │   ├── Footer.tsx        # Footer component
│   │   │   └── LoadingOverlay.tsx # Loading animation
│   │   ├── App.tsx               # Main App component
│   │   ├── App.css               # App styles
│   │   ├── index.tsx             # React entry point
│   │   └── index.css             # Global styles with Tailwind
│
├── 🌐 Public Files
│   ├── public/
│   │   ├── index.html            # HTML template
│   │   ├── favicon.ico           # Site icon
│   │   └── manifest.json         # PWA manifest
│
└── 📋 Configuration
    ├── tsconfig.json             # TypeScript configuration
    └── README.md                 # React app documentation
```

## 🚀 Quick Start

### Prerequisites
- **Node.js 14+** and npm
- **Python 3.8+** with Flask backend
- **Trained ML models** (run `python train_and_save_model.py` first)

### 1. Install Dependencies

```bash
# Install React dependencies
cd neuropredict-frontend
npm install

# Install Python dependencies (in main directory)
pip install flask flask-cors pandas numpy scikit-learn xgboost lightgbm catboost joblib
```

### 2. Train Models (if not done already)

```bash
# Train the advanced ML models
python train_and_save_model.py
```

### 3. Start the Application

#### Option A: Automatic Startup (Recommended)
```bash
# Start both backend and frontend automatically
python start_react_app.py
```

#### Option B: Manual Startup
```bash
# Terminal 1: Start Flask backend
python backend.py

# Terminal 2: Start React frontend
cd neuropredict-frontend
npm start
```

### 4. Access the Application

- **React Frontend**: http://localhost:3000
- **Flask Backend API**: http://localhost:5000

## 🎨 React Components

### 1. **Navigation.tsx**
- Fixed navigation bar with glass morphism effect
- Smooth scrolling to sections
- Active section highlighting
- Responsive mobile menu

### 2. **Hero.tsx**
- Animated hero section with neural network visualization
- Gradient text animations
- Call-to-action buttons
- Statistics display

### 3. **Assessment.tsx**
- Multi-step assessment form
- Real-time form validation
- Interactive input controls
- API integration for predictions

### 4. **Results.tsx**
- Animated risk gauge display
- Model performance metrics
- Health analysis breakdown
- Personalized recommendations

### 5. **Insights.tsx**
- AI model insights and explanations
- Performance comparison charts
- Feature importance visualization

### 6. **About.tsx**
- Technology stack showcase
- Feature highlights
- Company information

### 7. **Footer.tsx**
- Contact information
- Quick links
- Social media links
- Legal disclaimers

### 8. **LoadingOverlay.tsx**
- Animated loading screen
- Progress indicators
- Brain icon animation

## 🎨 Tailwind CSS Features

### Custom Configuration
```javascript
// tailwind.config.js
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'gradient': 'gradient 15s ease infinite',
        'neural': 'neural 2s ease-in-out infinite',
      }
    }
  }
}
```

### Custom Animations
```css
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-20px); }
}

@keyframes gradient {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

@keyframes neural {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
```

### Glass Morphism Effects
```css
.glass-effect {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}
```

## 🔧 Development

### Available Scripts

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Eject from Create React App (not recommended)
npm run eject
```

### Development Features
- **Hot Reload**: Changes reflect immediately
- **TypeScript**: Type safety and IntelliSense
- **ESLint**: Code quality and consistency
- **Prettier**: Code formatting
- **Source Maps**: Easy debugging

### Environment Variables
```bash
# .env file (optional)
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENVIRONMENT=development
```

## 🎯 API Integration

### Backend Communication
```typescript
// Example API call
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(formData)
});

const result = await response.json();
```

### Error Handling
```typescript
try {
  const result = await fetch('/api/predict', options);
  if (!result.ok) {
    throw new Error(`HTTP error! status: ${result.status}`);
  }
  return await result.json();
} catch (error) {
  console.error('API Error:', error);
  // Fallback to simulated prediction
  return getSimulatedPrediction();
}
```

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 640px (sm)
- **Tablet**: 640px - 1024px (md, lg)
- **Desktop**: > 1024px (xl, 2xl)

### Mobile Features
- **Touch-friendly** interface
- **Swipe gestures** for navigation
- **Optimized forms** for mobile input
- **Responsive charts** and visualizations

## 🎨 Design System

### Color Palette
```css
Primary Colors:
- Blue: #3B82F6 (Primary actions)
- Purple: #8B5CF6 (Secondary actions)
- Pink: #EC4899 (Accent colors)

Health Colors:
- Green: #10B981 (Low risk)
- Yellow: #F59E0B (Moderate risk)
- Red: #EF4444 (High risk)

Background:
- Dark: #111827 (Main background)
- Glass: rgba(255, 255, 255, 0.1) (Glass effects)
```

### Typography
- **Font Family**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700, 800
- **Responsive Sizing**: text-sm to text-7xl

## 🚀 Production Deployment

### Build for Production
```bash
# Build the React app
cd neuropredict-frontend
npm run build

# The build folder contains optimized static files
```

### Deployment Options
- **Netlify**: Drag and drop build folder
- **Vercel**: Connect GitHub repository
- **AWS S3**: Upload build folder to S3 bucket
- **Heroku**: Use static buildpack

### Environment Configuration
```bash
# Production environment variables
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_ENVIRONMENT=production
```

## 🧪 Testing

### Component Testing
```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

### Manual Testing Checklist
- [ ] Form validation works
- [ ] API calls succeed
- [ ] Animations are smooth
- [ ] Responsive design works
- [ ] Error handling works
- [ ] Loading states display correctly

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm start
```

#### 2. Module Not Found
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 3. Tailwind Not Working
```bash
# Check if Tailwind is properly configured
npx tailwindcss --init

# Rebuild CSS
npm run build
```

#### 4. API Connection Issues
- Check if Flask backend is running on port 5000
- Verify CORS settings in backend.py
- Check browser console for errors

## 📊 Performance

### Optimization Features
- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Remove unused code
- **Minification**: Compress JavaScript and CSS
- **Image Optimization**: Automatic image optimization
- **Caching**: Browser caching for static assets

### Performance Metrics
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1
- **Time to Interactive**: < 3.0s

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- **TypeScript**: Use strict mode
- **React**: Functional components with hooks
- **Tailwind**: Utility-first CSS approach
- **ESLint**: Follow Airbnb style guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **React Team** for the amazing framework
- **Tailwind CSS** for the utility-first CSS framework
- **TypeScript Team** for type safety
- **Create React App** for the development setup

---

**🎉 Enjoy building with NeuroPredict React Frontend!** 🧠✨

**📞 Support**: For questions or issues, create an issue in the repository or contact the development team.
