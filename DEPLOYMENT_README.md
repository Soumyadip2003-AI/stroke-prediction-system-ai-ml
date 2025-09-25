# 🚀 NeuroPredict Deployment Guide

This guide explains how to deploy NeuroPredict to Vercel and other platforms.

## 📋 Deployment Options

### Option 1: Frontend Only (Recommended)
Deploy just the React frontend to Vercel, with backend on Railway/Heroku.

### Option 2: Full Stack
Deploy everything to Vercel using serverless functions.

---

## 🛠️ Option 1: Frontend on Vercel + Backend on Railway/Heroku

### Step 1: Deploy Frontend to Vercel

#### Automatic Deployment (Recommended)
1. **Fork/Clone this repository to your GitHub**
2. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**
3. **Click "New Project"**
4. **Import your GitHub repository**
5. **Configure Project:**
   - **Root Directory**: `neuropredict-frontend`
   - **Framework Preset**: React
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
   - **Install Command**: `npm install`
6. **Add Environment Variables:**
   - `REACT_APP_API_BASE`: `https://your-backend-app.railway.app` (your Railway backend URL)
7. **Deploy!**

#### Manual Deployment
```bash
# Navigate to frontend directory
cd neuropredict-frontend

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Step 2: Deploy Backend to Railway

#### Railway Deployment
1. **Go to [Railway.app](https://railway.app/)**
2. **Connect your GitHub repository**
3. **Deploy from GitHub:**
   - **Root Directory**: `.` (root of repository)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend:app --bind 0.0.0.0:$PORT`
4. **Add Environment Variables:**
   - `PORT`: `5000`
5. **Deploy!**

#### Railway CLI Deployment
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Deploy
railway up
```

### Step 3: Update Frontend Configuration

After deployment, update your Vercel environment variables:

1. **Go to Vercel Dashboard → Your Project → Settings → Environment Variables**
2. **Set `REACT_APP_API_BASE`** to your Railway backend URL (e.g., `https://your-app.railway.app`)

---

## 🛠️ Option 2: Full Stack on Vercel (Serverless)

⚠️ **Note**: This option uses a simplified prediction model for demo purposes.

### Step 1: Deploy to Vercel

#### Automatic Deployment
1. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**
2. **Click "New Project"**
3. **Import your GitHub repository**
4. **Configure Project:**
   - **Root Directory**: `neuropredict-frontend`
   - **Framework Preset**: Create React App
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
5. **Deploy!**

### Step 2: Configure API Routes

The serverless API is already configured in `/api/predict.js`. This provides:
- ✅ CORS support
- ✅ Simplified risk calculation
- ✅ JSON response format matching your frontend

---

## 🌐 Production URLs

After deployment, your URLs will look like:
- **Frontend**: `https://your-project.vercel.app`
- **Backend**: `https://your-backend.railway.app`

---

## 🔧 Environment Configuration

### Development (Local)
```bash
# Frontend
cd neuropredict-frontend
npm start

# Backend (separate terminal)
python backend.py
```

### Production Environment Variables

#### Vercel (Frontend)
- `REACT_APP_API_BASE`: `https://your-backend.railway.app`

#### Railway (Backend)
- `PORT`: `5000`
- `FLASK_ENV`: `production`

---

## 📱 Testing Your Deployment

1. **Open your Vercel URL** (e.g., `https://your-project.vercel.app`)
2. **Fill out the assessment form**
3. **Verify the API call** works correctly
4. **Test on mobile devices** for responsive design

---

## 🔄 Continuous Deployment

### GitHub Integration
Both Vercel and Railway support automatic deployments:
- **Push to main branch** → Automatic deployment
- **Pull requests** → Preview deployments

### Manual Redeployment
```bash
# Vercel
vercel --prod

# Railway
railway up
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. API Connection Failed
- ✅ Check if backend is running
- ✅ Verify `REACT_APP_API_BASE` environment variable
- ✅ Check CORS configuration

#### 2. Build Failed
- ✅ Verify all dependencies are installed
- ✅ Check Node.js version (18+ recommended)
- ✅ Ensure build scripts are correct

#### 3. Models Not Loading
- ✅ Check if model files exist in backend
- ✅ Verify file paths in `backend.py`
- ✅ Check memory limits on deployment platform

---

## 📊 Performance Optimization

### Frontend Optimizations
- ✅ Image optimization with Next.js Image component
- ✅ Code splitting enabled
- ✅ CSS minification
- ✅ JavaScript minification

### Backend Optimizations
- ✅ Model caching
- ✅ Response compression
- ✅ Database connection pooling (if needed)

---

## 🔒 Security Considerations

### Production Security
- ✅ HTTPS enabled by default
- ✅ Environment variables for sensitive data
- ✅ CORS properly configured
- ✅ Input validation and sanitization

### API Security
- ✅ Rate limiting (consider implementing)
- ✅ API key authentication (optional)
- ✅ Request logging and monitoring

---

## 📈 Monitoring & Analytics

### Vercel Analytics
- ✅ Built-in analytics dashboard
- ✅ Performance monitoring
- ✅ Error tracking

### Railway Monitoring
- ✅ Application logs
- ✅ Performance metrics
- ✅ Health checks

---

## 💰 Cost Estimation

### Free Tier Limits
- **Vercel**: 100GB bandwidth/month, 100 hours/month
- **Railway**: $5/month credit, 512MB RAM

### Scaling Up
- **Vercel Pro**: $20/month (1TB bandwidth, 1000 hours)
- **Railway**: Pay-as-you-go starting at $5/month

---

## 🎉 Next Steps

1. **Deploy your frontend** to Vercel
2. **Deploy your backend** to Railway
3. **Test the complete application**
4. **Share your live NeuroPredict app!**

**Happy deploying! 🚀✨**
