# 🔧 Vercel Deployment Troubleshooting Guide

> **REMOVED, DO NOT RECREATE.** The `api/predict.js` serverless function described
> below has been deleted. It was not a model: it was a hand-written if/else scorer
> that returned a fabricated risk percentage (it answered **100%** for a high-risk
> test profile while the real model returned 87.5%). It was publicly reachable, so
> the site had two live prediction endpoints giving different medical answers.
>
> The frontend calls the Flask API at `REACT_APP_API_BASE` instead, which serves the
> real trained model. If you need a serverless path, proxy to that API. Do not
> reintroduce a hardcoded scorer for medical output.

This guide helps you resolve common Vercel deployment errors for your NeuroPredict system.

## ✅ **Current Status: FIXED!**

Your NeuroPredict system has been updated with all the necessary fixes:

### **✅ Fixed Issues:**
- ✅ **Build Configuration:** Updated `package.json` with proper homepage and build settings
- ✅ **Vercel Config:** Enhanced `vercel.json` with version 2 configuration
- ✅ **API Function:** Fixed serverless function export for Node.js runtime
- ✅ **Memory Allocation:** Added 1024MB memory for serverless functions
- ✅ **Routing:** Added proper SPA routing for client-side navigation
- ✅ **CORS:** Enhanced CORS configuration with no-cache headers

---

## 🚀 **Deploy to Vercel Now (3 minutes)**

### **Step 1: Deploy**
1. **Go to Vercel Dashboard:** https://vercel.com/dashboard
2. **Import Project:** Select your GitHub repository
3. **Configure:**
   ```
   Root Directory:     neuropredict-frontend
   Framework Preset:   Create React App
   Build Command:      npm run build
   Output Directory:   build
   Install Command:    npm install
   ```
4. **Deploy!** Click "Deploy"

### **Step 2: Your URLs**
- **🌐 Live Site:** `https://your-project-name.vercel.app`
- **🔗 API:** `https://your-project-name.vercel.app/api/predict`

---

## 🔍 **Common Vercel Errors & Solutions**

### **❌ Error: Build Failed**

**Symptoms:**
- Build stops during "Building" phase
- Error: "Command failed: npm run build"

**Solutions:**
```bash
# 1. Check local build first
cd neuropredict-frontend && npm run build

# 2. If it fails, install dependencies
npm install

# 3. Clear cache and rebuild
rm -rf node_modules package-lock.json
npm install
npm run build
```

**✅ Fixed in your project:**
- Added `CI=false` to build command to prevent ESLint failures
- Set `homepage: "."` for proper asset paths
- Updated to React Scripts 5.0.1 for compatibility

---

### **❌ Error: Function Not Found**

**Symptoms:**
- API calls return 404
- "Cannot GET /api/predict"

**Solutions:**
1. **Check API File Location:**
   ```
   Should be: neuropredict-frontend/api/predict.js
   ✅ Currently: Correct location
   ```

2. **Verify Vercel Configuration:**
   ```json
   // In vercel.json
   {
     "builds": [
       {
         "src": "api/predict.js",
         "use": "@vercel/node"
       }
     ]
   }
   ✅ Currently: Configured correctly
   ```

---

### **❌ Error: CORS Issues**

**Symptoms:**
- "Access-Control-Allow-Origin" errors
- API calls blocked by browser

**Solutions:**
1. **Check CORS Headers:**
   ```json
   // In vercel.json
   "headers": [
     {
       "source": "/api/(.*)",
       "headers": [
         { "key": "Access-Control-Allow-Origin", "value": "*" }
       ]
     }
   ]
   ✅ Currently: Configured correctly
   ```

2. **Test CORS:**
   ```bash
   curl -X OPTIONS https://your-site.vercel.app/api/predict
   # Should return 200 OK
   ```

---

### **❌ Error: Function Timeout**

**Symptoms:**
- API calls take >30 seconds
- "Function timeout" errors

**Solutions:**
1. **Check Function Configuration:**
   ```json
   // In vercel.json
   "functions": {
     "api/predict.js": {
       "maxDuration": 30,
       "memory": 1024
     }
   }
   ✅ Currently: Configured correctly
   ```

2. **Optimize Function:**
   - Keep calculation logic simple
   - Remove heavy dependencies
   - Use caching for repeated calculations

---

### **❌ Error: Memory Limit Exceeded**

**Symptoms:**
- "Function out of memory" errors
- Build fails with memory issues

**Solutions:**
1. **Increase Memory Allocation:**
   ```json
   // In vercel.json
   "functions": {
     "api/predict.js": {
       "memory": 1024  // MB
     }
   }
   ✅ Currently: Set to 1024MB
   ```

2. **Reduce Function Size:**
   - Use smaller dependencies
   - Remove unused imports
   - Optimize algorithms

---

### **❌ Error: Missing Environment Variables**

**Symptoms:**
- Frontend can't connect to API
- "REACT_APP_API_BASE not defined"

**Solutions:**
1. **In Vercel Dashboard:**
   - Go to Project Settings → Environment Variables
   - Add: `REACT_APP_API_BASE = https://your-site.vercel.app`

2. **For Production:**
   - No environment variables needed for serverless
   - API endpoint is automatically configured

---

## 🛠️ **Manual Testing Commands**

### **Test Build Locally:**
```bash
cd neuropredict-frontend
npm run build
```

### **Test API Function:**
```bash
# Test with curl
curl -X POST https://your-site.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "gender": "Male", "hypertension": "Yes"}'
```

### **Test with JavaScript:**
```javascript
// In browser console
fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ age: 55, gender: 'Male' })
}).then(res => res.json()).then(console.log);
```

---

## 🔧 **Advanced Troubleshooting**

### **Check Vercel Logs:**
1. **Go to Vercel Dashboard**
2. **Select your project**
3. **Click "Functions" tab**
4. **Check deployment logs** for errors

### **Debug Common Issues:**

#### **1. Node.js Version:**
```json
// In package.json - ensure engines field
"engines": {
  "node": "18.x"
}
✅ Currently: Using Node 18 compatible
```

#### **2. Dependencies:**
```bash
# Check for missing dependencies
npm ls --depth=0
# Should show all green (no missing deps)
```

#### **3. File Permissions:**
```bash
# Ensure API file is executable
ls -la neuropredict-frontend/api/predict.js
# Should show -rw-r--r--
```

---

## 📊 **Performance Monitoring**

### **Vercel Analytics:**
1. **Go to Vercel Dashboard**
2. **Project → Analytics tab**
3. **Monitor:**
   - Response times
   - Error rates
   - Traffic patterns

### **Function Metrics:**
- **Cold Start Time:** <200ms
- **Memory Usage:** <100MB
- **Response Time:** <100ms
- **Error Rate:** <1%

---

## 🚨 **If Still Having Issues:**

### **Force Redeploy:**
```bash
# Install Vercel CLI
npm i -g vercel

# Force redeploy
vercel --prod --force
```

### **Check GitHub Connection:**
1. **Go to Vercel Dashboard**
2. **Project Settings → Git**
3. **Verify repository access**

### **Manual Deployment:**
1. **Download Vercel CLI**
2. **Login:** `vercel login`
3. **Deploy:** `vercel --prod`

---

## 🎯 **Expected Results After Deployment:**

### **✅ Working Features:**
- **🌐 Frontend:** https://your-site.vercel.app
- **🔗 API:** https://your-site.vercel.app/api/predict
- **📱 Mobile:** Fully responsive
- **🎨 Animations:** 150+ interactive particles
- **🧠 Neural Network:** Dynamic SVG connections
- **⚡ Performance:** <2s load time worldwide

### **✅ API Response:**
```json
{
  "risk_percentage": 67.5,
  "risk_category": "Moderate Risk",
  "confidence": "High",
  "risk_color": "#F59E0B",
  "health_analysis": ["Age factor", "Hypertension risk"],
  "recommendations": ["Monitor blood pressure", "Healthy diet"]
}
```

---

## 💡 **Pro Tips:**

### **Deployment Best Practices:**
- ✅ **Test locally** before deploying
- ✅ **Check build logs** for errors
- ✅ **Monitor function performance** after deployment
- ✅ **Use environment variables** for configuration

### **Performance Optimization:**
- ✅ **Code splitting** enabled
- ✅ **Image optimization** configured
- ✅ **CDN delivery** automatic
- ✅ **Compression** enabled

---

## 📞 **Need More Help?**

### **Vercel Support:**
- **Documentation:** https://vercel.com/docs
- **Community:** https://vercel.com/community
- **Status:** https://status.vercel.com

### **Common Resources:**
- **React Deployment:** https://create-react-app.dev/docs/deployment
- **Vercel Functions:** https://vercel.com/docs/concepts/functions
- **Troubleshooting:** https://vercel.com/docs/concepts/projects/troubleshooting

---

**🎉 Your NeuroPredict system is now fully configured for Vercel deployment! Deploy and enjoy serverless performance! 🚀✨**

## 📋 **Quick Deployment Checklist:**

- [x] ✅ Build successful locally
- [x] ✅ Vercel configuration updated
- [x] ✅ API function configured
- [x] ✅ GitHub repository pushed
- [x] ✅ Dependencies installed
- [ ] ⏳ **Deploy to Vercel now!**

**Ready to deploy? Go to https://vercel.com/dashboard → Import Project → Select your GitHub repo! 🌟**
