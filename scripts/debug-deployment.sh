#!/bin/bash

echo "🔍 NeuroPredict Deployment Debugger"
echo "=================================="

echo ""
echo "Step 1: Checking Frontend Build..."
echo "----------------------------------"
cd neuropredict-frontend

if npm run build > /tmp/build.log 2>&1; then
    echo "✅ Frontend build successful!"
else
    echo "❌ Frontend build failed!"
    echo "Build errors:"
    cat /tmp/build.log
    exit 1
fi

echo ""
echo "Step 2: Checking API Configuration..."
echo "------------------------------------"
if [ -f "api/predict.js" ]; then
    echo "✅ Serverless API endpoint found"
else
    echo "❌ Serverless API endpoint missing"
fi

echo ""
echo "Step 3: Checking Environment Variables..."
echo "----------------------------------------"
if [ -f ".env" ]; then
    echo "✅ .env file found"
else
    echo "⚠️  No .env file found (this is normal for production)"
fi

echo ""
echo "Step 4: Checking Vercel Configuration..."
echo "----------------------------------------"
if [ -f "vercel.json" ]; then
    echo "✅ vercel.json configuration found"
    echo "Configuration:"
    cat vercel.json | jq . 2>/dev/null || cat vercel.json
else
    echo "❌ vercel.json missing!"
fi

echo ""
echo "Step 5: Checking Git Status..."
echo "------------------------------"
cd ..
if git status --porcelain | grep -q .; then
    echo "⚠️  There are uncommitted changes:"
    git status --porcelain
else
    echo "✅ Git repository is clean"
fi

echo ""
echo "Step 6: Checking Dependencies..."
echo "--------------------------------"
cd neuropredict-frontend
if npm list --depth=0 > /tmp/deps.log 2>&1; then
    echo "✅ Dependencies installed correctly"
else
    echo "❌ Dependency issues found"
fi

echo ""
echo "🎯 Deployment Recommendations:"
echo "-------------------------------"
echo "1. Push all changes to GitHub: git push origin main"
echo "2. Go to Vercel Dashboard: https://vercel.com/dashboard"
echo "3. Deploy from your GitHub repository"
echo "4. Set root directory to: neuropredict-frontend"
echo "5. Build command: npm run build"
echo "6. Output directory: build"

echo ""
echo "🔧 Quick Fix Commands:"
echo "----------------------"
echo "Fix unused variables:"
echo "cd neuropredict-frontend && npm run build 2>/dev/null"

echo ""
echo "📋 Manual Testing:"
echo "------------------"
echo "Test locally with: npm start"
echo "Test production build: npx serve build -l 3000"

echo ""
echo "If deployment still fails, check:"
echo "- Vercel deployment logs"
echo "- GitHub repository permissions"
echo "- Environment variables in Vercel dashboard"
