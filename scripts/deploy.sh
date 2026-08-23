#!/bin/bash

echo "🚀 NeuroPredict Deployment Helper"
echo "================================="

echo ""
echo "Choose deployment option:"
echo "1) Frontend to Vercel + Backend to Railway (Recommended)"
echo "2) Full Stack to Vercel (Serverless)"
echo "3) View deployment guide"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "📋 Recommended Deployment: Frontend (Vercel) + Backend (Railway)"
        echo ""
        echo "Step 1: Deploy Frontend to Vercel"
        echo "Go to: https://vercel.com/dashboard"
        echo "1. Click 'New Project'"
        echo "2. Import your GitHub repo"
        echo "3. Set root directory to 'neuropredict-frontend'"
        echo "4. Deploy!"
        echo ""
        echo "Step 2: Deploy Backend to Railway"
        echo "Go to: https://railway.app/"
        echo "1. Connect your GitHub repo"
        echo "2. Deploy from root directory"
        echo "3. Set start command: 'gunicorn backend:app --bind 0.0.0.0:\$PORT'"
        echo ""
        echo "Step 3: Configure Environment Variables"
        echo "In Vercel dashboard:"
        echo "REACT_APP_API_BASE = https://your-railway-app.railway.app"
        echo ""
        read -p "Press Enter when ready to continue..."
        ;;
    2)
        echo "🔧 Full Stack Deployment to Vercel"
        echo ""
        echo "This uses serverless functions (simplified prediction model)"
        echo ""
        echo "Step 1: Deploy to Vercel"
        echo "1. Go to Vercel dashboard"
        echo "2. Import your GitHub repo"
        echo "3. Set root directory to 'neuropredict-frontend'"
        echo "4. Deploy!"
        echo ""
        echo "✅ API is already configured in /api/predict.js"
        echo ""
        read -p "Press Enter when ready to continue..."
        ;;
    3)
        echo "📖 Opening deployment guide..."
        if command -v open &> /dev/null; then
            open DEPLOYMENT_README.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open DEPLOYMENT_README.md
        else
            echo "Please read DEPLOYMENT_README.md for detailed instructions"
        fi
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1-3."
        exit 1
        ;;
esac

echo ""
echo "📚 For detailed instructions, read DEPLOYMENT_README.md"
echo "🎉 Happy deploying!"
