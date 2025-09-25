#!/bin/bash

echo "🚀 NeuroPredict Railway Deployment Helper"
echo "=========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm i -g @railway/cli
fi

# Login to Railway
echo "🔐 Please login to Railway..."
railway login

# Check if models.zip exists
if [ ! -f "models.zip" ]; then
    echo "❌ models.zip not found!"
    echo "💡 Creating models.zip from your model files..."
    zip models.zip *.pkl
    echo "✅ models.zip created"
fi

echo "📦 Uploading models.zip to Railway..."
railway files upload models.zip

echo "🔄 Redeploying Railway service..."
railway up

echo "✅ Deployment complete!"
echo ""
echo "🔍 Check your Railway service logs:"
echo "railway logs"
echo ""
echo "🧪 Test your backend:"
echo "curl -X POST https://your-app.railway.app/api/predict -H 'Content-Type: application/json' -d '{\"age\":55,\"gender\":\"Male\"}'"
echo ""
echo "🎉 Your backend should now work with all models!"
