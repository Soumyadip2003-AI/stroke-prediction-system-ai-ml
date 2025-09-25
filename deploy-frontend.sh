#!/bin/bash

echo "🚀 Deploying NeuroPredict Frontend to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm i -g vercel
fi

# Navigate to frontend directory
cd neuropredict-frontend

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the application
echo "🔨 Building application..."
npm run build

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Frontend deployed successfully!"
echo "🎉 Your NeuroPredict app is now live on Vercel!"
