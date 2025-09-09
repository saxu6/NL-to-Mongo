#!/bin/bash

# Production deployment script for MongoDB Query Translator with Ollama

set -e

echo "🚀 Starting production deployment with Ollama..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env.example to .env and configure it."
    exit 1
fi

# Start Ollama first
echo "🤖 Starting Ollama service..."
docker-compose up -d ollama

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 30

# Setup Ollama with the model
echo "📥 Setting up Ollama with gpt-oss:20b model..."
./scripts/setup_ollama.sh

# Build and start the main application
echo "📦 Building Docker image..."
docker-compose build mongodb-query-translator

echo "🔄 Starting MongoDB Query Translator..."
docker-compose up -d mongodb-query-translator

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🔍 Performing health check..."
if python scripts/health_check.py; then
    echo "✅ Deployment successful!"
    echo "📍 API available at: http://localhost:8000"
    echo "📚 API docs at: http://localhost:8000/docs"
    echo "�� Ollama available at: http://localhost:11434"
    echo "🧠 Model: gpt-oss:20b"
else
    echo "❌ Health check failed. Check logs with: docker-compose logs"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
