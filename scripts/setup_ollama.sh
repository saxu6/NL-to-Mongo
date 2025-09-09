#!/bin/bash

# Setup script for Ollama with gpt-oss:20b model

echo "🚀 Setting up Ollama with gpt-oss:20b model..."

# Check if Ollama is running
echo "🔍 Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running. Please start Ollama first."
    echo "   Run: docker-compose up -d ollama"
    exit 1
fi

# Pull the gpt-oss:20b model
echo "📥 Pulling gpt-oss:20b model (this may take a while)..."
docker exec -it ollama ollama pull gpt-oss:20b

# Verify model is available
echo "🔍 Verifying model is available..."
if docker exec -it ollama ollama list | grep -q "gpt-oss:20b"; then
    echo "✅ gpt-oss:20b model is ready!"
else
    echo "❌ Failed to pull gpt-oss:20b model"
    exit 1
fi

echo "🎉 Ollama setup completed successfully!"
echo "📍 Ollama API: http://localhost:11434"
echo "🤖 Model: gpt-oss:20b"
echo "🚀 Ready to start the MongoDB Query Translator!"
