#!/usr/bin/env python3
"""
Production start script for the MongoDB Query Translator API server.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI server in production mode."""
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("🚀 Starting MongoDB Query Translator API Server...")
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔍 Health: http://{host}:{port}/health")
    print(f"👥 Workers: {workers}")
    print(f"📝 Log Level: {log_level}")
    print("-" * 50)
    
    # Production configuration
    if os.getenv("ENVIRONMENT") == "production":
        # Use gunicorn for production
        os.system(f"gunicorn api.main:app -w {workers} -k uvicorn.workers.UvicornWorker -b {host}:{port}")
    else:
        # Use uvicorn for development
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=False,
            log_level=log_level,
            access_log=True
        )

if __name__ == "__main__":
    main()
