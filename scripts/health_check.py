#!/usr/bin/env python3
"""
Health check script for production monitoring.
"""

import requests
import sys
import time
from pathlib import Path

def check_health(base_url="http://localhost:8000", timeout=10):
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data.get('status', 'unknown')}")
            print(f"📊 Database: {'Connected' if data.get('database_connected') else 'Disconnected'}")
            return True
        else:
            print(f"❌ API Health Check Failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Unreachable: {e}")
        return False

def main():
    """Main health check function."""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🔍 Checking health at {base_url}")
    print("-" * 40)
    
    if check_health(base_url):
        print("🎉 System is healthy!")
        sys.exit(0)
    else:
        print("💥 System is unhealthy!")
        sys.exit(1)

if __name__ == "__main__":
    main()
