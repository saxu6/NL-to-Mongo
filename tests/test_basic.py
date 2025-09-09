"""
Simple test to verify the basic functionality works with Ollama.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import get_database
from services.query_generator import QueryGenerator
from services.validation_engine import ValidationEngine

def test_database_connection():
    """Test database connection."""
    try:
        db = get_database()
        collections = db.list_collection_names()
        print(f"✅ Database connected: {db.name}")
        print(f"📊 Collections: {collections}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_query_generator():
    """Test query generator with Ollama."""
    try:
        generator = QueryGenerator()
        print("✅ Query generator initialized with Ollama")
        print(f"🤖 Ollama URL: {generator.ollama_url}")
        print(f"🧠 Model: {generator.model_name}")
        return True
    except Exception as e:
        print(f"❌ Query generator failed: {e}")
        return False

def test_validation_engine():
    """Test validation engine."""
    try:
        validator = ValidationEngine()
        print("✅ Validation engine initialized")
        return True
    except Exception as e:
        print(f"❌ Validation engine failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection."""
    try:
        import requests
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection successful")
            return True
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running basic tests with Ollama...")
    print("-" * 50)
    
    tests = [
        test_database_connection,
        test_ollama_connection,
        test_query_generator,
        test_validation_engine
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        print("🤖 System ready with Ollama and gpt-oss:20b!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
