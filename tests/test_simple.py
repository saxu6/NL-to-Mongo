"""
Simple test to verify the basic functionality works.
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
    """Test query generator."""
    try:
        generator = QueryGenerator()
        print("✅ Query generator initialized")
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

def main():
    """Run all tests."""
    print("🧪 Running simple tests...")
    print("-" * 40)
    
    tests = [
        test_database_connection,
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
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
