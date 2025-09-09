#!/usr/bin/env python3
"""
Test runner script for the MongoDB Query Translator system.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test(test_file):
    """Run a single test file."""
    print(f"🧪 Running {test_file}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            return True
        else:
            print(f"❌ {test_file} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {test_file} - ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running MongoDB Query Translator Test Suite...")
    print("=" * 60)
    
    tests_dir = project_root / "tests"
    test_files = [
        "test_basic.py"
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            if run_test(str(test_path)):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  {test_file} - NOT FOUND")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
