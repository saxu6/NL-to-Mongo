import ollama
import sys

def check_service():
    print("Checking Ollama Service...")
    try:
        ollama.list()
        print("Ollama is running.")
        return True
    except Exception as e:
        print(f"Service Offline: {e}")
        return False

if __name__ == "__main__":
    if not check_service(): sys.exit(1)
