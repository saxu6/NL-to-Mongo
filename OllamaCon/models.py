import ollama

def list_and_test(model_name="llama3.1"):
    print(f"M1.2: Testing capabilities with {model_name}...")
    try:
        # List models
        models = ollama.list()['models']
        print(f"Found {len(models)} local models.")
        
        # Test basic prompt
        resp = ollama.generate(model=model_name, prompt="Say 'Ready for logic'")
        print(f"Response: {resp['response']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_and_test()
