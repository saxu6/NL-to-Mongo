import ollama

def get_models():
    try:
        request = ollama.list()
        models = [m.model for m in request.models if "embed" not in m.model.lower()]
        to_use = [m for m in models if float(m.split(':')[1].split('b')[0]) <= 14]
        return to_use
    except Exception as e:
        print(f"Error: {e}")
        return []

def list_and_test():
    try:
        request = ollama.list()
        models = [m.model for m in request.models if "embed" not in m.model.lower()]

        if not models:
            print("No models found.")
            return

        try:
            to_use = [m for m in models if float(m.split(':')[1].split('b')[0]) <= 14]
            for model in to_use:
                print(model)
                response = ollama.generate(model=model, prompt="Say 'Ready to Code'")
                print(f"Response: {response['response'].strip()}\n")
        except Exception as e:
            print(f"Failed to run {model}: {e}\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_and_test()
