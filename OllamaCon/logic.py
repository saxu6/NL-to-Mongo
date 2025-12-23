import ollama
import json
from models import get_models

def convert_to_mongodb(nl_query: str):
    models = get_models()
    model = models[0] if models else "llama3.1:8b"
    print(f"Converting NL to MongoDB -> '{nl_query}'")
    
    system_prompt = "You are a MongoDB expert. Respond ONLY with valid JSON."
    user_prompt = f"Convert to MongoDB JSON: {nl_query}. Required keys: collection, filter."

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json'
        )
        
        query_data = json.loads(response['message']['content'])
        print("Successfully generated logic:")
        print(json.dumps(query_data, indent=2))
        return query_data
        
    except Exception as e:
        print(f"Logic Error: {e}")

if __name__ == "__main__":
    convert_to_mongodb("Find users older than 25 in the 'customers' group")
