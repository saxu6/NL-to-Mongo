import ollama
import json
from models import get_models

def load_schema():
    try:
        with open("full_schema.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading schema: {e}")
        return None

def test_query_with_schema(nl_query: str, schema_data: dict):
    models = get_models()
    model = models[0] if models else "llama3.1:8b"
    
    schema_str = json.dumps(schema_data, indent=2)
    
    system_prompt = """You are a MongoDB expert. Use the provided database schema to convert natural language queries to MongoDB queries.
Respond ONLY with valid JSON in this format:
{
  "collection": "database.collection_name",
  "filter": {},
  "projection": {},
  "sort": {},
  "limit": null
}

Use MongoDB operators like $gt, $lt, $gte, $lte, $in, $regex, etc. in the filter field.
Match field names exactly as they appear in the schema."""
    
    user_prompt = f"""Database Schema:
{schema_str}

Natural Language Query: {nl_query}

Convert this query to a MongoDB query JSON using the schema above."""
    
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
        print(f"\nQuery: {nl_query}")
        print("Generated MongoDB Query:")
        print(json.dumps(query_data, indent=2))
        return query_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    schema = load_schema()
    if schema is None:
        print("Failed to load schema")
        exit(1)
    
    print("Schema loaded successfully")
    print(f"Found {len(schema)} database(s)\n")
    
    test_queries = [
        "Find all events where camera_id is 'CAM001'",
        "Get events with up_event_duration greater than 10",
        "Show events where initial_idle_time is null",
        "Find events with up_event_start less than 1000"
    ]
    
    for query in test_queries:
        test_query_with_schema(query, schema)
        print("\n" + "="*60 + "\n")

