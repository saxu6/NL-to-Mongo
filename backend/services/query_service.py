import sys
import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

from OllamaCon.logic import convert_to_mongodb
from OllamaCon.test_with_schema import test_query_with_schema
from backend.config import settings

def load_schema():
    try:
        with open(settings.SCHEMA_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading schema: {e}")
        return None

def convert_nl_to_mongodb(query: str, use_schema: bool = True):
    if use_schema:
        schema = load_schema()
        if schema is None:
            raise Exception("Failed to load database schema")
        return test_query_with_schema(query, schema)
    else:
        return convert_to_mongodb(query)

def get_schema():
    return load_schema()

