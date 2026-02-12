import sys
import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

from parser import parse_query
from backend.config import settings


def load_schema():
    try:
        with open(settings.SCHEMA_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def convert_nl_to_mongodb(query: str, use_schema: bool = True):
    schema = load_schema()
    if schema is None:
        raise Exception("Failed to load database schema")

    return parse_query(query, schema)


def get_schema():
    return load_schema()
