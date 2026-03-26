from functools import lru_cache
import json
from pathlib import Path

from parser import parse_query
from backend.config import settings


@lru_cache(maxsize=1)
def load_schema() -> dict:
    schema_path = Path(settings.SCHEMA_FILE_PATH)
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_nl_to_mongodb(query: str, use_schema: bool = True):
    # The parser requires schema context; keep `use_schema` for API compatibility.
    schema = load_schema()
    return parse_query(query, schema)


def get_schema():
    return load_schema()
