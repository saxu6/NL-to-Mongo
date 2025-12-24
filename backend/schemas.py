from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    use_schema: bool = True

class QueryResponse(BaseModel):
    mongodb_query: Dict[str, Any]
    collection: str
    filter: Dict[str, Any]
    projection: Dict[str, Any] = {}
    sort: Dict[str, Any] = {}
    limit: Optional[int] = None

class SchemaResponse(BaseModel):
    schema: Dict[str, Any]

