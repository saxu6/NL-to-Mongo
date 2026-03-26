from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    use_schema: bool = True


class QueryResponse(BaseModel):
    mongodb_query: Dict[str, Any]
    collection: str
    operation: Optional[str] = None
    filter: Dict[str, Any]
    projection: Dict[str, Any] = Field(default_factory=dict)
    sort: Dict[str, Any] = Field(default_factory=dict)
    limit: Optional[int] = None
    pipeline: Optional[List[Dict[str, Any]]] = None
    update: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    warnings: Optional[List[str]] = None


class SchemaResponse(BaseModel):
    schema_data: Dict[str, Any]
