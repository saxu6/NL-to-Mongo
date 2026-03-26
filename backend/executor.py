"""Query executor — runs parsed MongoDB queries against the database (read-only)."""

from __future__ import annotations

from typing import Any, Dict

from bson import ObjectId
from datetime import datetime
from pymongo import MongoClient


MAX_RESULTS = 50


def _serialise(obj: Any) -> Any:
    """Make MongoDB documents JSON-serialisable."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def execute_query(client: MongoClient, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a parsed query result against MongoDB.

    Only read operations (find / aggregate) are supported.
    Write operations (update / delete) return a preview instead.
    """
    collection_path = parsed.get("collection", "")

    if "." not in collection_path:
        return {"error": f"Invalid collection path: {collection_path}", "results": [], "count": 0}

    db_name, coll_name = collection_path.split(".", 1)
    db = client[db_name]
    collection = db[coll_name]

    operation = parsed.get("operation")

    # Block write operations
    if operation in ("updateMany", "deleteMany"):
        return {
            "results": [],
            "count": 0,
            "preview": True,
            "message": f"Write operation '{operation}' detected. "
                       f"Showing generated query only (execution disabled).",
            "generated_query": {
                "operation": operation,
                "collection": collection_path,
                "filter": parsed.get("filter", {}),
                "update": parsed.get("update"),
            },
        }

    # Aggregate pipeline
    pipeline = parsed.get("pipeline")
    if pipeline:
        pipeline = list(pipeline)
        # Inject a $limit at the end if not already present
        has_limit = any("$limit" in stage for stage in pipeline)
        if not has_limit:
            pipeline.append({"$limit": MAX_RESULTS})

        cursor = collection.aggregate(pipeline)
        docs = [_serialise(doc) for doc in cursor]
        return {"results": docs, "count": len(docs)}

    # Standard find
    filt = parsed.get("filter", {})
    projection = parsed.get("projection") or None
    sort_spec = parsed.get("sort", {})
    limit = parsed.get("limit") or MAX_RESULTS

    # Cap to prevent huge result sets
    limit = min(limit, MAX_RESULTS)

    cursor = collection.find(filt, projection)

    if sort_spec:
        sort_list = [(k, v) for k, v in sort_spec.items()]
        cursor = cursor.sort(sort_list)

    cursor = cursor.limit(limit)

    docs = [_serialise(doc) for doc in cursor]
    return {"results": docs, "count": len(docs)}
