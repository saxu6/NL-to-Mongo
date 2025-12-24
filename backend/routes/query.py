from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse, SchemaResponse
from query_service import convert_nl_to_mongodb, get_schema

router = APIRouter()

@router.post("/convert", response_model=QueryResponse)
async def convert_query(request: QueryRequest):
    try:
        result = convert_nl_to_mongodb(request.query, request.use_schema)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate MongoDB query")
        
        return QueryResponse(
            mongodb_query=result,
            collection=result.get("collection", ""),
            filter=result.get("filter", {}),
            projection=result.get("projection", {}),
            sort=result.get("sort", {}),
            limit=result.get("limit")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema", response_model=SchemaResponse)
async def get_schema_endpoint():
    try:
        schema = get_schema()
        if schema is None:
            raise HTTPException(status_code=500, detail="Failed to load database schema")
        return SchemaResponse(schema=schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

