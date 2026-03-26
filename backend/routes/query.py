from fastapi import APIRouter, HTTPException

from backend.query_service import convert_nl_to_mongodb, get_schema
from backend.schemas import QueryRequest, QueryResponse, SchemaResponse

router = APIRouter()


def _to_query_response(result: dict) -> QueryResponse:
    return QueryResponse(
        mongodb_query=result,
        collection=result.get("collection", ""),
        operation=result.get("operation"),
        filter=result.get("filter", {}),
        projection=result.get("projection", {}),
        sort=result.get("sort", {}),
        limit=result.get("limit"),
        pipeline=result.get("pipeline"),
        update=result.get("update"),
        confidence=result.get("confidence"),
        warnings=result.get("warnings"),
    )


@router.post("/convert", response_model=QueryResponse)
async def convert_query(request: QueryRequest):
    try:
        result = convert_nl_to_mongodb(request.query, request.use_schema)

        if not result:
            raise HTTPException(
                status_code=500, detail="Failed to generate MongoDB query",
            )

        return _to_query_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema", response_model=SchemaResponse)
async def get_schema_endpoint():
    try:
        schema = get_schema()
        if schema is None:
            raise HTTPException(
                status_code=500, detail="Failed to load database schema",
            )
        return SchemaResponse(schema_data=schema)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
