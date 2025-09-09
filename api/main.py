"""
Simple FastAPI application for MongoDB Query Translator.
"""

import os
import sys
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.query_generator import QueryGenerator
from services.validation_engine import ValidationEngine
from config.database import get_database, get_collections

# Initialize FastAPI app
app = FastAPI(
    title="MongoDB Query Translator",
    description="Convert natural language to MongoDB queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
query_generator = QueryGenerator()
validation_engine = ValidationEngine()

# Request models
class QueryRequest(BaseModel):
    query: str
    execute_query: bool = False

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MongoDB Query Translator API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_query": "/query/generate",
            "execute_query": "/query/execute",
            "database_info": "/database/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        db = get_database()
        collections = get_collections()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "collections_count": len(collections)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e)
        }

@app.post("/query/generate")
async def generate_query(request: QueryRequest):
    """Generate MongoDB query from natural language."""
    try:
        result = query_generator.generate_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/execute")
async def execute_query(request: QueryRequest):
    """Generate and execute MongoDB query."""
    try:
        result = validation_engine.validate_and_execute(
            request.query, 
            execute_query=request.execute_query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/info")
async def database_info():
    """Get database information."""
    try:
        db = get_database()
        collections = get_collections()
        
        return {
            "database_name": db.name,
            "collections": collections,
            "collections_count": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)