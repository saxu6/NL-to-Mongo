from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routes import query
from backend.database import connect_to_mongo, close_mongo_connection
from backend.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_to_mongo()
    yield
    close_mongo_connection()

app = FastAPI(
    title="NL to MongoDB API",
    description="Natural Language to MongoDB Query Converter",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix=settings.API_V1_PREFIX, tags=["queries"])

@app.get("/")
def root():
    return {"message": "NL to MongoDB API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

