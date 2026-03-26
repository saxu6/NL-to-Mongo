from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import close_mongo_connection, connect_to_mongo
from backend.routes import chat, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_to_mongo()
    yield
    close_mongo_connection()


app = FastAPI(
    title="NL to MongoDB API",
    description="Natural Language to MongoDB Query Converter",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix=settings.API_V1_PREFIX, tags=["queries"])
app.include_router(chat.router, prefix=settings.API_V1_PREFIX, tags=["chat"])


@app.get("/health")
def health():
    return {"status": "healthy"}


# Serve frontend static files (catch-all mount; keep this last).
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
