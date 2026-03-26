import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME: str = os.getenv("ATLAS_DATABASE_NAME", "testdb")
    SCHEMA_FILE_PATH: str = str(BASE_DIR / "full_schema.json")

    API_V1_PREFIX: str = "/api/v1"


settings = Settings()

