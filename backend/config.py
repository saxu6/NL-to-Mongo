import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME: str = os.getenv("ATLAS_DATABASE_NAME", "testdb")
    SCHEMA_FILE_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "full_schema.json")
    
    API_V1_PREFIX: str = "/api/v1"
    
settings = Settings()

