"""
Simple database connection for MongoDB.
"""

import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def get_database():
    """Get MongoDB database connection."""
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    try:
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        # Get database name from URI or use default
        db_name = os.getenv("ATLAS_DATABASE_NAME", "default_database")
        return client[db_name]
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise

def get_collections():
    """Get list of collections in the database."""
    try:
        db = get_database()
        return db.list_collection_names()
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []