"""
Simple schema discovery for MongoDB collections.
"""

import logging
from typing import Dict, List, Any
from config.database import get_database
from utils.logger import get_logger

logger = get_logger(__name__)

class SchemaDiscoverer:
    """Simple schema discovery for MongoDB collections."""
    
    def __init__(self):
        self.db = get_database()
    
    def discover_collections(self) -> List[str]:
        """Get list of collections in the database."""
        try:
            return self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Error discovering collections: {e}")
            return []
    
    def analyze_collection(self, collection_name: str) -> Dict[str, Any]:
        """Analyze a collection to understand its schema."""
        try:
            collection = self.db[collection_name]
            
            # Get sample documents
            sample_docs = list(collection.find().limit(10))
            
            if not sample_docs:
                return {
                    "collection": collection_name,
                    "fields": {},
                    "sample_count": 0
                }
            
            # Analyze fields
            fields = {}
            for doc in sample_docs:
                for key, value in doc.items():
                    if key not in fields:
                        fields[key] = {
                            "type": type(value).__name__,
                            "description": f"Field: {key}"
                        }
            
            return {
                "collection": collection_name,
                "fields": fields,
                "sample_count": len(sample_docs)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing collection {collection_name}: {e}")
            return {
                "collection": collection_name,
                "error": str(e)
            }